/****************************************************************************
 *
 *    Copyright (c) 2020 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include "core/framework/compute_capability.h"
#include "core/graph/graph_utils.h"
#include "vsi_npu_execution_provider.h"
#include "vsi_npu_ort_interpreter.h"
#include "core/framework/kernel_registry.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

constexpr const char* VSI_NPU = "VsiNpu";

VsiNpuExecutionProvider::VsiNpuExecutionProvider(const VsiNpuExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kVsiNpuExecutionProvider}, device_id_(info.device_id) {
    AllocatorCreationInfo default_memory_info{
        [](int) {
            return std::make_unique<CPUAllocator>(
                OrtMemoryInfo(VSI_NPU, OrtAllocatorType::OrtDeviceAllocator));
    }};

    InsertAllocator(CreateAllocator(default_memory_info));

    AllocatorCreationInfo cpu_memory_info{
        [](int) {
            return std::make_unique<CPUAllocator>(
                OrtMemoryInfo(VSI_NPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
    }};

    InsertAllocator(CreateAllocator(cpu_memory_info));
}

VsiNpuExecutionProvider::~VsiNpuExecutionProvider() {}

static bool IsTypeSupported(const NodeArg* node_arg) {
    const auto* type_proto = node_arg->TypeAsProto();
    if (!type_proto) {
        return false;
    }

    switch (type_proto->tensor_type().elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
            return true;
        default:
            return false;
    }
}

static bool IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                            const Node* node,
                            std::string& reason) {
    /*
    1. Check input and output data types are supported.
    2. Check Op is supported
           2a. Check if Op is of known unsupported modes (edge cases). If yes return false right
    away.
    */

    // Check 1
    bool are_types_supported = true;

    node->ForEachDef(
        [&are_types_supported](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
            are_types_supported &= IsTypeSupported(&node_arg);
        });

    if (!are_types_supported) {
        reason = "data type not supported.";
        return false;
    }

    if (node->Domain() != "") {
        reason = "only support default domain.";
        return false;
    }

    if (VsiSupported(node->OpType())) {
        auto op_info = getVsiFunc(node->OpType());
        auto cb = op_info->GetCallbackInfo(node, &graph_viewer);
        if (cb == nullptr) {
            reason = "op version not supported.";
            return false;
        }
        if (cb->IsNodeSupported(graph_viewer, node, reason)) {
            reason = "OK";
            return true;
        } else {
            reason += "## partial op features not supported.";
            return false;
        }
    } else {
        reason = "op type not supported.";
        return false;
    }
}

static void AppendClusterToSubGraph(const std::vector<NodeIndex>& nodes,
                                    const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& outputs,
                                    std::vector<std::unique_ptr<ComputeCapability>>& result) {
    static size_t op_counter = 0;

    auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
    meta_def->name = "VsiNpuCustomOp_" + std::to_string(++op_counter);
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    meta_def->inputs = inputs;
    meta_def->outputs = outputs;

    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes = nodes;
    sub_graph->SetMetaDef(std::move(meta_def));
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
}

static std::vector<NodeIndex> GetUnsupportedNodeIndices(
    const GraphViewer& graph_viewer,
    /*out*/ std::unordered_set<std::string>& vsi_npu_required_initializers) {
    std::vector<NodeIndex> unsupported_nodes_idx;

    for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
        auto node = graph_viewer.GetNode(node_idx);
        std::string reason = "";
        if (IsNodeSupported(graph_viewer, node, reason)) {
            // Collect inputs that are initializers
            LOGS_DEFAULT(VERBOSE) << "node:" << node->OpType();
            node->ForEachDef(
                [&vsi_npu_required_initializers, &graph_viewer](
                    const onnxruntime::NodeArg& node_arg, bool is_input) {
                    if (is_input &&
                        graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                        vsi_npu_required_initializers.insert(node_arg.Name());
                        LOGS_DEFAULT(VERBOSE) << "input tensor:" << vsi_npu::PrintNode(node_arg);
                    }
                },
                true);
        } else {
            LOGS_DEFAULT(WARNING) << "unsupported node:" << node->OpType();
            unsupported_nodes_idx.push_back(node_idx);
        }
    }

    return unsupported_nodes_idx;
}

/**
 * Returns a vector clusters(or node_idx). For each unsupported node, the graph is split into 3
 * parts. supported_cluster + (UNsupported_node + rest_of_the_graph). This functions returns vector
 * of all supported_clusters by VsiNpu
 */
static std::vector<std::vector<NodeIndex>> GetPartitionedClusters(
    const std::vector<NodeIndex>& topological_order,
    const std::vector<NodeIndex>& unsupported_nodes) {
    std::vector<std::vector<NodeIndex>> vsi_npu_clusters;

    auto prev = topological_order.begin();

    for (const auto& unsup_node : unsupported_nodes) {
        auto it = std::find(prev, topological_order.end(), unsup_node);
        // Create a cluster vector[supported_node_idx, unsupported_node_idx) and append it to return
        // list.
        std::vector<NodeIndex> this_cluster{prev, it};
        if (!this_cluster.empty()) {
            vsi_npu_clusters.push_back(std::move(this_cluster));
        }
        // Point prev to node idx past this unsuported node.
        prev = ++it;
    }

    // Tail
    std::vector<NodeIndex> this_cluster{prev, topological_order.end()};
    if (!this_cluster.empty()) {
        vsi_npu_clusters.push_back(std::move(this_cluster));
    }

    return vsi_npu_clusters;
}

static void GetInputsOutputsOfCluster(
    const GraphViewer& graph_viewer,
    const std::vector<NodeIndex>& cluster,
    const std::unordered_set<std::string>& vsi_npu_required_initializers,
    /*out*/ std::vector<std::string>& cluster_inputs,
    /*out*/ std::vector<std::string>& cluster_outputs) {
    std::unordered_set<std::string> input_args;
    std::vector<std::string> ordered_input_args;
    std::unordered_set<std::string> output_args;
    std::unordered_set<std::string> external_output_args;

    for (const auto& node_idx : cluster) {
        const auto& node = graph_viewer.GetNode(node_idx);

        // Collect all inputs and outputs
        node->ForEachDef(
            [&input_args, &ordered_input_args, &output_args](const NodeArg& node_arg,
                                                             bool is_input) {
                if (is_input) {
                    if (!input_args.count(node_arg.Name())) {
                        ordered_input_args.push_back(node_arg.Name());
                    }
                    input_args.insert(node_arg.Name());
                } else {
                    output_args.insert(node_arg.Name());
                }
            },
            true);

        // Check if output of this node is used by nodes outside this_cluster. If yes add this to
        // cluster outputs
        for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
            const auto& ext_node = graph_viewer.GetNode((*it).Index());

            if (std::find(cluster.begin(), cluster.end(), ext_node->Index()) == cluster.end()) {
                // Node is external to this_cluster. Search through its inputs to find the output
                // that is generated by this_cluster.
                std::set<std::string> ext_node_inputs;
                ext_node->ForEachDef(
                    [&ext_node_inputs](const onnxruntime::NodeArg& arg, bool is_input) {
                        if (is_input) {
                            ext_node_inputs.insert(arg.Name());
                        }
                    },
                    true);

                for (const auto& out_def : node->OutputDefs()) {
                    if (ext_node_inputs.find(out_def->Name()) != ext_node_inputs.end()) {
                        external_output_args.insert(out_def->Name());
                    }
                }
            }
        }
    }

    // Extract initializers used by this_cluster.
    std::unordered_set<std::string> original_graph_inputs;
    for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
        original_graph_inputs.insert(node_arg->Name());
    }

    const auto& initializers = graph_viewer.GetAllInitializedTensors();
    std::vector<std::string> const_inputs;
    for (const auto& in_arg : ordered_input_args) {
        if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
            vsi_npu_required_initializers.count(in_arg)) {
            const_inputs.push_back(in_arg);
        }
    }

    for (const auto& in_arg : ordered_input_args) {
        if (!output_args.count(in_arg) &&
            !((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
              vsi_npu_required_initializers.count(in_arg))) {
            cluster_inputs.push_back(in_arg);
        }
    }

    for (const auto& in_arg : const_inputs) {
        cluster_inputs.push_back(in_arg);
    }

    std::copy(external_output_args.begin(),
              external_output_args.end(),
              std::back_inserter(cluster_outputs));
    for (const auto& node_arg : graph_viewer.GetOutputs()) {
        const auto& name = node_arg->Name();
        if (output_args.count(name) && !external_output_args.count(name)) {
            cluster_outputs.push_back(name);
        }
    }
}

std::vector<std::unique_ptr<ComputeCapability>> VsiNpuExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& kernel_registries) const {
    // ref: onnxruntime/core/providers/ngraph/ngraph_execution_provider.cc: 465
    ORT_UNUSED_PARAMETER(kernel_registries);

    std::vector<std::unique_ptr<ComputeCapability>> result;

    // TODO:(nivas) Handle If and Loop operators
    if (graph_viewer.IsSubgraph()) {
        return result;
    }

    // Need access to model_path_
    for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
        if (tensor.second->has_data_location()) {
            LOGS_DEFAULT(VERBOSE) << "location:" << tensor.second->data_location();
            if (tensor.second->data_location() ==
                ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
                LOGS_DEFAULT(WARNING) << "VsiNpu: Initializers with external data location are not "
                                         "currently supported";
                return result;
            }
        }
    }

    /* This is a list of initializers that nGraph considers as constants. Example weights, reshape
       shape etc.
       TODO: Support overridable initializers */
    std::unordered_set<std::string> vsi_npu_required_initializers;

    const auto unsupported_nodes =
        GetUnsupportedNodeIndices(graph_viewer, vsi_npu_required_initializers);

    // If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
    if (unsupported_nodes.empty()) {
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;

        // Fill inputs with names
        std::for_each(graph_viewer.GetInputs().begin(),
                      graph_viewer.GetInputs().end(),
                      [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

        /* In scenarios, when there are no inputs or all inputs being initializers,
             ConstantFolding optimization in onnxruntime pre-computes the value.*/
        if (inputs.empty()) {
            return result;
        }

        // Initializers need to be part of meta_def->inputs
        std::for_each(vsi_npu_required_initializers.begin(),
                      vsi_npu_required_initializers.end(),
                      [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

        // Fill outputs with names
        std::for_each(graph_viewer.GetOutputs().begin(),
                      graph_viewer.GetOutputs().end(),
                      [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

        // Create and add this graph to result.
        AppendClusterToSubGraph(graph_viewer.GetNodesInTopologicalOrder(), inputs, outputs, result);

    } else {  // unsupported_nodes_idx.empty()
        const auto vsi_npu_clusters =
            GetPartitionedClusters(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

        for (const auto& this_cluster : vsi_npu_clusters) {
            std::vector<std::string> cluster_inputs, cluster_outputs;
            GetInputsOutputsOfCluster(graph_viewer,
                                      this_cluster,
                                      vsi_npu_required_initializers,
                                      cluster_inputs,
                                      cluster_outputs);

            if (!cluster_inputs.empty()) {
                AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
            }
        }
    }

    return result;
}

void SetupNNRTGraph(const Node* node,
                    ModelShellPtr model,
                    const onnxruntime::GraphViewer* graph_viewer) {
    node->ForEachDef([](const onnxruntime::NodeArg& node_arg, bool is_input) {
        if (is_input) {
            LOGS_DEFAULT(VERBOSE) << "node input tensor:" << vsi_npu::PrintNode(node_arg);
        } else {
            LOGS_DEFAULT(VERBOSE) << "node output tensor:" << vsi_npu::PrintNode(node_arg);
        }
    });
    if (VsiSupported(node->OpType())) {
        auto op_info = getVsiFunc(node->OpType());
        auto cb = op_info->GetCallbackInfo(node, graph_viewer);
        if (cb != nullptr) {
            cb->Setup(node, model, graph_viewer);
        }
    }
}

Status ComputeStateFunc(FunctionState state,
                        const OrtApi* api,
                        OrtKernelContext* context,
                        onnxruntime::Node* fused_node) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);

    {
        auto input_num = ort.KernelContext_GetInputCount(context);
        LOGS_DEFAULT(VERBOSE) << "input_num:" << input_num;
        auto output_num = ort.KernelContext_GetOutputCount(context);
        LOGS_DEFAULT(VERBOSE) << "output_num:" << output_num;
    }

    const auto* func_body = fused_node->GetFunctionBody();
    if (!func_body) {
        return common::Status(
            common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    }
    const Graph& graph_body = func_body->Body();
    const onnxruntime::GraphViewer graph_viewer(graph_body);

    for (const auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
        auto node = graph_viewer.GetNode(node_index);
        auto op_info = getVsiFunc(node->OpType());
        auto cb = op_info->GetCallbackInfo(node, &graph_viewer);
        if (cb != nullptr) {
            cb->Compute(state, api, context, node_index);
        }
    }

    std::vector<uint32_t> in_operand_ids;
    for (auto input : model->GetGraphInputs()) {
        if (!input->is_initializer && input->shape.NumDimensions() != 0) {
            in_operand_ids.push_back(input->operand_id);
        }
    }

    std::vector<uint32_t> out_operand_Ids;
    for (auto output : model->GetGraphOutputs()) {
        if (output->shape.NumDimensions() != 0) {
            out_operand_Ids.push_back(output->operand_id);
        }
    }

    model->IdentifyInputsAndOutputs(in_operand_ids.data(),
                                    in_operand_ids.size(),
                                    out_operand_Ids.data(),
                                    out_operand_Ids.size());

    size_t graph_inputs_num = model->GetGraphInputs().size();
    for (size_t i = 0, j = 0; i < graph_inputs_num; i++) {
        if (!model->GetGraphInputs()[i]->is_initializer) {
            const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
            const auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
            const auto& tensor_shape = ort.GetTensorShape(tensor_info);
            LOGS_DEFAULT(VERBOSE) << "TensorBytes:" << vsi_npu::GetTensorBytes(ort, tensor_info);
            model->SetInput(j,
                            nullptr,
                            ort.GetTensorData<void>(input_tensor),
                            vsi_npu::GetTensorBytes(ort, tensor_info));
            ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
            j++;
        }
    }

    for (size_t i = 0; i < ort.KernelContext_GetOutputCount(context); i++) {
        auto shape = model->GetGraphOutputs()[i]->shape.GetDims();
        OrtValue* output_tensor =
            ort.KernelContext_GetOutput(context, i, shape.data(), shape.size());
        const auto tensor_info = ort.GetTensorTypeAndShape(output_tensor);
        model->SetOutput(i,
                         nullptr,
                         ort.GetTensorMutableData<void>(output_tensor),
                         vsi_npu::GetTensorBytes(ort, tensor_info));
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
    }

    model->Compute();

    return Status::OK();
}

Status VsiNpuExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
    for (const auto& fused_node : fused_nodes) {
        NodeComputeInfo compute_info;
        LOGS_DEFAULT(VERBOSE) << "fused_node:" << fused_node->OpType();

        const auto* func_body = fused_node->GetFunctionBody();
        if (!func_body) {
            return common::Status(
                common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
        }
        const Graph& graph_body = func_body->Body();
        const onnxruntime::GraphViewer graph_viewer(graph_body);
        ModelShellPtr model = std::make_shared<ModelShell>();
        model_list_.push_back(model);

        for (auto tensor : graph_viewer.GetInputsIncludingInitializers()) {
            LOGS_DEFAULT(VERBOSE) << "fused_node input init:" << vsi_npu::PrintNode(*tensor) << "#"
                                  << graph_viewer.IsConstantInitializer(tensor->Name(), true) << "#"
                                  << graph_utils::IsInitializer(graph_body, tensor->Name(), true);
            auto input = std::make_shared<VsiGraphTensorInfo>();
            input->name = tensor->Name();
            if (graph_utils::IsInitializer(graph_body, tensor->Name(), true)) {
                input->is_initializer = true;
            } else {
                input->is_initializer = false;
            }
            model->GetGraphInputs().push_back(input);
        }
        for (auto tensor : graph_viewer.GetOutputs()) {
            LOGS_DEFAULT(VERBOSE) << "fused_node output:" << vsi_npu::PrintNode(*tensor);
            auto output = std::make_shared<VsiGraphTensorInfo>();
            output->name = tensor->Name();
            output->is_initializer = false;
            model->GetGraphOutputs().push_back(output);
        }

        for (const auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
            auto node = graph_viewer.GetNode(node_index);
            LOGS_DEFAULT(VERBOSE) << "sub node:" << node->OpType();
            SetupNNRTGraph(node, model, &graph_viewer);
        }

        compute_info.create_state_func = [model](ComputeContext* /*context*/,
                                                 FunctionState* state) {
            *state = model.get();
            return 0;
        };
        compute_info.release_state_func = [](FunctionState /*state*/) {};
        compute_info.compute_func =
            [fused_node, this](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
                std::lock_guard<OrtMutex> lock(this->GetMutex());
                return ComputeStateFunc(state, api, context, fused_node);
            };

        node_compute_funcs.push_back(compute_info);
    }

    return Status::OK();
}

std::shared_ptr<KernelRegistry> VsiNpuExecutionProvider::GetKernelRegistry() const {
    static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
    return kernel_registry;
}

}  // namespace onnxruntime
