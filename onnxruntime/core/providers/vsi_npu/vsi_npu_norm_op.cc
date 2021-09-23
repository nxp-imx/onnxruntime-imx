/****************************************************************************
 *
 *    Copyright (c) 2021 Vivante Corporation
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
#include "vsi_npu_norm_op.h"

namespace onnxruntime {

void VsiOpCallbackInfoLRN::SetupAttribute(nnrt::op::OperationPtr op,
                                          const Node* node,
                                          ModelShellPtr& model,
                                          const onnxruntime::GraphViewer* graph_viewer) {
    auto lrn = std::dynamic_pointer_cast<nnrt::op::LocalResponseNormOperation>(op);
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    int32_t size;
    bool status = vsi_npu::GetAttr<int32_t>(attrs, "size", &size).IsOK();
    ORT_ENFORCE(status);

    lrn->radius = (size - 1) / 2;

    float bias;
    status = vsi_npu::GetAttr<float>(attrs, "bias", &bias).IsOK();
    ORT_ENFORCE(status);
    lrn->bias = bias;

    float alpha;
    status = vsi_npu::GetAttr<float>(attrs, "alpha", &alpha).IsOK();
    ORT_ENFORCE(status);
    lrn->scale = alpha / size;

    float beta;
    status = vsi_npu::GetAttr<float>(attrs, "beta", &beta).IsOK();
    ORT_ENFORCE(status);
    lrn->exponent = beta;

    lrn->axis = 1;
}

void VsiOpCallbackInfoInstanceNormalization::Setup(const onnxruntime::Node* node,
                                                   onnxruntime::ModelShellPtr& model,
                                                   const onnxruntime::GraphViewer* graph_viewer) {
    auto input_defs = node->InputDefs();
    std::vector<uint32_t> in_operand_ids;

    auto output_defs = node->OutputDefs();
    std::vector<uint32_t> out_operand_ids;

    const int32_t kIndexScale = 2;
    const int32_t kIndexB = 1;
    std::vector<int32_t> compute_input_index({kIndexScale, kIndexB});

    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() == 4) {
        uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
        in_operand_ids.push_back(input_operand_id);

        uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
        out_operand_ids.push_back(output_operand_id);
    } else if (shape.NumDimensions() == 3) {
        // reshape input
        uint32_t input_pre_reshape_operand_id = model->AddOperand(input_defs[0], graph_viewer);
        uint32_t output_pre_reshape_operand_id =
            model->AddOperand(input_defs[0]->Name() + "@@reshape_0");
        auto operand_pre_reshape = model->GetOperand(output_pre_reshape_operand_id);
        operand_pre_reshape->type = vsi_npu::convertToOperandType(input_defs[0]->Type());
        vsi_npu::SetTensorDims(*input_defs[0], operand_pre_reshape->dimensions);
        operand_pre_reshape->dimensions.push_back(1);

        std::vector<uint32_t> in_operand_ids_pre_reshape;
        std::vector<uint32_t> out_operand_ids_pre_reshape;
        in_operand_ids_pre_reshape.push_back(input_pre_reshape_operand_id);
        out_operand_ids_pre_reshape.push_back(output_pre_reshape_operand_id);

        auto pre_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        pre_reshape->setInputs(in_operand_ids_pre_reshape.data(),
                               in_operand_ids_pre_reshape.size());
        pre_reshape->setOutputs(out_operand_ids_pre_reshape.data(),
                                out_operand_ids_pre_reshape.size());

        for (auto dim : operand_pre_reshape->dimensions) {
            pre_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(pre_reshape, nullptr);

        in_operand_ids.push_back(output_pre_reshape_operand_id);

        // reshape output
        auto output_post_reshape_operand_id = model->AddOperand(output_defs[0], graph_viewer);
        auto input_post_reshape_operand_id =
            model->AddOperand(output_defs[0]->Name() + "@@reshape_1");
        auto operand_post_reshape = model->GetOperand(input_post_reshape_operand_id);
        operand_post_reshape->type = vsi_npu::convertToOperandType(output_defs[0]->Type());
        vsi_npu::SetTensorDims(*output_defs[0], operand_post_reshape->dimensions);
        operand_post_reshape->dimensions.push_back(1);

        std::vector<uint32_t> in_operand_ids_post_reshape;
        std::vector<uint32_t> out_operand_ids_post_reshape;
        in_operand_ids_post_reshape.push_back(input_post_reshape_operand_id);
        out_operand_ids_post_reshape.push_back(output_post_reshape_operand_id);

        auto post_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        post_reshape->setInputs(in_operand_ids_post_reshape.data(),
                                in_operand_ids_post_reshape.size());
        post_reshape->setOutputs(out_operand_ids_post_reshape.data(),
                                 out_operand_ids_post_reshape.size());

        for (auto dim : operand_post_reshape->dimensions) {
            post_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(post_reshape, nullptr);

        out_operand_ids.push_back(input_post_reshape_operand_id);
    } else if (shape.NumDimensions() > 4) {
        // reshape input
        uint32_t input_pre_reshape_operand_id = model->AddOperand(input_defs[0], graph_viewer);
        uint32_t output_pre_reshape_operand_id =
            model->AddOperand(input_defs[0]->Name() + "@@reshape_0");
        auto operand_pre_reshape = model->GetOperand(output_pre_reshape_operand_id);
        operand_pre_reshape->type = vsi_npu::convertToOperandType(input_defs[0]->Type());
        const std::vector<int64_t>& dims_pre_reshape = shape.GetDims();
        for (size_t i = 0; i < 3; i++) {
            operand_pre_reshape->dimensions.push_back(static_cast<uint32_t>(dims_pre_reshape[i]));
        }
        uint32_t val = 1;
        for (size_t i = 3; i < dims_pre_reshape.size(); i++) {
            val *= static_cast<uint32_t>(dims_pre_reshape[i]);
        }
        operand_pre_reshape->dimensions.push_back(val);

        std::vector<uint32_t> in_operand_ids_pre_reshape;
        std::vector<uint32_t> out_operand_ids_pre_reshape;
        in_operand_ids_pre_reshape.push_back(input_pre_reshape_operand_id);
        out_operand_ids_pre_reshape.push_back(output_pre_reshape_operand_id);

        auto pre_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        pre_reshape->setInputs(in_operand_ids_pre_reshape.data(),
                               in_operand_ids_pre_reshape.size());
        pre_reshape->setOutputs(out_operand_ids_pre_reshape.data(),
                                out_operand_ids_pre_reshape.size());

        for (auto dim : operand_pre_reshape->dimensions) {
            pre_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(pre_reshape, nullptr);

        in_operand_ids.push_back(output_pre_reshape_operand_id);

        // reshape output
        auto output_post_reshape_operand_id = model->AddOperand(output_defs[0], graph_viewer);
        auto input_post_reshape_operand_id =
            model->AddOperand(output_defs[0]->Name() + "@@reshape_1");
        auto operand_post_reshape = model->GetOperand(input_post_reshape_operand_id);
        operand_post_reshape->type = vsi_npu::convertToOperandType(output_defs[0]->Type());
        shape = vsi_npu::GetTensorShape(*output_defs[0]);
        const std::vector<int64_t>& dims_post_reshape = shape.GetDims();
        for (size_t i = 0; i < 3; i++) {
            operand_post_reshape->dimensions.push_back(static_cast<uint32_t>(dims_post_reshape[i]));
        }
        val = 1;
        for (size_t i = 3; i < dims_post_reshape.size(); i++) {
            val *= static_cast<uint32_t>(dims_post_reshape[i]);
        }
        operand_post_reshape->dimensions.push_back(val);

        std::vector<uint32_t> in_operand_ids_post_reshape;
        std::vector<uint32_t> out_operand_ids_post_reshape;
        in_operand_ids_post_reshape.push_back(input_post_reshape_operand_id);
        out_operand_ids_post_reshape.push_back(output_post_reshape_operand_id);

        auto post_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        post_reshape->setInputs(in_operand_ids_post_reshape.data(),
                                in_operand_ids_post_reshape.size());
        post_reshape->setOutputs(out_operand_ids_post_reshape.data(),
                                 out_operand_ids_post_reshape.size());

        for (auto dim : operand_post_reshape->dimensions) {
            post_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(post_reshape, nullptr);

        out_operand_ids.push_back(input_post_reshape_operand_id);
    }

    for (auto index : compute_input_index) {
        uint32_t input_operand_id = model->AddOperand(input_defs[index], graph_viewer);
        in_operand_ids.push_back(input_operand_id);
    }

    auto op = std::make_shared<nnrt::op::InstanceNormOperation<float>>();
    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    float epsilon;
    bool status = vsi_npu::GetAttr<float>(attrs, "epsilon", &epsilon).IsOK();
    ORT_ENFORCE(status);
    op->eps = epsilon;

    model->AddOperation(op, nullptr);

    auto compute_info = std::make_shared<VsiComputeInfo>();
    model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
}

Status VsiOpCallbackInfoInstanceNormalization::Compute(FunctionState state,
                                                       const OrtApi* api,
                                                       OrtKernelContext* context,
                                                       NodeIndex node_index) {
    ConstInputOprands(state, api, context, node_index);
    return Status::OK();
}

void VsiOpCallbackInfoBatchNormalization::Setup(const onnxruntime::Node* node,
                                                onnxruntime::ModelShellPtr& model,
                                                const onnxruntime::GraphViewer* graph_viewer) {
    auto input_defs = node->InputDefs();
    std::vector<uint32_t> in_operand_ids;
    const int32_t kIndexScale = 3;
    const int32_t kIndexB = 4;
    const int32_t kIndexMean = 1;
    const int32_t kIndexVar = 2;
    std::vector<int32_t> compute_input_index({kIndexScale, kIndexB, kIndexMean, kIndexVar});
    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    for (auto index : compute_input_index) {
        uint32_t input_operand_id = model->AddOperand(input_defs[index], graph_viewer);
        in_operand_ids.push_back(input_operand_id);
    }

    auto output_defs = node->OutputDefs();
    std::vector<uint32_t> out_operand_ids;
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    auto op = std::make_shared<nnrt::op::BatchNormalization>();
    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    float epsilon;
    bool status = vsi_npu::GetAttr<float>(attrs, "epsilon", &epsilon).IsOK();
    ORT_ENFORCE(status);
    op->eps = epsilon;

    model->AddOperation(op, nullptr);

    auto compute_info = std::make_shared<VsiComputeInfo>();
    model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
}

Status VsiOpCallbackInfoBatchNormalization::Compute(FunctionState state,
                                                    const OrtApi* api,
                                                    OrtKernelContext* context,
                                                    NodeIndex node_index) {
    ConstInputOprands(state, api, context, node_index);
    return Status::OK();
}

bool VsiOpCallbackInfoBatchNormalization::IsNodeSupported(
    const onnxruntime::GraphViewer& graph_viewer, const Node* node, std::string& reason) {
    auto input_defs = node->InputDefs();
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() != 4) {
        reason += "## Only support BN2D now.";
        return false;
    }
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}

}  // namespace onnxruntime