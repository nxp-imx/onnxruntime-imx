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

#include "vsi_npu_ort_interpreter.h"

namespace onnxruntime {

ModelShell::ModelShell() {
    compiler_ = std::make_unique<nnrt::Compilation>(local_model_.get());
    compiler_->setInterpreter(new VsiOrtInterpreter());
}

void ModelShell::AddOperandHelper(const NodeArg* node,
                                  nnrt::op::OperandPtr operand,
                                  uint32_t operandId,
                                  nnrt::OperandType type) {
    auto shape = vsi_npu::GetTensorShape(*node);
    if (type == nnrt::OperandType::NONE) {
        type = vsi_npu::convertToOperandType(node->Type());
    }
    operand->type = type;

    const std::vector<int64_t>& dims = shape.GetDims();
    for (auto dim : dims) {
        uint32_t value = static_cast<uint32_t>(dim);
        operand->dimensions.push_back(value);
    }
    for (auto input : graph_inputs_) {
        if (input->name == node->Name()) {
            input->operand_id = operandId;
            input->shape = shape;
            return;
        }
    }
    for (auto output : graph_outputs_) {
        if (output->name == node->Name()) {
            output->operand_id = operandId;
            output->shape = shape;
            return;
        }
    }
}

uint32_t ModelShell::AddOperand(const NodeArg* node, const onnxruntime::GraphViewer* graph_viewer) {
    uint32_t operandId{0};
    auto search = all_operand_ids_.find(node->Name());
    if (search != all_operand_ids_.end()) {
        operandId = search->second;
    } else {
        nnrt::op::OperandPtr operand = local_model_->addOperand(nullptr, &operandId);
        AddOperandHelper(node, operand, operandId,nnrt::OperandType::NONE);
        const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
        graph_viewer->GetInitializedTensor(node->Name(), tensor_proto);
        if (tensor_proto) {
            std::shared_ptr<uint8_t> unpackedTensor = vsi_npu::UnpackTensor(node, *tensor_proto);
            for (auto input : graph_inputs_) {
                if (input->name == node->Name()) {
                    input->initializer_data = unpackedTensor;
                    break;
                }
            }
            const void* valueAddr = reinterpret_cast<const void*>(unpackedTensor.get());
            local_model_->setOperandValue(operandId, valueAddr, operand->bytes());
        }
        all_operand_ids_.insert({node->Name(), operandId});
    }

    return operandId;
}

void ModelShell::CollectComputeInfo(const Node* node,
                                    const onnxruntime::GraphViewer* graph_viewer,
                                    std::vector<int32_t>& compute_input_index,
                                    std::shared_ptr<VsiComputeInfo>& compute_info) {
    auto input_defs = node->InputDefs();

    // save the input tensor name for attribute
    for (auto index : compute_input_index) {
        auto name = input_defs[index]->Name();
        for (auto input : graph_inputs_) {
            if (input->name == name && input->is_initializer == false) {
                input->is_initializer = true;
                compute_info->compute_input_names.push_back(name);
                break;
            }
        }
    }

    // get the node_index for current node
    for (const auto& node_index : graph_viewer->GetNodesInTopologicalOrder()) {
        auto node_compare = graph_viewer->GetNode(node_index);
        if (node_compare == node) {
            graph_compute_infos_.insert({node_index, compute_info});
            break;
        }
    }
}

std::shared_ptr<VsiComputeInfo> ModelShell::GetComputeInfo(NodeIndex node_id) {
    auto search = graph_compute_infos_.find(node_id);
    if (search != graph_compute_infos_.end()) {
        return search->second;
    }
    return nullptr;
}

std::vector<int32_t> ModelShell::GetComputeInputIds(std::vector<std::string>& compute_input_names) {
    std::vector<int32_t> compute_input_ids;
    auto input_info = GetGraphInputs();
    for (size_t i = 0; i < compute_input_names.size(); i++) {
        for (size_t j = 0; j < input_info.size(); j++) {
            if (input_info[j]->name == compute_input_names[i]) {
                compute_input_ids.push_back(j);
                break;
            }
        }
    }
    return compute_input_ids;
}

void ModelShell::ConstInputOprand(const OrtApi* api,
                                  OrtKernelContext* context,
                                  std::shared_ptr<VsiComputeInfo>& compute_info,
                                  std::vector<int32_t>& compute_input_ids,
                                  int32_t index) {
    Ort::CustomOpApi ort{*api};
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, compute_input_ids[index]);
    const auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    const void* valueAddr = ort.GetTensorData<void>(input_tensor);
    auto operandId = all_operand_ids_[compute_info->compute_input_names[index]];
    local_model_->setOperandValue(operandId, valueAddr, vsi_npu::GetTensorBytes(ort, tensor_info));
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
};

void ModelShell::IdentifyInputsAndOutputs(const uint32_t* inputs_ptr,
                                          uint32_t input_count,
                                          const uint32_t* outputs_ptr,
                                          uint32_t output_count) {
    local_model_->identifyInputsAndOutputs(inputs_ptr, input_count, outputs_ptr, output_count);
    local_model_->finish();
    if (execution_ptr_ == nullptr) {
        execution_ptr_ = std::make_unique<nnrt::Execution>(compiler_.get());
    }
}

int ModelShell::SetInput(uint32_t index,
                         const nnrt::op::OperandPtr& operand_type,
                         const void* buffer,
                         size_t length) {
    return execution_ptr_->setInput(index, operand_type, buffer, length);
}

int ModelShell::SetOutput(uint32_t index,
                          const nnrt::op::OperandPtr& operand_type,
                          void* buffer,
                          size_t length) {
    return execution_ptr_->setOutput(index, operand_type, buffer, length);
}

int ModelShell::Compute() {
    local_model_->relax(false);
    auto errCode = execution_ptr_->compute();
    if (0 != errCode) {
        // assert(false);
        LOGS_DEFAULT(WARNING) << "Execution Model failed";
    }
    return errCode;
}

nnrt::op::OperationPtr ModelShell::AddOperation(nnrt::op::OperationPtr new_operation,
                                                uint32_t* out_index) {
    return local_model_->addOperation(new_operation, out_index);
}

}  // namespace onnxruntime