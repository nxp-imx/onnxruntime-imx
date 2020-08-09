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

#pragma once
#include "nnrt/execution.hpp"
#include "nnrt/types.hpp"
#include "vsi_npu_util.h"

namespace onnxruntime {

class ModelShell;
struct ModelInfo;

struct VsiGraphTensorInfo {
    std::string name;
    bool is_initializer;
    uint32_t operand_id;
    TensorShape shape;
    std::shared_ptr<uint8_t> initializer_data;
};

struct VsiComputeInfo {
    std::vector<uint32_t> operand_ids;
    std::vector<std::string> compute_input_names;
    nnrt::op::OperationPtr op;
    std::vector<std::string> backup_names;
};

using ModelShellPtr = std::shared_ptr<ModelShell>;

class ModelShell {
   public:
    explicit ModelShell();
    ~ModelShell(){};

    uint32_t AddOperand(const NodeArg* node, const onnxruntime::GraphViewer* graph_viewer);

    uint32_t AddOperand(const std::string& name) {
        uint32_t operandId{0};
        auto search = all_operand_ids_.find(name);
        if (search != all_operand_ids_.end()) {
            operandId = search->second;
        } else {
            nnrt::op::OperandPtr operand = local_model_->addOperand(nullptr, &operandId);
            all_operand_ids_.insert({name, operandId});
        }

        return operandId;
    };

    template <typename T>
    void GetInitializerAsParameters(const NodeArg* node,
                                    const onnxruntime::GraphViewer* graph_viewer,
                                    std::vector<T>& result) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
        graph_viewer->GetInitializedTensor(node->Name(), tensor_proto);
        if (tensor_proto) {
            auto shape = vsi_npu::GetTensorShape(*node);
            size_t elementCount = shape.Size();
            std::shared_ptr<uint8_t> unpackedTensor = vsi_npu::UnpackTensor(node, *tensor_proto);
            for (auto input : graph_inputs_) {
                if (input->name == node->Name()) {
                    input->initializer_data = unpackedTensor;
                }
            }
            if (std::is_same<T, int32_t>::value) {
                const int64_t* valueAddr = reinterpret_cast<const int64_t*>(unpackedTensor.get());
                for (size_t i = 0; i < elementCount; i++) {
                    result.push_back(static_cast<int32_t>(valueAddr[i]));
                }
            } else {
                const T* valueAddr = reinterpret_cast<const T*>(unpackedTensor.get());
                for (size_t i = 0; i < elementCount; i++) {
                    result.push_back(valueAddr[i]);
                }
            }
        }
    }

    std::vector<std::shared_ptr<VsiGraphTensorInfo>>& GetGraphInputs() { return graph_inputs_; };

    std::vector<std::shared_ptr<VsiGraphTensorInfo>>& GetGraphOutputs() { return graph_outputs_; };

    nnrt::ModelPtr& GetModelPtr() { return local_model_; }

    void CollectComputeInfo(const Node* node,
                            const onnxruntime::GraphViewer* graph_viewer,
                            std::vector<int32_t>& compute_input_index,
                            std::shared_ptr<VsiComputeInfo>& compute_info);

    std::shared_ptr<VsiComputeInfo> GetComputeInfo(NodeIndex node_id);

    std::vector<int32_t> GetComputeInputIds(std::vector<std::string>& compute_input_names);

    void ConstInputOprand(const OrtApi* api,
                          OrtKernelContext* context,
                          std::shared_ptr<VsiComputeInfo>& compute_info,
                          std::vector<int32_t>& compute_input_ids,
                          int32_t index);

    void IdentifyInputsAndOutputs(const uint32_t* inputs_ptr,
                                  uint32_t input_count,
                                  const uint32_t* outputs_ptr,
                                  uint32_t output_count);

    int SetInput(uint32_t index,
                 const nnrt::op::OperandPtr& operand_type,
                 const void* buffer,
                 size_t length);

    int SetOutput(uint32_t index,
                  const nnrt::op::OperandPtr& operand_type,
                  void* buffer,
                  size_t length);

    int Compute();

    nnrt::op::OperationPtr AddOperation(nnrt::op::OperationPtr new_operation,
                                        uint32_t* out_index = nullptr);

    nnrt::op::OperandPtr GetOperand(uint32_t id) {
        return local_model_->operand(id);
    }

   private:
    void AddOperandHelper(const NodeArg* node, nnrt::op::OperandPtr operand, uint32_t operandId,nnrt::OperandType type);

    nnrt::ModelPtr local_model_ = std::make_shared<nnrt::Model>();
    nnrt::CompilerUniquePtr compiler_;
    nnrt::ExecUniquePtr execution_ptr_ = nullptr;
    std::vector<std::shared_ptr<VsiGraphTensorInfo>> graph_inputs_;
    std::vector<std::shared_ptr<VsiGraphTensorInfo>> graph_outputs_;
    std::map<std::string, uint32_t> all_operand_ids_;
    std::map<NodeIndex, std::shared_ptr<VsiComputeInfo>> graph_compute_infos_;
};

}  // namespace onnxruntime