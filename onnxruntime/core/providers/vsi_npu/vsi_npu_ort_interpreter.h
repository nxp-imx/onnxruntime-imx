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
#include "vsi_npu_model.h"

namespace onnxruntime {

class VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfo(){};
    virtual ~VsiOpCallbackInfo(){};

    virtual void SetupIO(nnrt::op::OperationPtr op,
                         const Node* node,
                         ModelShellPtr& model,
                         const onnxruntime::GraphViewer* graph_viewer);

    virtual void SetupAttribute(nnrt::op::OperationPtr op,
                                const Node* node,
                                ModelShellPtr& model,
                                const onnxruntime::GraphViewer* graph_viewer){};

    virtual nnrt::op::OperationPtr createOperationPtr() { return nullptr; };
    virtual void Setup(const Node* node,
                       ModelShellPtr& model,
                       const onnxruntime::GraphViewer* graph_viewer) {
        auto op = createOperationPtr();
        SetupIO(op, node, model, graph_viewer);
        SetupAttribute(op, node, model, graph_viewer);
        model->AddOperation(op, nullptr);
    };
    virtual bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&);
    virtual Status Compute(FunctionState state,
                           const OrtApi* api,
                           OrtKernelContext* context,
                           NodeIndex node_index) {
        return Status::OK();
    };

    void ConstInputOprands(FunctionState state,
                           const OrtApi* api,
                           OrtKernelContext* context,
                           NodeIndex node_index);

    int32_t version_start_{INT_MIN};
    int32_t version_end_{INT_MAX};
};

class VsiOpInfo {
   public:
    VsiOpInfo(){};
    void SetupCallbackInfo(std::shared_ptr<VsiOpCallbackInfo>&& call_back_info) {
        call_back_info_.push_back(call_back_info);
    }
    std::shared_ptr<VsiOpCallbackInfo> GetCallbackInfo(int32_t version) {
        for (auto cb : call_back_info_) {
            if (version >= cb->version_start_ && version <= cb->version_end_) {
                return cb;
            }
        }
        return nullptr;
    }
    std::shared_ptr<VsiOpCallbackInfo> GetCallbackInfo(
        const Node* node, const onnxruntime::GraphViewer* graph_viewer) {
        auto version_map = graph_viewer->DomainToVersionMap();
        auto version = version_map[node->Domain()];
        return GetCallbackInfo(version);
    }

   protected:
    std::vector<std::shared_ptr<VsiOpCallbackInfo>> call_back_info_;
};

class VsiOrtInterpreter : public nnrt::Interpreter {
   public:
    VsiOrtInterpreter();
    virtual ~VsiOrtInterpreter(){};

    const char* name() override { return "onnxruntime_Interpreter"; }

    int run(nnrt::Model* model, bool* modified) override;
};

bool VsiSupported(const std::string& opName);

std::shared_ptr<VsiOpInfo> getVsiFunc(const std::string& opName);

class VsiOpCallbackInfoRelu : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoRelu(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::ReluOperation>();
    };
};

class VsiOpCallbackInfoConv : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoConv(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::GroupedConv2DOperation>();
    };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;
};

class VsiOpCallbackInfoSoftmax : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoSoftmax(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::SoftmaxOperation>();
    };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;
};

class VsiOpCallbackInfoGemm : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoGemm(){};
    void Setup(const onnxruntime::Node* node,
               onnxruntime::ModelShellPtr& model,
               const onnxruntime::GraphViewer* graph_viewer) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;

   private:
    void AddTransposeOp(const onnxruntime::Node* node,
                        onnxruntime::ModelShellPtr& model,
                        std::vector<uint32_t> trans_operand_ids,
                        std::string trans_add_name);
    void AddAddOp(const onnxruntime::Node* node,
                  onnxruntime::ModelShellPtr& model,
                  std::vector<uint32_t> add_operand_ids,
                  std::string add_add_name);
    void AddMulOp(const onnxruntime::Node* node,
                  onnxruntime::ModelShellPtr& model,
                  std::vector<uint32_t> mul_operand_ids,
                  std::vector<std::string> mul_add_names,
                  uint32_t num);
    void AddMatmulOp(const onnxruntime::Node* node,
                     onnxruntime::ModelShellPtr& model,
                     std::vector<uint32_t> matmul_operand_ids);
};

class VsiOpCallbackInfoLRN : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoLRN(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::LocalResponseNormOperation>();
    };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override;
};

class VsiOpCallbackInfoLeakyRelu : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoLeakyRelu(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::LeakyReluOperation>();
    };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override;
};

class VsiOpCallbackInfoUpsample : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoUpsample(){};
    void Setup(const Node*, ModelShellPtr&, const onnxruntime::GraphViewer*) override;
    Status Compute(FunctionState state,
                   const OrtApi* api,
                   OrtKernelContext* context,
                   NodeIndex node_index) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;
};

class VsiOpCallbackInfoInstanceNormalization : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoInstanceNormalization(){};
    void Setup(const Node*, ModelShellPtr&, const onnxruntime::GraphViewer*) override;
    Status Compute(FunctionState state,
                   const OrtApi* api,
                   OrtKernelContext* context,
                   NodeIndex node_index) override;
};

class VsiOpCallbackInfoBatchNormalization : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoBatchNormalization(){};
    void Setup(const Node*, ModelShellPtr&, const onnxruntime::GraphViewer*) override;
    Status Compute(FunctionState state,
                   const OrtApi* api,
                   OrtKernelContext* context,
                   NodeIndex node_index) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;
};

#define MAP_OP_COMMON(name)                                                                    \
    class VsiOpInfo##name : public VsiOpInfo {                                                 \
       public:                                                                                 \
        VsiOpInfo##name() { SetupCallbackInfo(std::make_shared<VsiOpCallbackInfo##name>()); }; \
    };

MAP_OP_COMMON(Relu)
MAP_OP_COMMON(Conv)
MAP_OP_COMMON(Softmax)
MAP_OP_COMMON(Gemm)
MAP_OP_COMMON(LRN)
MAP_OP_COMMON(LeakyRelu)
MAP_OP_COMMON(Upsample)
MAP_OP_COMMON(InstanceNormalization)
MAP_OP_COMMON(BatchNormalization)

}  // namespace onnxruntime
