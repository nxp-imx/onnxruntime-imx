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
#include "vsi_npu_ort_interpreter.h"

namespace onnxruntime {

bool PoolIsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                         const Node* node,
                         std::string& reason);

template <typename T>
class VsiOpCallbackInfoPoolOp : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoPoolOp(){};
    nnrt::op::OperationPtr createOperationPtr() override { return std::make_shared<T>(); };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override {
        ProtoHelperNodeContext ctx(*node);
        OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

        auto pool = std::dynamic_pointer_cast<T>(op);
        std::vector<int32_t> pads;
        bool status = vsi_npu::GetAttrs<int32_t>(attrs, "pads", pads, false).IsOK();
        if (status) {
            for (int i = pads.size() / 2 - 1, j = 0; i >= 0; i--, j += 2) {
                pool->pad[j] = pads[i];
                pool->pad[j + 1] = pads[i + pads.size() / 2];
            }
        } else {
            std::vector<int32_t> vpads(4, 0);
            pool->pad = std::move(vpads);
        }

        std::string auto_pad;
        status = vsi_npu::GetAttr<std::string>(attrs, "auto_pad", &auto_pad).IsOK();
        nnrt::PadType pad_type = nnrt::PadType::AUTO;
        if (status) {
            pad_type = vsi_npu::GetPadType(auto_pad);
        }
        pool->padType = pad_type;

        // add stride
        std::vector<int32_t> strides;
        status = vsi_npu::GetAttrs<int32_t>(attrs, "strides", strides, true).IsOK();
        if (status) {
            pool->strides = std::move(strides);
        } else {
            pool->strides = std::move(std::vector<int32_t>{1, 1});
        }

        // add kernel_shape
        std::vector<int32_t> kernel_shape;
        status = vsi_npu::GetAttrs<int32_t>(attrs, "kernel_shape", kernel_shape, true).IsOK();
        ORT_ENFORCE(status);
        pool->ksize = std::move(kernel_shape);

        int32_t ceil_mode;
        status = vsi_npu::GetAttr<int32_t>(attrs, "ceil_mode", &ceil_mode).IsOK();
        if (status) {
            if (ceil_mode == 0) {
                pool->roundType = nnrt::Rounding::FLOOR;
            } else {
                pool->roundType = nnrt::Rounding::CEILING;
            }
        }
    };
    bool IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                         const Node* node,
                         std::string& reason) override {
        return PoolIsNodeSupported(graph_viewer, node, reason);
    };
};

template <typename T>
class VsiOpCallbackInfoGlobalPoolOp : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoGlobalPoolOp(){};
    nnrt::op::OperationPtr createOperationPtr() override { return std::make_shared<T>(); };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override {
        auto pool = std::dynamic_pointer_cast<T>(op);
        // add padding
        pool->pad[0] = 0;
        pool->pad[1] = 0;
        pool->pad[2] = 0;
        pool->pad[3] = 0;

        // add stride
        pool->strides[0] = 1;
        pool->strides[1] = 1;

        // add kernel_shape
        auto input_defs = node->InputDefs();
        auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
        const std::vector<int64_t>& dims = shape.GetDims();
        pool->ksize[0] = static_cast<int32_t>(dims[3]);
        pool->ksize[1] = static_cast<int32_t>(dims[2]);
    };
    bool IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                         const Node* node,
                         std::string& reason) override {
        return PoolIsNodeSupported(graph_viewer, node, reason);
    };
};

class VsiOpCallbackInfoMaxPool : public VsiOpCallbackInfoPoolOp<nnrt::op::MaxPool2DOperation> {
   public:
    VsiOpCallbackInfoMaxPool(){};
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;
};

class VsiOpCallbackInfoAveragePool
    : public VsiOpCallbackInfoPoolOp<nnrt::op::AveragePool2DOperation> {
   public:
    VsiOpCallbackInfoAveragePool(){};
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override;
};

class VsiOpCallbackInfoGlobalMaxPool
    : public VsiOpCallbackInfoGlobalPoolOp<nnrt::op::MaxPool2DOperation> {
   public:
    VsiOpCallbackInfoGlobalMaxPool(){};
};

class VsiOpCallbackInfoGlobalAveragePool
    : public VsiOpCallbackInfoGlobalPoolOp<nnrt::op::AveragePool2DOperation> {
   public:
    VsiOpCallbackInfoGlobalAveragePool(){};
};

#define MAP_OP_COMMON(name)                                                                    \
    class VsiOpInfo##name : public VsiOpInfo {                                                 \
       public:                                                                                 \
        VsiOpInfo##name() { SetupCallbackInfo(std::make_shared<VsiOpCallbackInfo##name>()); }; \
    };

MAP_OP_COMMON(MaxPool)
MAP_OP_COMMON(AveragePool)
MAP_OP_COMMON(GlobalMaxPool)
MAP_OP_COMMON(GlobalAveragePool)

}  // namespace onnxruntime