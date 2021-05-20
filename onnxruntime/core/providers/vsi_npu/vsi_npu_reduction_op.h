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
#pragma once
#include "vsi_npu_ort_interpreter.h"

namespace onnxruntime {

class VsiOpCallbackInfoArgMax : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoArgMax(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::ArgmaxOperation>();
    };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;
};

class VsiOpCallbackInfoReduceMean : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoReduceMean(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::ReduceMeanOperation>();
    };
    void SetupAttribute(nnrt::op::OperationPtr op,
                        const Node* node,
                        ModelShellPtr& model,
                        const onnxruntime::GraphViewer* graph_viewer) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer&, const Node*, std::string&) override;
};

#define MAP_OP_COMMON(name)                                                                    \
    class VsiOpInfo##name : public VsiOpInfo {                                                 \
       public:                                                                                 \
        VsiOpInfo##name() { SetupCallbackInfo(std::make_shared<VsiOpCallbackInfo##name>()); }; \
    };

MAP_OP_COMMON(ArgMax)
MAP_OP_COMMON(ReduceMean)

}  // namespace onnxruntime