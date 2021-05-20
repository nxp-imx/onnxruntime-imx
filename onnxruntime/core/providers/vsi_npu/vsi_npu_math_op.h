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

class VsiOpCallbackInfoAbs : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoAbs(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::AbsOperation>();
    };
};

class VsiOpCallbackInfoSqrt : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoSqrt(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::SqrtOperation>();
    };
};

class VsiOpCallbackInfoAdd : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoAdd(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::AddOperation>();
    };
};

class VsiOpCallbackInfoSub : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoSub(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::SubOperation>();
    };
};

class VsiOpCallbackInfoMul : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoMul(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::MulOperation>();
    };
};

class VsiOpCallbackInfoDiv : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoDiv(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::DivOperation>();
    };
};

class VsiOpCallbackInfoSum : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoSum(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::AddNOperation>();
    };
};

class VsiOpCallbackInfoMatMul : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoMatMul(){};
    void Setup(const Node*, ModelShellPtr&, const onnxruntime::GraphViewer*) override;
    bool IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                         const Node* node,
                         std::string& reason) override;
};

class VsiOpCallbackInfoLog : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoLog(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::LogOperation>();
    };
};

class VsiOpCallbackInfoPow : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoPow(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::PowOperation>();
    };
};

class VsiOpCallbackInfoExp : public VsiOpCallbackInfo {
   public:
    VsiOpCallbackInfoExp(){};
    nnrt::op::OperationPtr createOperationPtr() override {
        return std::make_shared<nnrt::op::ExpOperation>();
    };
};

#define MAP_OP_COMMON(name)                                                                    \
    class VsiOpInfo##name : public VsiOpInfo {                                                 \
       public:                                                                                 \
        VsiOpInfo##name() { SetupCallbackInfo(std::make_shared<VsiOpCallbackInfo##name>()); }; \
    };

MAP_OP_COMMON(Abs)
MAP_OP_COMMON(Sqrt)
MAP_OP_COMMON(Add)
MAP_OP_COMMON(Sub)
MAP_OP_COMMON(Mul)
MAP_OP_COMMON(Div)
MAP_OP_COMMON(Sum)
MAP_OP_COMMON(MatMul)
MAP_OP_COMMON(Log)
MAP_OP_COMMON(Pow)
MAP_OP_COMMON(Exp)

}  // namespace onnxruntime