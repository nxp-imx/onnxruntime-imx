// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_common.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator TransposeIterator;

template <typename T>
class Transpose : public onnxruntime::Transpose {
 public:
  explicit Transpose(const OpKernelInfo& info) : onnxruntime::Transpose(info) {

    provider_ = (const_cast<ArmNNExecutionProvider*>(
        dynamic_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Transpose() {
    TransposeIterator it = Transpose::rt->layers.find((OpKernel*)this);
    if (it != Transpose::rt->layers.end()) {
      Transpose::rt->run->UnloadNetwork(it->second);
    }
    Transpose::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!Transpose::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      Transpose::rt =  &runtime_obj;
    }
  }

 protected:
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;
};

}  // namespace armnn_ep
}  // namespace onnxruntime
