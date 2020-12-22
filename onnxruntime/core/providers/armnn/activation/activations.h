// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
// Licensed under the MIT License

#ifdef RELU_ARMNN

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_common.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator ReluLayersIterator;

template <typename T>
class Relu : public OpKernel {
 public:
  explicit Relu(const OpKernelInfo& info) : OpKernel(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        static_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Relu() {
    ReluLayersIterator it = Relu::rt->layers.find((OpKernel*)this);
    if (it != Relu::rt->layers.end()) {
      Relu::rt->run->UnloadNetwork(it->second);
    }
    Relu::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!Relu::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      Relu::rt =  &runtime_obj;
    }
  }

 private:
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;
};

}  // namespace armnn_ep
}  // namespace onnxruntime

#endif
