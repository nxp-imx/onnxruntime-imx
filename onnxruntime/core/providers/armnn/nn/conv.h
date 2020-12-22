// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_common.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep{

typedef std::map<OpKernel*, armnn::NetworkId>::iterator ConvLayersIterator;

template <typename T>
class Conv : public onnxruntime::Conv<T> {
 public:
  explicit Conv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info), conv_attrs_(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        static_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Conv() {
    ConvLayersIterator it = Conv::rt->layers.find((OpKernel*)this);
    if (it != Conv::rt->layers.end()) {
      Conv::rt->run->UnloadNetwork(it->second);
    }
    Conv::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!Conv::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      Conv::rt =  &runtime_obj;
    }
  }

 protected:
  static thread_local Runtime* rt;
  ConvAttributes conv_attrs_;
  ArmNNExecutionProvider* provider_;
  std::string activation_type;

};

}  // namespace armnn_ep
}  // namespace onnxruntime
