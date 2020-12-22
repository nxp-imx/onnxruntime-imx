// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
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
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator ConvLayersIterator;

template <typename T>
class NHWCConv : public onnxruntime::Conv<T> {
 public:
  explicit NHWCConv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info), conv_attrs_(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        dynamic_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~NHWCConv() {
    ConvLayersIterator it = NHWCConv::rt->layers.find((OpKernel*)this);
    if (it != NHWCConv::rt->layers.end()) {
      NHWCConv::rt->run->UnloadNetwork(it->second);
    }
    NHWCConv::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!NHWCConv::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      NHWCConv::rt =  &runtime_obj;
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
