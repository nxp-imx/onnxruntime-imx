// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020-2021 NXP
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/concat.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_common.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator ConcatIterator;

template <typename T>
class NHWCConcat : public onnxruntime::Concat {
 public:
  explicit NHWCConcat(const OpKernelInfo& info) : onnxruntime::Concat(info) {

    provider_ = (const_cast<ArmNNExecutionProvider*>(
        dynamic_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~NHWCConcat() {
    ConcatIterator it = NHWCConcat::rt->layers.find((OpKernel*)this);
    if (it != NHWCConcat::rt->layers.end()) {
      NHWCConcat::rt->run->UnloadNetwork(it->second);
    }
    NHWCConcat::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!NHWCConcat::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      NHWCConcat::rt =  &runtime_obj;
    }
  }

 protected:
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;
};

}  // namespace armnn_ep
}  // namespace onnxruntime
