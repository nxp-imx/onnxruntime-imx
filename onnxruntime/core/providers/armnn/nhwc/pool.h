// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_common.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator PoolLayersIterator;

template <typename T, typename PoolType>
class NHWCPool final : public onnxruntime::Pool<T, PoolType> {
 public:
  explicit NHWCPool(const OpKernelInfo& info) : onnxruntime::Pool<T, PoolType>(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        dynamic_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~NHWCPool() {
    PoolLayersIterator it = NHWCPool::rt->layers.find((OpKernel*)this);
    if (it != NHWCPool::rt->layers.end()) {
      NHWCPool::rt->run->UnloadNetwork(it->second);
    }
    NHWCPool::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!NHWCPool::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      NHWCPool::rt =  &runtime_obj;
    }
  }

 private:
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;
};

}  // namespace contrib
}  // namespace onnxruntime
