// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
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
class Pool final : public onnxruntime::Pool<T, PoolType> {
 public:
  explicit Pool(const OpKernelInfo& info) : onnxruntime::Pool<T, PoolType>(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        static_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Pool() {
    PoolLayersIterator it = Pool::rt->layers.find((OpKernel*)this);
    if (it != Pool::rt->layers.end()) {
      Pool::rt->run->UnloadNetwork(it->second);
    }
    Pool::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!Pool::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      Pool::rt =  &runtime_obj;
    }
  }

 private:
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;
};

template <typename T>
class MaxPoolV8 final : public onnxruntime::MaxPoolV8 {
 public:
  explicit MaxPoolV8(const OpKernelInfo& info) : onnxruntime::MaxPoolV8(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        static_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~MaxPoolV8() {
    PoolLayersIterator it = MaxPoolV8::rt->layers.find((OpKernel*)this);
    if (it != MaxPoolV8::rt->layers.end()) {
      MaxPoolV8::rt->run->UnloadNetwork(it->second);
    }
    MaxPoolV8::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!MaxPoolV8::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      MaxPoolV8::rt =  &runtime_obj;
    }
  }

 private:
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;
};

}  // namespace armnn_ep
}  // namespace onnxruntime
