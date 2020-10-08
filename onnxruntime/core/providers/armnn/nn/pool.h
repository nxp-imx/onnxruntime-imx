// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool.h"
#include "core/providers/armnn/armnn_execution_provider.h"

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
        dynamic_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Pool() {
    poolLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
    if (Pool::run)
      return std::move(Pool::run);
    armnn::IRuntime::CreationOptions options;
    return std::move(armnn::IRuntime::Create(options));
  }

 private:
  static thread_local std::map<OpKernel*, armnn::NetworkId> poolLayers;
  ArmNNExecutionProvider* provider_;
  static armnn::IRuntimePtr run;
};

}  // namespace armnn_ep
}  // namespace onnxruntime
