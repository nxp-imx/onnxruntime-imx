// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/batch_norm.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_common.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator BatchNormLayersIterator;

template <typename T>
class NHWCBatchNorm final : public OpKernel {
 public:
  explicit NHWCBatchNorm(const OpKernelInfo& info) : OpKernel(info) {
    auto st = info.GetAttr<float>("epsilon", &epsilon_);
    ORT_ENFORCE(st.IsOK(), st.ErrorMessage());

    provider_ = (const_cast<ArmNNExecutionProvider*>(
        dynamic_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~NHWCBatchNorm() {
    BatchNormLayersIterator it = NHWCBatchNorm::rt->layers.find((OpKernel*)this);
    if (it != NHWCBatchNorm::rt->layers.end()) {
      NHWCBatchNorm::rt->run->UnloadNetwork(it->second);
    }
    NHWCBatchNorm::rt->layers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static void initRuntime(){
    if(!NHWCBatchNorm::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      NHWCBatchNorm::rt =  &runtime_obj;
    }
  }

 protected:
  float epsilon_;
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;
};

}  // namespace contrib
}  // namespace onnxruntime
