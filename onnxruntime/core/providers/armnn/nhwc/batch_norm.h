// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/batch_norm.h"
#include "core/providers/armnn/armnn_execution_provider.h"

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
	batchNormLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
  	if (NHWCBatchNorm::run)
  		return std::move(NHWCBatchNorm::run);
	armnn::IRuntime::CreationOptions options;
  	return std::move(armnn::IRuntime::Create(options));
  }

 protected:
  float epsilon_;
  static thread_local std::map<OpKernel*, armnn::NetworkId> batchNormLayers;
  ArmNNExecutionProvider* provider_;
  static armnn::IRuntimePtr run;
};

}  // namespace contrib
}  // namespace onnxruntime
