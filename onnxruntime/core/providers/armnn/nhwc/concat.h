// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/concat.h"
#include "core/providers/armnn/armnn_execution_provider.h"

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
	concatLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
  	if (NHWCConcat::run)
  		return std::move(NHWCConcat::run);
	armnn::IRuntime::CreationOptions options;
  	return std::move(armnn::IRuntime::Create(options));
  }

 protected:
  static thread_local std::map<OpKernel*, armnn::NetworkId> concatLayers;
  ArmNNExecutionProvider* provider_;
  static armnn::IRuntimePtr run;
};

}  // namespace contrib
}  // namespace onnxruntime
