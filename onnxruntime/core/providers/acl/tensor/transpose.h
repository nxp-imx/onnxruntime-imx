// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/acl/acl_execution_provider.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEPermute.h"

namespace onnxruntime {
namespace acl {

typedef struct
{
  std::shared_ptr<arm_compute::NEPermute> layer;
  std::shared_ptr<arm_compute::MemoryManagerOnDemand> mm_layer;
  std::shared_ptr<arm_compute::Tensor> in;
  std::shared_ptr<arm_compute::Tensor> out;
} ACLNEPermute;

typedef std::map<OpKernel*, ACLNEPermute>::iterator permuteLayersIterator;

template <typename T>
class Transpose : public onnxruntime::Transpose {
 public:
  explicit Transpose(const OpKernelInfo& info) : onnxruntime::Transpose(info) {

    provider_ = (const_cast<ACLExecutionProvider*>(
        static_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Transpose() {
    Transpose::permuteLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  static thread_local std::map<OpKernel*, ACLNEPermute> permuteLayers;
  ACLExecutionProvider* provider_;
};

}  // namespace acl
}  // namespace onnxruntime
