// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/concat.h"
#include "core/providers/acl/acl_execution_provider.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"

namespace onnxruntime {
namespace acl {

template <typename T>
class Concat final : public OpKernel, public ConcatBase {
 public:
  explicit Concat(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info) {

    provider_ = (const_cast<ACLExecutionProvider*>(
        dynamic_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Concat() {}

  Status Compute(OpKernelContext* context) const override;

 private:
  ACLExecutionProvider* provider_;
};


}  // namespace acl
}  // namespace onnxruntime
