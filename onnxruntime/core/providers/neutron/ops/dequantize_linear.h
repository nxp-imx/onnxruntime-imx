// Copyright 2025 NXP

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/neutron/neutron_kernel.h"


namespace onnxruntime {
namespace neutron {

/*
    From CPU Provider implementation
*/

template <typename T>
class DequantizeLinear final : public NeutronKernel {
 public:
  explicit DequantizeLinear(const OpKernelInfo& info) : NeutronKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace neutron
}  // namespace onnxruntime
