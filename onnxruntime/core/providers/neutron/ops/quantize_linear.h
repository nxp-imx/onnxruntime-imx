// Copyright (c) NXP. All rights reserved.

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
class QuantizeLinear final : public NeutronKernel {
 public:
  explicit QuantizeLinear(const OpKernelInfo& info) : NeutronKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("saturate", &saturate_).IsOK()) {
      saturate_ = 1;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
};

}  // namespace neutron
}  // namespace onnxruntime
