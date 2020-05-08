// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/nhwc/nhwc_ops.h"
#include "core/providers/acl/acl_fwd.h"
#include "core/providers/cpu/nn/conv_attributes.h"

// NEON
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

#define CONV_ACL
#undef DEPTHWISE_CPU

#define PREF_DIM 4

namespace onnxruntime {
namespace acl {

template <typename T>
class NhwcFusedConv final : public NhwcConv<T> {
 public:
  explicit NhwcFusedConv(const OpKernelInfo& info) : NhwcConv<T>(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &(this->activation_type)).IsOK());
  }

};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcFusedConv<float>);

}  // namespace acl
}  // namespace onnxruntime
