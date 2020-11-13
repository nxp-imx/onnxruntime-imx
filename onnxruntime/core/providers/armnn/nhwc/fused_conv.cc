// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/nhwc/conv.h"
#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/armnn_fwd.h"
#include "contrib_ops/cpu/fused_activation.h"


namespace onnxruntime {
namespace armnn_ep {

class NHWCFusedConv final : public armnn_ep::NHWCConv<float> {
 public:
  explicit NHWCFusedConv(const OpKernelInfo& info) : NHWCConv<float>(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &(NHWCConv::activation_type)).IsOK());
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NHWCFusedConv);

}  // namespace acl
}  // namespace onnxruntime
