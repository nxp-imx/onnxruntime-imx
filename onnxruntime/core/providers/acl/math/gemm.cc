// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2019-2021 NXP
// Licensed under the MIT License.

#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/math/gemm.h"
#include "core/providers/acl/acl_fwd.h"

namespace onnxruntime {
namespace acl {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7,
    8,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    9,
    10,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    11,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

}  // namespace acl
}  // namespace onnxruntime
