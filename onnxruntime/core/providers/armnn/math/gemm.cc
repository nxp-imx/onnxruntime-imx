// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020-2021 NXP
// Licensed under the MIT License.

#include "core/providers/armnn/math/gemm.h"
#include "core/providers/armnn/armnn_fwd.h"

namespace onnxruntime {
namespace armnn_ep {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7,
    8,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    9,
    10,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    11,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime
