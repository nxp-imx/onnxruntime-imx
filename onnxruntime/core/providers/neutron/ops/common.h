// Copyright 2025 NXP

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace neutron {

/*
    From CPU Provider implementation
*/

static void PrepareForQDQ(const TensorShape& input_shape,
                          const Tensor& scale,
                          const Tensor* zero_point_ptr,
                          int64_t axis,
                          int64_t& block_count,
                          int64_t& broadcast_dim,
                          int64_t& block_size) {
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear
    block_count = 1;
    broadcast_dim = 1;
    block_size = static_cast<size_t>(input_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel QuantizeLinear/DequantizeLinear
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());
    block_count = input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis_no_neg));
    broadcast_dim = input_shape[onnxruntime::narrow<size_t>(axis_no_neg)];
    block_size = input_shape.SizeFromDimension(SafeInt<size_t>(axis_no_neg) + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                "scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(zero_point_ptr == nullptr ||
               (zero_point_ptr->Shape().NumDimensions() == 1 && zero_point_ptr->Shape()[0] == broadcast_dim),
                "x_zero_point must be null or 1D tensor with size ",
                broadcast_dim);
  }
}

}  // namespace neutron
}  // namespace onnxruntime
