// Copyright 2026 NXP

#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace neutron {

void PrepareForQDQ(const TensorShape& input_shape,
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

uint32_t ScaleToNeutron(float scale_data) {
  float *scale_ptr = &scale_data;
  uint32_t u32 = *(uint32_t*) (scale_ptr);
  uint32_t scaler = (u32 >>8) & 0x7fff ; // extract mantissa (15bits)
  int8_t exp_tmp = (u32 >> 23) & 0xff; // extract exponent

  // Add hidden bit or zero out (if zero or subnormal)
  scaler = (exp_tmp==0) ? 0 :  scaler | 0x8000;
  // We subtract FP32 offset as well as 16bit growth of our scaler
  // (126 is power of -1 so mantissa is in range 0.5 to 1, 126 + 16=142,
  // where 16 is the factor we multiply by in scaler)
  exp_tmp = -(exp_tmp -142);
  // Ensure that we don't exceed available shift bits
  // (note that this step could, in theory be skipped if this never happens.
  // Not sure if we can take the chance)
  int8_t exp = (exp_tmp>63) ? 63 : exp_tmp;
  // Merge scaler and downshift factor into the Neutron 32bit scaler format
  // (16bit scaler in LSB and then 6bits of downshift)
  scaler = (exp<<16) | scaler;

  return scaler;
}

int32_t
GetMatmulTypeFlag(bool packed, bool signedData) {
    int32_t type = 0;
    if (packed && signedData) {
        type = 2;
    } else if (packed && !signedData) {
        type = 1;
    } else if (!packed && signedData) {
        type = -2;
    } else {
        type = -1;
    }
    return type;
}

}  // namespace neutron
}  // namespace onnxruntime
