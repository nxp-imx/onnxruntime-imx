// Copyright (c) NXP. All rights reserved.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/neutron/neutron_kernel.h"
#include "core/providers/cpu/quantization/matmul_integer_base.h"


namespace onnxruntime {
namespace neutron {

class MatMulIntegerToFloatBase : public MatMulIntegerBase {
 public:
  MatMulIntegerToFloatBase(const OpKernelInfo& info) : MatMulIntegerBase(info) {}

  enum OutputTensors : int { OUT_Y = 0 };

 protected:
  Status ComputeCommon(OpKernelContext* ctx,
                       const uint8_t* a_data,
                       const TensorShape& a_shape,
                       float a_scale,
                       uint8_t a_zp,
                       bool a_is_signed,
                       const Tensor* b_tensor,
                       const Tensor* b_scale,
                       const Tensor* b_zp,
                       const Tensor* bias_tensor) const;
};

class MatMulIntegerToFloat final : public MatMulIntegerToFloatBase {
 public:
  MatMulIntegerToFloat(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_SCALE = 2,
    IN_B_SCALE = 3,
    IN_A_ZERO_POINT = 4,
    IN_B_ZERO_POINT = 5,
    IN_BIAS = 6
  };

 protected:
  int GetBIdx() const override { return IN_B; }

  // neutron parameters
  size_t    m_handle{0};
  uint32_t* m_header{NULL};
  int8_t   *m_b_neutron{NULL};
  int32_t  *m_b_bias{NULL};
  uint32_t *m_b_factors{NULL};

  //pre-packing
  float    m_a_scale_data;
  uint8_t  m_a_zp;
  uint32_t m_b_rows;
  uint32_t m_b_cols;
  const float  *m_output_bias{NULL};
  std::vector<float> out_scale;

  bool useCPU{false};
 private:
  // a scale and b scale may be switched in fusion stage because of lack of shape information.
  // Fix them up before computation.
  static void FixupScaleTensor(const Tensor*& a_scale_tensor, const Tensor*& b_scale_tensor);
};


}  // namespace neutron
}  // namespace onnxruntime
