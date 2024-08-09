// Copyright (c) NXP. All rights reserved.

#include "core/providers/neutron/ops/matmul_integer_to_float.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"

// CPU matmulintegertofloat, remove when neutron integrated
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

#if NEUTRON_AARCH64
#include "NeutronDriver.h"
#endif

namespace onnxruntime {
namespace neutron {

#ifndef NDEBUG
extern double time_diff(struct timespec start_time, struct timespec end_time);
#endif

extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;

ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
    MatMulIntegerToFloat,                                                   \
    kMSDomain,                                                              \
    1,                                                                      \
    uint8_t,                                                                \
    kNeutronExecutionProvider,                                              \
    KernelDefBuilder()                                                      \
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())       \
        .TypeConstraint("T2", { DataTypeImpl::GetTensorType<uint8_t>(),     \
                                DataTypeImpl::GetTensorType<int8_t>() })    \
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),        \
    MatMulIntegerToFloat);  

ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
    MatMulIntegerToFloat,                                                   \
    kMSDomain,                                                              \
    1,                                                                      \
    int8_t,                                                                 \
    kNeutronExecutionProvider,                                              \
    KernelDefBuilder()                                                      \
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())        \
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())        \
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),        \
    MatMulIntegerToFloat);


void ScaleOutput(const Tensor& scale, Tensor& output) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.ScalarInput0<float>() * per_iter_bh.EigenInput1<float>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().array() * per_iter_bh.ScalarInput1<float>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().cwiseProduct(per_iter_bh.EigenInput1<float>());
      }};

  InputBroadcaster input_broadcaster(scale, output);
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(),
                                       output);
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster);

  BroadcastLooper(broadcast_helper, funcs);
}

Status MatMulIntegerToFloatBase::ComputeCommon(OpKernelContext* ctx,
                                               const uint8_t* a_data,
                                               const TensorShape& a_shape,
                                               float a_scale,
                                               uint8_t a_zp,
                                               bool a_is_signed,
                                               const Tensor* b_tensor,
                                               const Tensor* b_scale_tensor,
                                               const Tensor* b_zp_tensor,
                                               const Tensor* bias_tensor) const {
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape,
                                     b_tensor ? b_tensor->Shape() : b_shape_,
                                     b_scale_tensor ? &b_scale_tensor->Shape() : nullptr,
                                     b_zp_tensor ? &b_zp_tensor->Shape() : nullptr));
  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();
  const auto* bias_data = bias_tensor != nullptr ? bias_tensor->Data<float>() : nullptr;

  // process zero point of b
  bool is_b_zp_per_column = false;
  uint8_t b_zp_default = 0;
  const uint8_t* b_zp_ptr = &b_zp_default;
  if (nullptr != b_zp_tensor) {
    ORT_ENFORCE(IsBQuantParamSupported(b_zp_tensor->Shape(), b_tensor ? b_tensor->Shape() : b_shape_),
                "MatmulInteger : b zero point is not valid");

    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zp_tensor);
    b_zp_ptr = static_cast<const uint8_t*>(b_zp_tensor->DataRaw());
  }

  // process scale of b
  bool is_b_scale_per_column = false;
  float multiplier_per_tensor = a_scale;
  const float* b_scale_data = &multiplier_per_tensor;
  std::vector<float> multipliers_per_column;
  if (nullptr != b_scale_tensor) {
    is_b_scale_per_column = !IsScalarOr1ElementVector(b_scale_tensor);
    const float* b_scale_tensor_data = b_scale_tensor->Data<float>();

    if (is_b_scale_per_column) {
      multipliers_per_column.reserve(narrow<size_t>(b_scale_tensor->Shape().Size()));
      std::transform(b_scale_tensor_data,
                     b_scale_tensor_data + b_scale_tensor->Shape().Size(),
                     std::back_inserter(multipliers_per_column),
                     [&a_scale](float b_scale) {
                       return a_scale * b_scale;
                     });
      b_scale_data = multipliers_per_column.data();
    } else {
      multiplier_per_tensor *= *b_scale_tensor_data;
    }
  }

  // batch gemm
  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.AIsSigned = a_is_signed;
  gemm_shape.BIsSigned = b_tensor ? b_tensor->IsDataType<int8_t>() : b_is_signed_;

  const size_t num_gemms = helper.OutputOffsets().size();
  std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> gemm_scale_procs;
  gemm_scale_procs.reserve(num_gemms);
  std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(num_gemms);

  for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
    gemm_scale_procs.emplace_back(y_data + helper.OutputOffsets()[gemm_idx],
                                  gemm_shape.N,
                                  b_scale_data + helper.RightScaleOffsets()[gemm_idx],
                                  bias_data,
                                  MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                  is_b_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
    auto& params = gemm_data_vec[gemm_idx];
    params.OutputProcessor = &(gemm_scale_procs[gemm_idx]);
    params.A = a_data + helper.LeftOffsets()[gemm_idx];
    params.lda = gemm_shape.K;
    params.ZeroPointA = a_zp;
    params.BIsPacked = bool(packed_b_);
    params.B = b_tensor ? static_cast<const uint8_t*>(b_tensor->DataRaw()) + helper.RightOffsets()[gemm_idx] : packed_b_.get();
    params.ldb = gemm_shape.N;
    params.ZeroPointB = b_zp_ptr + helper.RightZeroPointOffsets()[gemm_idx];
    params.PerColumnZeroPoints = is_b_zp_per_column;
    params.C = reinterpret_cast<int32_t*>(y_data + helper.OutputOffsets()[gemm_idx]);
    params.ldc = gemm_shape.N;
  }

  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, ctx->GetOperatorThreadPool());

  return Status::OK();
}


void MatMulIntegerToFloat::FixupScaleTensor(const Tensor*& a_scale_tensor, const Tensor*& b_scale_tensor) {
  const TensorShape a_scale_shape = a_scale_tensor->Shape();
  const TensorShape b_scale_shape = b_scale_tensor->Shape();
  if (!IsScalarOr1ElementVector(a_scale_tensor)) {
    size_t a_scale_rank = a_scale_shape.NumDimensions();
    if (a_scale_rank == 1 || a_scale_shape[a_scale_rank - 1] != 1) {
      std::swap(a_scale_tensor, b_scale_tensor);
    }
  } else if (!IsScalarOr1ElementVector(b_scale_tensor)) {
    size_t b_scale_rank = b_scale_shape.NumDimensions();
    if (b_scale_rank > 1 && b_scale_shape[b_scale_rank - 2] != 1) {
      std::swap(a_scale_tensor, b_scale_tensor);
    }
  }
}

Status MatMulIntegerToFloat::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                     /*out*/ bool& is_packed,
                                     /*out*/ PrePackedWeights* prepacked_weights) {
 try {
    if (!useCPU) {
      switch (input_idx) {
      case IN_A:
        break;
      case IN_B:
        {
          m_b_rows = tensor.Shape()[1];
          m_b_cols = tensor.Shape()[0];
          m_handle = neutronAlloc->getMemoryHandle();
          m_header = (uint32_t*) neutronAlloc->Alloc(16*sizeof(uint32_t), m_handle);
          m_b_neutron = (int8_t*) neutronAlloc->Alloc(m_b_rows * m_b_cols, m_handle);
          const int8_t *b_data = static_cast<const int8_t*>(tensor.DataRaw());
          for(uint32_t i=0; i<m_b_cols; i++) {
            for(uint32_t j=0; j<m_b_rows; j++) {
              m_b_neutron[m_b_cols*j+i] = b_data[m_b_rows*i+j];
            }
          }
        }
        break;
      case IN_A_SCALE:
        {
          m_a_scale_data = *(tensor.Data<float>());
        }
        break;
      case IN_B_SCALE:
        {
          out_scale.resize(m_b_rows);
          for (size_t i = 0; i < out_scale.size(); i++) {
            out_scale[i] = tensor.Data<float>()[i] * m_a_scale_data;
          }
        }
        break;
      case IN_A_ZERO_POINT:
        {
          m_a_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));
          // Neutron expects the bias as a parameter anyway.
          m_b_bias = (int32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(int32_t), m_handle);

          for (uint32_t i=0; i< m_b_rows; i++){
            int32_t row_sum = 0;
            for (uint32_t j=0; j< m_b_cols; j++) {
              row_sum += *(m_b_neutron + i * m_b_cols + j);
            }
            m_b_bias[i] = (int32_t)( - row_sum * m_a_zp );
          }

          m_b_factors = (uint32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(uint32_t), m_handle);
          float scale = 1;
          for (uint32_t i=0; i< m_b_rows; i++){                                                                                  
            float *pfloat = &scale;                                                                               
            uint32_t u32 = *(uint32_t*) pfloat;                                                                                

            uint32_t scaler = (u32 >>8) & 0x7fff ; // extract mantissa (15bits)                                                
            int8_t exp_tmp = (u32 >> 23) & 0xff; // extract exponent                                                           

            scaler = (exp_tmp==0) ? 0 :  scaler | 0x8000; // add hidden bit or zero out (if zero or subnormal                  
            exp_tmp = -(exp_tmp -142); // we subtract FP32 offset as well as 16bit growth of our scaler (126 is power of -1 so mantissa is in range 0.5 to 1, 126 + 16=142, where 16 is the factor we multiply by in scaler)
            int8_t exp = (exp_tmp>63) ? 63 : exp_tmp; // ensure that we don't exceed available shift bits (note that this step could, in theory be skipped if this never happens. Not sure if we can take the chance) 
            //if (exp == 63)                                                                                                   
            //      printf("scalar %0d, exp %0d", (uint32_t)scaler, (uint32_t)exp);                                            
            scaler = (exp<<16) | scaler; // merge scaler and downshift factor into the Neutron 32bit scaler format (16bit scaler in LSB and then 6bits of downshift)

            m_b_factors[i] = scaler;
          }
        }
        break;
      case IN_B_ZERO_POINT:
        // we assume B has ZP equal to 0
        // todo: implement a check
        break;
      case IN_BIAS:
        {
          m_output_bias = tensor.Data<float>();
        }      
        break;
      }
    }
  }
  catch (const std::bad_alloc &e) {
    // Do not delegate this instance if out of memory
#ifndef NDEBUG
    printf("[MatMulIntegerToFloat] Unable to alocate Neutron memory\n");
#endif
    useCPU = true;
  }
  return Status::OK();
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  struct timespec t1, t2, t3, t4, t5, t6;

#ifndef NDEBUG
  printf("MatMulIntegerToFloat::Compute\n");
#endif

  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  if (!useCPU && m_header && m_b_neutron && m_b_bias && m_b_factors && !out_scale.empty()) {

    clock_gettime(CLOCK_REALTIME, &t1);

    neutronAlloc->pushMemoryState(m_handle);

    // non-transposed b
    uint32_t neutron_a_rows = a->Shape()[1];
    uint32_t neutron_a_cols = a->Shape()[2];
    uint32_t neutron_b_rows = b ? b->Shape()[1] : m_b_rows;
    uint32_t neutron_b_cols = b ? b->Shape()[0] : m_b_cols;
    if (neutron_a_cols != neutron_b_cols) {
      printf("Neutron dimenssions do not match!\n");
    }

    clock_gettime(CLOCK_REALTIME, &t2);

    uint32_t a_size = neutron_a_rows * neutron_a_cols;
    uint8_t *a_neutron = (uint8_t *) neutronAlloc->AllocReserved(a_size*sizeof(uint8_t), m_handle);
    auto  a_data = static_cast<const uint8_t*>(a->DataRaw());
    memcpy(a_neutron, a_data, a_size);

    int32_t *y_neutron = (int32_t *) neutronAlloc->AllocReserved(neutron_a_rows * neutron_b_rows * sizeof(int32_t), m_handle);

    m_header[0] = 0;
    m_header[1] = 0;
    m_header[2] = neutron_a_rows;
    m_header[3] = neutron_a_cols;
    m_header[4] = neutron_b_rows;
    m_header[5] = (uint8_t *)a_neutron - (uint8_t *)m_header;
    m_header[6] = (uint8_t *)m_b_neutron - (uint8_t *)m_header;
    m_header[7] = (uint8_t *)m_b_bias - (uint8_t *)m_header;
    m_header[8] = (uint8_t *)m_b_factors - (uint8_t *)m_header;
    m_header[9] = (uint8_t *)y_neutron - (uint8_t *)m_header;
    m_header[10] = 0; // m_y_zp;
    m_header[11] = 4; // result num bytes

    clock_gettime(CLOCK_REALTIME, &t3);

    NeutronError ret = ENONE;
    ret = matmul((const void *)m_header, 0, m_handle, 0, 0, 0, 0);
    if (ret != ENONE){
        printf("matmul() error %d\n", ret);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "matmul() error");
    }

    clock_gettime(CLOCK_REALTIME, &t4);

    Tensor* y = ctx->Output(OUT_Y, {1, neutron_a_rows, neutron_b_rows});
    float* y_data = static_cast<float*>(y->MutableDataRaw());

    int32_t* y_new = (int32_t*) malloc(neutron_a_rows * neutron_b_rows * sizeof(int32_t));
    memcpy(y_new, y_neutron, neutron_a_rows * neutron_b_rows * sizeof(int32_t));

    clock_gettime(CLOCK_REALTIME, &t5);

    int32_t* input = y_new; // y_neutron;
    auto* output = y_data;
    for (uint32_t i=0; i<static_cast<uint32_t>(neutron_a_rows); i++) {
      float* scale = (float*) out_scale.data();
      if (m_output_bias) {
        auto* bias = m_output_bias;
        for (uint32_t j=0; j<static_cast<uint32_t>(neutron_b_rows); j++) {
          *output++ = static_cast<float>(*bias++) + static_cast<float>(static_cast<int32_t>(*input++)) * static_cast<float>(*scale++);
        }
      } else {
        for (uint32_t j=0; j<static_cast<uint32_t>(neutron_b_rows); j++) {
          *output++ = static_cast<float>(static_cast<int32_t>(*input++)) * static_cast<float>(*scale++);
        }
      }
    }
    free(y_new);

    neutronAlloc->popMemoryState(m_handle);
    clock_gettime(CLOCK_REALTIME, &t6);

#ifndef NDEBUG
    printf("\nA shape=%ld %ld %ld\n\n",a->Shape()[0],a->Shape()[1],a->Shape()[2]);
    printf("\n");
    for (uint32_t i=0; i<neutron_a_rows; i++) {
      for (uint32_t j=0; j<neutron_b_rows; j++) {
        printf("C[%d][%d]=%f ",i,j,y_data[i * neutron_b_rows + j] * 1.0);
      }
      printf("\n");
    }
    printf("Neutron: Prepared matmul in %f us\n", time_diff(t1,t3));
    printf("Neutron: Computed matmul in %f us\n", time_diff(t3,t4));
    printf("Neutron: Copying result in %f us\n", time_diff(t4,t5));
    printf("Neutron: Dequant of MatMulIntegerToFloat in %f us\n", time_diff(t5,t6));
#endif
  }
#ifdef NDEBUG
  else
#endif
  {
    clock_gettime(CLOCK_REALTIME, &t1);
    const Tensor* a_scale_tensor = ctx->Input<Tensor>(IN_A_SCALE);
    const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
    FixupScaleTensor(a_scale_tensor, b_scale_tensor);
    bool is_a_scale_scalar = IsScalarOr1ElementVector(a_scale_tensor);
    bool is_b_scale_supported = IsBQuantParamSupported(b_scale_tensor->Shape(), nullptr != b ? b->Shape() : b_shape_);

    // validate zero point of a
    uint8_t a_zero_point = 0;
    const Tensor* a_zero_point_tensor = ctx->Input<Tensor>(IN_A_ZERO_POINT);
    if (a_zero_point_tensor != nullptr) {
      ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point_tensor),
                  "MatMulIntegerToFloat : input a zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
      a_zero_point = *(static_cast<const uint8_t*>(a_zero_point_tensor->DataRaw()));
    }

    const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);
    ORT_RETURN_IF_ERROR(ComputeCommon(
                                      ctx,
                                      static_cast<const uint8_t*>(a->DataRaw()),
                                      a->Shape(),
                                      is_a_scale_scalar ? *a_scale_tensor->Data<float>() : 1.f,
                                      a_zero_point,
                                      a->IsDataType<int8_t>(),
                                      b,
                                      is_b_scale_supported ? b_scale_tensor : nullptr,
                                      b_zp_tensor,
                                      ctx->Input<Tensor>(IN_BIAS)));

    if (!is_a_scale_scalar) {
      ScaleOutput(*a_scale_tensor, *ctx->Output<Tensor>(0));
    }
    if (!is_b_scale_supported) {
      ScaleOutput(*b_scale_tensor, *ctx->Output<Tensor>(0));
    }

    clock_gettime(CLOCK_REALTIME, &t4);

#ifndef NDEBUG
    // Dump the output
    const Tensor* y = ctx->Output<Tensor>(0);
    printf("\nY shape=%ld %ld %ld\n\n",y->Shape()[0],y->Shape()[1],y->Shape()[2]);
    const float *y_data = static_cast<const float*>(y->DataRaw());
    printf("\n");
    for (int i=0; i<y->Shape()[1]; i++) {
      for (int j=0; j<y->Shape()[2]; j++) {
        printf("Y[%d][%d]=%f ",i,j,y_data[i * y->Shape()[2] + j]);
      }
      printf("\n");
    }
    printf("CPU: Computed MatMulIntegerToFloat in %f us\n", time_diff(t1,t4));
#endif
  }
  return Status::OK();
}

}  // namespace neutron
}  // namespace onnxruntime
