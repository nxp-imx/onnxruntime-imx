// Copyright 2025 NXP 

#include "core/providers/neutron/ops/matmul_integer.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

#if NEUTRON_AARCH64
#include "neutron/NeutronDriver.h"
#endif

namespace onnxruntime {
namespace neutron {

#ifndef NDEBUG
extern double time_diff(struct timespec start_time, struct timespec end_time);
#endif

extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;

ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
    MatMulInteger,                                                         \
    kOnnxDomain,                                                           \
    10,                                                                    \
    uint8_t,                                                               \
    kNeutronExecutionProvider,                                             \
    KernelDefBuilder()                                                     \
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())      \
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(),     \
                               DataTypeImpl::GetTensorType<int8_t>()})     \
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),     \
    MatMulInteger);

ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
    MatMulInteger,                                                         \
    kOnnxDomain,                                                           \
    10,                                                                    \
    int8_t,                                                                \
    kNeutronExecutionProvider,                                             \
    KernelDefBuilder()                                                     \
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())       \
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())       \
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),     \
    MatMulInteger);

Status MatMulInteger::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
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

          if (m_b_rows % 16 || (m_b_rows * 16 >= 1024*1024))
              throw std::bad_alloc();

          m_handle = neutronAlloc->getMemoryHandle();
          m_header = (uint32_t*) neutronAlloc->Alloc(16*sizeof(uint32_t), m_handle);
          m_b_neutron = (int8_t*) neutronAlloc->Alloc(m_b_rows * m_b_cols, m_handle);

          const int8_t *b_data = static_cast<const int8_t*>(tensor.DataRaw());
          for(uint32_t i=0; i<m_b_cols; i++) {
            for(uint32_t j=0; j<m_b_rows; j++) {
              m_b_neutron[m_b_cols*j+i] = b_data[m_b_rows*i+j];
            }
          }
          clean_cache(m_b_neutron, m_b_rows*m_b_cols);

          // m_b_bias
          m_b_row_sum = (int32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(int32_t), m_handle);
          for (uint32_t i=0; i< m_b_rows; i++){
            int32_t row_sum = 0;
            for (uint32_t j=0; j< m_b_cols; j++) {
              row_sum += *(m_b_neutron + i * m_b_cols + j);
            }
            m_b_row_sum[i] = row_sum;
          }
          m_b_bias = (int32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(int32_t), m_handle);

          // m_b_factors
          m_b_factors = (uint32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(uint32_t), m_handle);
          float scale = 1;
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

          for (uint32_t i=0; i< m_b_rows; i++){
            m_b_factors[i] = scaler;
          }
          clean_cache(m_b_factors, m_b_rows*sizeof(uint32_t));
        }
        break;
      case IN_A_ZERO_POINT:
        {
          m_dynamic_bias = false;
          m_a_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));

          for (uint32_t i=0; i< m_b_rows; i++){
            m_b_bias[i] = (int32_t)( - m_b_row_sum[i] * m_a_zp );
          }
          clean_cache(m_b_bias, m_b_rows*sizeof(int32_t));
        }
        break;
      case IN_B_ZERO_POINT:
        // we assume B has ZP equal to 0
        // todo: implement a check
        break;
      }
    }
  }
  catch (const std::bad_alloc &e) {
    // Do not delegate this instance if out of memory
    printf("[NeutronEP:MatMulInteger] W[%d, %d] will be executed on CPU\n", m_b_cols, m_b_rows);
    useCPU = true;

    // Fast CPU prepacking
    return MatMulIntegerBase::PrePack(tensor, input_idx, alloc, is_packed, prepacked_weights);
  }
  return Status::OK();
}


Status MatMulInteger::Compute(OpKernelContext* ctx) const {
  struct timespec t1, t2, t3, t4, t5;

#ifndef NDEBUG
  printf("MatMulInteger::Compute\n");
#endif

  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  if (!useCPU && m_header && m_b_neutron) {

    clock_gettime(CLOCK_REALTIME, &t1);

    if (m_dynamic_bias) {
          uint8_t a_zp = *(static_cast<const uint8_t*>(ctx->Input<Tensor>(IN_A_ZERO_POINT)->DataRaw()));

          for (uint32_t i=0; i< m_b_rows; i++){
            m_b_bias[i] = (int32_t)( - m_b_row_sum[i] * a_zp );
          }

          //TODO: move it to matmul call
          clean_cache(m_b_bias, m_b_rows*sizeof(int32_t));
    }

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

    uint32_t a_size = neutron_a_rows * neutron_a_cols * sizeof(uint8_t);
    uint8_t *a_neutron = (uint8_t *) neutronAlloc->AllocReserved(a_size, m_handle);
    auto  a_data = static_cast<const uint8_t*>(a->DataRaw());
    memcpy(a_neutron, a_data, a_size);

    uint32_t y_size = neutron_a_rows * neutron_b_rows * sizeof(int32_t);
    int32_t *y_neutron = (int32_t *) neutronAlloc->AllocReserved(y_size, m_handle);

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
    m_header[12] = 8; // Weight Bits
    m_header[13] = -1; // Group Size equal to negative means no group size
    m_header[14] = 0;
    m_header[15] = 0; 

    clock_gettime(CLOCK_REALTIME, &t3);

    NeutronError ret = ENONE;
    ret = matmul((const void *)m_header, 16*sizeof(uint32_t), (const void*)a_neutron, a_size, (const void*)y_neutron, y_size, m_handle);
    if (ret != ENONE){
        printf("matmul() error %d\n", ret);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "matmul() error");
    }

    clock_gettime(CLOCK_REALTIME, &t4);

    Tensor* y = ctx->Output(OUT_Y, {1, neutron_a_rows, neutron_b_rows});
    int32_t* y_data = static_cast<int32_t*>(y->MutableDataRaw());
    memcpy(y_data, y_neutron, sizeof(int32_t) * neutron_a_rows * m_b_rows);

    neutronAlloc->popMemoryState(m_handle);
    clock_gettime(CLOCK_REALTIME, &t5);

#ifndef NDEBUG
    printf("Neutron MatMulIntegerToFloat [%d,%d]*[%d,%d]: in_copy %f us, matmul %f us, out_copy %f\n",
            neutron_a_rows, neutron_a_cols, neutron_b_cols, neutron_b_rows, time_diff(t1,t3), time_diff(t3,t4), time_diff(t4,t5));
#endif

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
#endif
  }
#ifdef NDEBUG
  else
#endif
 {
     clock_gettime(CLOCK_REALTIME, &t1);

  // validate zero points
  uint8_t a_offset = 0;
  const auto* a_zero_point = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (a_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *(static_cast<const uint8_t*>(a_zero_point->DataRaw()));
  }

  bool is_b_zp_per_column = false;
  uint8_t b_default_offset = 0;
  const uint8_t* b_offset_ptr = &b_default_offset;
  const auto* b_zero_point = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  if (b_zero_point != nullptr) {
    ORT_ENFORCE(IsBQuantParamSupported(b_zero_point->Shape(), b ? b->Shape() : b_shape_),
                "MatmulInteger : B zero point is not valid");
    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zero_point);
    b_offset_ptr = static_cast<const uint8_t*>(b_zero_point->DataRaw());
  }

  MatMulComputeHelper helper;
  const uint8_t* b_data;
  bool b_is_signed;
  if (nullptr != b) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(b->DataRaw());
    b_is_signed = b->IsDataType<int8_t>();
  } else {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_, nullptr, b_zero_point ? &b_zero_point->Shape() : nullptr));
    b_data = static_cast<const uint8_t*>(packed_b_.get());
    b_is_signed = b_is_signed_;
  }

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  const uint8_t* a_data = static_cast<const uint8_t*>(a->DataRaw());
  auto* y_data = y->MutableData<int32_t>();

  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.AIsSigned = a->IsDataType<int8_t>();
  gemm_shape.BIsSigned = b_is_signed;

  const size_t batch_size = helper.OutputOffsets().size();
  std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(batch_size);

  for (size_t batch = 0; batch < batch_size; batch++) {
    auto& gemm_params = gemm_data_vec[batch];
    gemm_params.lda = gemm_shape.K;
    gemm_params.ZeroPointA = a_offset;
    gemm_params.ldb = gemm_shape.N;
    gemm_params.ZeroPointB = b_offset_ptr + helper.RightZeroPointOffsets()[batch];
    gemm_params.PerColumnZeroPoints = is_b_zp_per_column;
    gemm_params.ldc = gemm_shape.N;
    gemm_params.BIsPacked = bool(packed_b_);
    gemm_params.A = a_data + helper.LeftOffsets()[batch];
    gemm_params.B = b_data + helper.RightOffsets()[batch];
    gemm_params.C = y_data + helper.OutputOffsets()[batch];
  }
  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch_size, ctx->GetOperatorThreadPool());

 clock_gettime(CLOCK_REALTIME, &t4);

#ifndef NDEBUG
    // non-transposed b
    uint32_t neutron_a_rows = a->Shape()[1];
    uint32_t neutron_a_cols = a->Shape()[2];
    uint32_t neutron_b_rows = b ? b->Shape()[1] : m_b_rows;
    uint32_t neutron_b_cols = b ? b->Shape()[0] : m_b_cols;

    printf("CPU MatMulInteger [%d,%d,%d]*[%d,%d]: matmul %f us\n",
            (uint32_t) a->Shape()[0], neutron_a_rows, neutron_a_cols, neutron_b_cols, neutron_b_rows, time_diff(t1,t4));
#endif

#ifndef NDEBUG
    // Dump the output
    y = ctx->Output<Tensor>(0);
    printf("\nY shape=%ld %ld %ld\n\n",y->Shape()[0],y->Shape()[1],y->Shape()[2]);
    const float *y_data_f = static_cast<const float*>(y->DataRaw());
    printf("\n");
    for (int i=0; i<y->Shape()[1]; i++) {
      for (int j=0; j<y->Shape()[2]; j++) {
        printf("Y[%d][%d]=%f ",i,j,y_data_f[i * y->Shape()[2] + j]);
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
