// Copyright (c) NXP. All rights reserved.

#include "core/providers/neutron/ops/qlinear_matmul.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"


// CPU matmul, remove when neutron integrated
#include "core/common/narrow.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"
#if NEUTRON_AARCH64
#include "core/providers/neutron/platform/NeutronDriver.h"
#endif
#include "core/providers/neutron/neutron_allocator.h"

namespace onnxruntime {
namespace neutron {

#ifndef NDEBUG
double time_diff(struct timespec start_time, struct timespec end_time)
{
  double ns_diff = (double)(end_time.tv_sec - start_time.tv_sec) * 1e9 + (end_time.tv_nsec - start_time.tv_nsec);
  return ns_diff / 1e3;
}
#endif

extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;

ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
    QLinearMatMul,                                                    \
    kOnnxDomain,                                                      \
    10,                                                               \
    int8_t,                                                           \
    kNeutronExecutionProvider,                                        \
    KernelDefBuilder()                                                \
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())  \
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())  \
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>()), \
    QLinearMatMul);


ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
    QLinearMatMul,                                                       \
    kOnnxDomain,                                                         \
    10,                                                                  \
    uint8_t,                                                             \
    kNeutronExecutionProvider,                                           \
    KernelDefBuilder()                                                   \
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())    \
        .TypeConstraint("T2", { DataTypeImpl::GetTensorType<uint8_t>(),  \
                                DataTypeImpl::GetTensorType<int8_t>() }) \
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),   \
    QLinearMatMul);


/*
    From CPU Provider
*/

Status QLinearMatMul::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                              /*out*/ bool& is_packed,
                              /*out*/ PrePackedWeights* prepacked_weights) {
  try {
    if (!useCPU) {
      switch (input_idx) {
      case IN_A:
        break;
      case IN_A_SCALE:
        m_a_scale_data = *(tensor.Data<float>());
        break;
      case IN_A_ZERO_POINT:
        m_a_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));
        break;
      case IN_B:
        {
          m_b_rows = tensor.Shape()[1];
          m_b_cols = tensor.Shape()[0];

          if ((m_b_rows % 16) || (m_b_rows * 16 >= 1024*1024))
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

          m_b_bias = (int32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(int32_t), m_handle);
          for (uint32_t i=0; i< m_b_rows; i++){
            int32_t row_sum = 0;
            for (uint32_t j=0; j< m_b_cols; j++) {
              row_sum += *(m_b_neutron + i * m_b_cols + j);
            }
            m_b_bias[i] = row_sum;
          }
        }
        break;
      case IN_B_SCALE:
        m_b_scale_data = tensor.Data<float>();
        break;
      case IN_B_ZERO_POINT:
        // we assume B has ZP equal to 0
        // todo: implement a check
        break;
      case IN_Y_SCALE:
        {
          auto y_scale_data = *(tensor.Data<float>());

          const int64_t output_scale_size = m_b_rows;
          for (int64_t i = 0; i < output_scale_size; i++)
            m_output_scales.push_back(m_a_scale_data * m_b_scale_data[narrow<size_t>(i)] / y_scale_data);

          m_b_factors = (uint32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(uint32_t), m_handle);

          for (uint32_t i=0; i< m_b_rows; i++){
            float *pfloat = &(m_output_scales[i]);
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
          clean_cache(m_b_factors, m_b_rows*sizeof(uint32_t));
        }
        break;
      case IN_Y_ZERO_POINT:
        {
          m_y_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));
          for (uint32_t i=0; i< m_b_rows; i++){
            m_b_bias[i] = (int32_t)(m_y_zp / m_output_scales[i] - m_b_bias[i] * m_a_zp);
          }
          clean_cache(m_b_bias, m_b_rows*sizeof(int32_t));
        }
        break;
      }
    }
  }
  catch (const std::bad_alloc &e) {
    // Do not delegate this instance if out of memory
    printf("[NeutronEP:QLinearMatMul} W{%d, %d} will be executed on CPU\n", m_b_cols, m_b_rows);
    useCPU = true;

    //Fast CPU pre-packing
    return MatMulIntegerBase::PrePack(tensor, input_idx, alloc, is_packed, prepacked_weights);
  }
  return Status::OK();
  /*
  (void)tensor;
  (void)input_idx;
  (void)alloc;
  (void)is_packed;
  (void)prepacked_weights;
  return Status::OK();
  */
}

Status QLinearMatMul::Compute(OpKernelContext* ctx) const {
/* @TODO, based in cpu for testing. Modify to add neutron management */
  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  //  printf("[QLinearMatMul] Input A ptr : %p \n", a->DataRaw());
  //  printf("[QLinearMatMul] Input B ptr : %p \n", b->DataRaw());

  // validate offsets
  const auto* a_offset = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  const auto* b_offset = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  const auto* y_offset = ctx->Input<Tensor>(IN_Y_ZERO_POINT);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_offset->Shape(), b ? b->Shape() : b_shape_),
              "QLinearMatmul : weight zero point must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  const auto* a_scale = ctx->Input<Tensor>(IN_A_SCALE);
  const auto* b_scale = ctx->Input<Tensor>(IN_B_SCALE);
  const auto* y_scale = ctx->Input<Tensor>(IN_Y_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_scale->Shape(), b ? b->Shape() : b_shape_),
              "QLinearMatmul : weight scale must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  MatMulComputeHelper helper;
  const uint8_t* b_data;
  bool b_is_signed;
  if (nullptr != b) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), &b_scale->Shape(), &b_offset->Shape()));
    b_data = static_cast<const uint8_t*>(b->DataRaw());
    b_is_signed = b->IsDataType<int8_t>();
  } else {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_, &b_scale->Shape(), &b_offset->Shape()));
    b_data = static_cast<const uint8_t*>(packed_b_.get());
    b_is_signed = b_is_signed_;
  }

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  //  printf("[QLinearMatMul] Output Y ptr : %p \n", y->DataRaw());
  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  struct timespec t1, t2, t3, t4, t5;

  if (!useCPU && m_header && m_b_neutron && m_b_bias && m_b_factors) {
    clock_gettime(CLOCK_REALTIME, &t1);

    neutronAlloc->pushMemoryState(m_handle);

    // non-transposed b
    uint32_t neutron_a_rows = a->Shape()[1];
    uint32_t neutron_a_cols = a->Shape()[2];
    uint32_t neutron_b_rows = b ? b->Shape()[1] : b_shape_[1];
    uint32_t neutron_b_cols = b ? b->Shape()[0] : b_shape_[0];
    if (neutron_a_cols != neutron_b_cols) {
      printf("Neutron dimenssions do not match!\n");
    }

    clock_gettime(CLOCK_REALTIME, &t2);

    uint32_t a_size = neutron_a_rows * neutron_a_cols;
    uint8_t *a_neutron = (uint8_t *) neutronAlloc->AllocReserved(a_size*sizeof(uint8_t), m_handle);
    auto  a_data = static_cast<const uint8_t*>(a->DataRaw());
    memcpy(a_neutron, a_data, a_size);

    clock_gettime(CLOCK_REALTIME, &t3);

    uint32_t y_size = neutron_a_rows * neutron_b_rows;
    uint8_t *y_neutron = (uint8_t *) neutronAlloc->AllocReserved(y_size * sizeof(uint8_t), m_handle);

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
    m_header[10] = m_y_zp;
    m_header[11] = 1; // result num bytes

    NeutronError ret = ENONE;
    ret = matmul((const void *)m_header, 16*sizeof(uint32_t), (const void*)a_neutron, a_size, (const void*)y_neutron, y_size, m_handle);
    if (ret != ENONE){
        printf("matmul() error %d\n", ret);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "matmul() error");
    }

    clock_gettime(CLOCK_REALTIME, &t4);
    memcpy(static_cast<uint8_t*>(y->MutableDataRaw()), y_neutron, neutron_a_rows * m_b_rows);

    neutronAlloc->popMemoryState(m_handle);
    clock_gettime(CLOCK_REALTIME, &t5);

#ifndef NDEBUG
    printf("NeutronEP: QlinearMatmul [%d,%d]*[%d,%d]: in_copy %f us, matmul %f us, dequant %f\n",
            neutron_a_rows, neutron_a_cols, neutron_b_cols, neutron_b_rows, time_diff(t1,t3), time_diff(t3,t4), time_diff(t4,t5));
#endif

#ifndef NDEBUG
    printf("Neutron: Computed QLinearMatmul of size %ld * %ld * %ld in %f us\n", helper.M(),helper.N(),helper.K(),time_diff(t1,t5));
#endif
  }
  else
  {
    const auto* b_scale_data = b_scale->Data<float>();
    auto a_scale_data = *(a_scale->Data<float>());
    auto y_scale_data = *(y_scale->Data<float>());

    const int64_t output_scale_size = b_scale->Shape().Size();
    std::vector<float> output_scales(narrow<size_t>(output_scale_size));
    for (int64_t i = 0; i < output_scale_size; i++) {
      output_scales[narrow<size_t>(i)] = (a_scale_data * b_scale_data[narrow<size_t>(i)] / y_scale_data);
    }

    const size_t num_gemms = helper.OutputOffsets().size();
    MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
    gemm_shape.M = static_cast<size_t>(helper.M());
    gemm_shape.N = static_cast<size_t>(helper.N());
    gemm_shape.K = static_cast<size_t>(helper.K());
    gemm_shape.AIsSigned = a->IsDataType<int8_t>();
    gemm_shape.BIsSigned = b_is_signed;

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(gemm_shape.M) *
                                         gemm_shape.N * sizeof(int32_t) * num_gemms);
    BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(std::move(alloc)));
    auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_params(num_gemms);
    std::vector<MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR> requant_procs;
    requant_procs.reserve(num_gemms);

    bool is_output_signed = y->IsDataType<int8_t>();
    int32_t output_offset = is_output_signed ? *(static_cast<const int8_t*>(y_offset->DataRaw()))
      : *(static_cast<const uint8_t*>(y_offset->DataRaw()));
    auto b_zp_data = static_cast<const uint8_t*>(b_offset->DataRaw());
    for (size_t i = 0; i < num_gemms; i++) {
      gemm_params[i].A = static_cast<const uint8_t*>(a->DataRaw()) + helper.LeftOffsets()[i];
      gemm_params[i].lda = gemm_shape.K;
      gemm_params[i].ZeroPointA = *(static_cast<const uint8_t*>(a_offset->DataRaw()));

      gemm_params[i].B = b_data + helper.RightOffsets()[i];
      gemm_params[i].ldb = gemm_shape.N;
      gemm_params[i].BIsPacked = bool(packed_b_);
      gemm_params[i].ZeroPointB = b_zp_data + helper.RightZeroPointOffsets()[i];

      gemm_params[i].C = gemm_output + (gemm_shape.M * gemm_shape.N * i);
      gemm_params[i].ldc = gemm_shape.N;

      gemm_params[i].PerColumnZeroPoints = !IsScalarOr1ElementVector(b_offset);

      requant_procs.emplace_back(static_cast<uint8_t*>(y->MutableDataRaw()) + helper.OutputOffsets()[i],
                                 static_cast<size_t>(helper.N()),
                                 nullptr,
                                 output_scales.data() + helper.RightScaleOffsets()[i],
                                 output_scales.size() > 1,
                                 output_offset,
                                 is_output_signed);
      gemm_params[i].OutputProcessor = &(requant_procs[i]);
    }

    clock_gettime(CLOCK_REALTIME, &t3);
    MlasGemmBatch(gemm_shape, gemm_params.data(), num_gemms, ctx->GetOperatorThreadPool());
    clock_gettime(CLOCK_REALTIME, &t4);
#ifndef NDEBUG
    printf("CPU: Computed QLinearMatmul of size %ld * %ld * %ld in %f us\n", helper.M(),helper.N(),helper.K(),time_diff(t3,t4));
#endif
  }

  return Status::OK();
}
}  // namespace neutron
}  // namespace onnxruntime
