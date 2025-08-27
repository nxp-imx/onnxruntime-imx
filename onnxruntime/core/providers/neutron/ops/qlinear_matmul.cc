// Copyright 2025 NXP

#include "core/providers/neutron/ops/qlinear_matmul.h"
#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"

#if NEUTRON_AARCH64
#include "neutron/NeutronDriver.h"
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
            throw std::invalid_argument("NeutronEP:QLinearMatMul invalid argument(s)");

          auto [channelDensity, numNeutrons, divisions] = TilingSolver(m_b_cols, -1, 4, 8, false, false);

          m_handle = neutronAlloc->getMemoryHandle();
          m_header = (uint32_t*) neutronAlloc->Alloc(16*sizeof(uint32_t), m_handle);

          m_b_neutron = (int8_t*) neutronAlloc->Alloc(m_b_rows * m_b_cols, m_handle);
          const int8_t *b_data = static_cast<const int8_t*>(tensor.DataRaw());
          OrganizeWeightsData(b_data, m_b_neutron, m_b_rows,
                              m_b_cols, channelDensity, numNeutrons, 8, 16, true);
          clean_cache(m_b_neutron, m_b_rows*m_b_cols);

          m_b_bias = (int32_t *)neutronAlloc->Alloc(m_b_rows*sizeof(int32_t), m_handle);
          for (uint32_t i=0; i< m_b_rows; i++){
            int32_t row_sum = 0;
            for (uint32_t j=0; j< m_b_cols; j++) {
              row_sum += *(b_data + j * m_b_rows + i);
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
            m_b_factors[i] = ScaleToNeutron(m_output_scales[i]);
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
  catch (const std::exception &e) {
    // Do not delegate this instance if out of memory
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }
  return Status::OK();
}

Status QLinearMatMul::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

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
  if (nullptr != b) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), &b_scale->Shape(), &b_offset->Shape()));
  } else {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_, &b_scale->Shape(), &b_offset->Shape()));
  }

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  struct timespec t1, t2, t3, t4, t5;

  if (!m_header || !m_b_neutron || !m_b_bias || !m_b_factors) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "NeutronEP:QLinearMatMul falied to init.");
  } else {
    clock_gettime(CLOCK_REALTIME, &t1);

    neutronAlloc->pushMemoryState(m_handle);

    // non-transposed b
    uint32_t neutron_a_rows = a->Shape()[1];
    uint32_t neutron_a_cols = a->Shape()[2];
    uint32_t neutron_b_rows = b ? b->Shape()[1] : b_shape_[1];
    uint32_t neutron_b_cols = b ? b->Shape()[0] : b_shape_[0];
    if (neutron_a_cols != neutron_b_cols) {
      LOGS_DEFAULT(WARNING) << "Neutron dimenssions do not match!";
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
    m_header[12] = 8; // Weight Bits
    m_header[13] = -1; // Group Size equal to negative means no group size
    m_header[14] = 0;
    m_header[15] = 0;

    NeutronError ret = ENONE;
    ret = matmul((const void *)m_header, 16*sizeof(uint32_t), (const void*)a_neutron, a_size, (const void*)y_neutron, y_size, m_handle);
    if (ret != ENONE){
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

  return Status::OK();
}
}  // namespace neutron
}  // namespace onnxruntime
