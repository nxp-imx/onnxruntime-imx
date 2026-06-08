// Copyright 2025 NXP

#include "core/providers/neutron/ops/qgemm.h"
#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/quantization/matmul_integer_base.h"
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

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QGemm,
    kMSDomain,
    1,
    uint8_t,
    kNeutronExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TA", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TB", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("TC", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TYZ", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TY", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<uint8_t>()}),
    QGemm);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QGemm,
    kMSDomain,
    1,
    int8_t,
    kNeutronExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TA", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TB", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TC", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TYZ", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TY", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<int8_t>()}),
    QGemm);

/*
    From CPU Provider
*/

Status QGemm::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
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
	  size_t num_dims = tensor.Shape().NumDimensions();
	  m_b_rows = tensor.Shape()[num_dims - 1];
	  m_b_cols = ((num_dims == 1) ? 1 : tensor.Shape()[num_dims - 2]);

          if ((m_b_rows % 16) || (m_b_rows * 16 >= 1024*1024))
            throw std::invalid_argument("NeutronEP:QGemm invalid argument(s)");

          //unpacked_b_ data
          auto unpacked_b = static_cast<const uint8_t*>(tensor.DataRaw());

          const uint32_t MAGIC_WORD = 0x20250918;
          uint32_t magic = *((uint32_t*)unpacked_b);

          uint32_t header_len   = 16 * sizeof(uint32_t);
          uint32_t bias_len     = m_b_rows * sizeof(int32_t);
          uint32_t factor_len   = m_b_rows * sizeof(int32_t);
          uint32_t idecode_len  = 16 * sizeof(uint8_t);

          m_handle = neutronAlloc->getMemoryHandle();

          if (offline_packed_ || magic == MAGIC_WORD) {
            // ---- Offline prepacked path ----
            uint32_t weight_len = *((uint32_t*)unpacked_b + 1);
            uint32_t compress_num = *((uint32_t*)unpacked_b + 2);
            uint32_t compress_len = compress_num * sizeof(int32_t);

            auto total_len = ALIGN16_SIZE(header_len) + ALIGN16_SIZE(weight_len) + ALIGN16_SIZE(bias_len)
                           + ALIGN16_SIZE(factor_len) + ALIGN16_SIZE(compress_len) + ALIGN16_SIZE(idecode_len);

            m_buffer = neutronAlloc->Alloc(total_len, m_handle);

            m_header = (uint32_t*)m_buffer;
            memset(m_header, 0, header_len);
            clean_cache(m_header, header_len);

            m_b_bias       = (int32_t*)((int8_t*)m_header       + ALIGN16_SIZE(header_len));
            m_b_factors    = (int32_t*)((int8_t*)m_b_bias       + ALIGN16_SIZE(bias_len));
            m_b_neutron    =  (int8_t*)((int8_t*)m_b_factors    + ALIGN16_SIZE(factor_len));
            m_compress_len = (int32_t*)((int8_t*)m_b_neutron    + ALIGN16_SIZE(weight_len));
            m_decode_input = (uint8_t*)((int8_t*)m_compress_len + ALIGN16_SIZE(compress_len));

            // weight layout
            // magic_word      weight_length   compress_lengths_number
            // bias   factors compress_weight compress_lengths
            int offset = 12;
            memcpy(m_b_bias,       unpacked_b + offset, bias_len);

            offset += bias_len;
            memcpy(m_b_factors,    unpacked_b + offset, factor_len);

            offset += factor_len;
            memcpy(m_b_neutron,    unpacked_b + offset, weight_len);

            offset += weight_len;
            memcpy(m_compress_len, unpacked_b + offset, compress_len);

            clean_cache(m_b_neutron,    weight_len);
            clean_cache(m_b_factors,    factor_len);
            clean_cache(m_b_bias,       bias_len);
            clean_cache(m_compress_len, compress_len);
          } else {
            // ---- Inline prepack for raw ONNX models ----
            inline_prepacked_ = true;
            bool b_signed = tensor.IsDataType<int8_t>();

            // Transpose B from [m_b_cols, m_b_rows] to [m_b_rows, m_b_cols]
            int8_t* B_trans = (int8_t*)malloc(m_b_rows * m_b_cols);
            for (uint32_t i = 0; i < m_b_rows; i++) {
              for (uint32_t j = 0; j < m_b_cols; j++) {
                B_trans[i * m_b_cols + j] = static_cast<int8_t>(unpacked_b[j * m_b_rows + i]);
              }
            }

            // Compute row sums
            int32_t* row_sum = (int32_t*)calloc(m_b_rows, sizeof(int32_t));
            for (uint32_t i = 0; i < m_b_rows; i++) {
              int32_t sum = 0;
              for (uint32_t j = 0; j < m_b_cols; j++) {
                if (b_signed)
                  sum += static_cast<int32_t>(static_cast<int8_t>(unpacked_b[j * m_b_rows + i]));
                else
                  sum += static_cast<int32_t>(unpacked_b[j * m_b_rows + i]);
              }
              row_sum[i] = sum;
            }

            // Compute bias = -m_a_zp * row_sum (m_a_zp available: IN_A_ZERO_POINT at index 2 before IN_B at index 3)
            int32_t* bias = (int32_t*)malloc(bias_len);
            for (uint32_t i = 0; i < m_b_rows; i++) {
              bias[i] = -(int32_t)m_a_zp * row_sum[i];
            }

            // Call prepack
            PrepackCfg cfg;
            cfg.rearrange     = false;
            cfg.miniWeights   = false;
            cfg.weightBits    = 8;
            cfg.groupSize     = -1;
            cfg.useDecodeBias = false;
            cfg.compress      = true;
            cfg.numMacs       = 16;
            cfg.numNeutrons   = 4;
            cfg.tcmSize       = 1024 * 1024;
            cfg.numBanks      = 16;

            Dyn8  dyn8  = {};
            Dyn32 dyn32 = {};
            PrepackOut pckOut;
            pckOut.Bpacked = &dyn8;
            pckOut.lengths = &dyn32;

            PrePackWeight(B_trans, static_cast<int>(m_b_rows), static_cast<int>(m_b_cols), &cfg, &pckOut);

            size_t weight_len = pckOut.Bpacked->size;
            int32_t compress_num = static_cast<int32_t>(pckOut.lengths->size);
            uint32_t compress_len = compress_num * sizeof(int32_t);

            // Compute factors: all ScaleToNeutron(1.0)
            uint32_t factor_val = ScaleToNeutron(1.0f);
            int32_t* factors = (int32_t*)malloc(factor_len);
            for (uint32_t i = 0; i < m_b_rows; i++) {
              factors[i] = static_cast<int32_t>(factor_val);
            }

            // Allocate Neutron memory
            auto total_len = ALIGN16_SIZE(header_len) + ALIGN16_SIZE(weight_len) + ALIGN16_SIZE(bias_len)
                           + ALIGN16_SIZE(factor_len) + ALIGN16_SIZE(compress_len) + ALIGN16_SIZE(idecode_len);

            m_buffer = neutronAlloc->Alloc(total_len, m_handle);

            m_header = (uint32_t*)m_buffer;
            memset(m_header, 0, header_len);
            clean_cache(m_header, header_len);

            m_b_bias       = (int32_t*)((int8_t*)m_header       + ALIGN16_SIZE(header_len));
            m_b_factors    = (int32_t*)((int8_t*)m_b_bias       + ALIGN16_SIZE(bias_len));
            m_b_neutron    =  (int8_t*)((int8_t*)m_b_factors    + ALIGN16_SIZE(factor_len));
            m_compress_len = (int32_t*)((int8_t*)m_b_neutron    + ALIGN16_SIZE(weight_len));
            m_decode_input = (uint8_t*)((int8_t*)m_compress_len + ALIGN16_SIZE(compress_len));

            memcpy(m_b_bias,    bias, bias_len);
            memcpy(m_b_factors, factors, factor_len);
            memcpy(m_b_neutron, pckOut.Bpacked->data, weight_len);
            memcpy(m_compress_len, pckOut.lengths->data, compress_len);

            clean_cache(m_b_neutron,    weight_len);
            clean_cache(m_b_factors,    factor_len);
            clean_cache(m_b_bias,       bias_len);
            clean_cache(m_compress_len, compress_len);

            free(row_sum);
            free(bias);
            free(factors);
            free(B_trans);
            dyn8_free(pckOut.Bpacked);
            dyn32_free(pckOut.lengths);
          }
        }
        break;
      case IN_B_SCALE:
        {
          if (inline_prepacked_) {
            auto data = tensor.Data<float>();
            if (IsScalarOr1ElementVector(&tensor)) {
              m_b_scales.assign(m_b_rows, *data);
            } else {
              m_b_scales.assign(data, data + m_b_rows);
            }
          }
        }
        break;
      case IN_B_ZERO_POINT:
        // we assume B has ZP equal to 0
        // todo: implement a check
        break;
      case IN_C:
        {
          if (inline_prepacked_) {
            m_c_data = tensor.Data<int32_t>();
          }
        }
	break;
      case IN_Y_SCALE:
        {
          if (inline_prepacked_) {
            m_y_scale_data = *(tensor.Data<float>());

            const int64_t output_scale_size = m_b_rows;
            for (int64_t i = 0; i < output_scale_size; i++) {
              m_output_scales.push_back(m_a_scale_data * m_b_scales[i] / m_y_scale_data);
            }

            for (uint32_t i=0; i< m_b_rows; i++){
              m_b_factors[i] = static_cast<int32_t>(ScaleToNeutron(m_output_scales[i]));
            }
            clean_cache(m_b_factors, m_b_rows*sizeof(int32_t));
          }
        }
        break;
      case IN_Y_ZERO_POINT:
        {
          if (inline_prepacked_) {
            m_y_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));
            for (uint32_t i=0; i < m_b_rows; i++){
              m_b_bias[i] = (int32_t)(m_y_zp / m_output_scales[i] + m_c_data[i] + m_b_bias[i]);
            }
            clean_cache(m_b_bias, m_b_rows*sizeof(int32_t));
          }
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

Status QGemm::Compute(OpKernelContext* ctx) const {
#ifndef NDEBUG
  struct timespec t0, t00, t1, t2, t3, t4, t5;
  clock_gettime(CLOCK_REALTIME, &t0);
#endif
  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = ctx->Input<Tensor>(IN_B);

  // validate offsets
  const auto* a_offset = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  const auto* b_offset = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  const auto* y_offset = ctx->Input<Tensor>(IN_Y_ZERO_POINT);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_offset->Shape(), b->Shape()),
              "QLinearMatmul : weight zero point must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  const auto* a_scale = ctx->Input<Tensor>(IN_A_SCALE);
  const auto* b_scale = ctx->Input<Tensor>(IN_B_SCALE);
  const auto* y_scale = ctx->Input<Tensor>(IN_Y_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_scale->Shape(), b->Shape()),
              "QLinearMatmul : weight scale must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), &b_scale->Shape(), &b_offset->Shape()));

  // non-transposed b
  uint32_t neutron_a_rows = static_cast<uint32_t>(helper.M());
  uint32_t neutron_a_cols = static_cast<uint32_t>(helper.K());
  uint32_t neutron_b_rows = static_cast<uint32_t>(helper.N());
  auto num_matmuls = helper.OutputOffsets().size();

#ifndef NDEBUG
  uint32_t neutron_b_cols = static_cast<uint32_t>(helper.K());
  clock_gettime(CLOCK_REALTIME, &t00);
#endif

  for (size_t batch = 0; batch < num_matmuls; batch++) {
    neutronAlloc->pushMemoryState(m_handle);

#ifndef NDEBUG
    clock_gettime(CLOCK_REALTIME, &t1);
#endif

    Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
    // Bail out early if the output is going to be empty
    if (y->Shape().Size() == 0)
      return Status::OK();

    uint32_t a_size = neutron_a_rows * neutron_a_cols;
    uint8_t *a_neutron = (uint8_t *) neutronAlloc->AllocReserved(a_size*sizeof(uint8_t), m_handle);
    auto  a_data = static_cast<const uint8_t*>(a->DataRaw());
#ifndef NDEBUG
    clock_gettime(CLOCK_REALTIME, &t2);
#endif
    memcpy(a_neutron, a_data + helper.LeftOffsets()[batch], a_size);
    clean_cache(a_neutron, a_size);

#ifndef NDEBUG
    clock_gettime(CLOCK_REALTIME, &t3);
#endif

    uint32_t y_size = neutron_a_rows * neutron_b_rows;
    uint8_t *y_neutron = (uint8_t *) neutronAlloc->AllocReserved(y_size * sizeof(uint8_t), m_handle);
    memset(m_decode_input, 1, 16);
    clean_cache(m_decode_input, 16);


    m_header[0] = (uint8_t *)m_compress_len  - (uint8_t *)m_header;
    m_header[1] = 0;
    m_header[2] = neutron_a_rows;
    m_header[3] = neutron_a_cols;
    m_header[4] = neutron_b_rows | (1 << 18);
    m_header[5] = (uint8_t *)a_neutron - (uint8_t *)m_header;
    m_header[6] = (uint8_t *)m_b_neutron - (uint8_t *)m_header;
    m_header[7] = (uint8_t *)m_b_bias - (uint8_t *)m_header;
    m_header[8] = (uint8_t *)m_b_factors - (uint8_t *)m_header;
    m_header[9] = (uint8_t *)y_neutron - (uint8_t *)m_header;
    m_header[10] = GetMatmulTypeFlag(true, a->IsDataType<int8_t>());
    m_header[11] = 1; // result num bytes
    m_header[12] = 8; // Weight Bits
    m_header[13] = -1; // Group Size equal to negative means no group size
    m_header[14] = 0;
    m_header[15] = (uint8_t *)m_decode_input - (uint8_t *)m_header;

    NeutronError ret = ENONE;
    ret = matmul((const void *)m_header, 16*sizeof(uint32_t), (const void*)a_neutron,
		 a_size, (const void*)y_neutron, y_size, m_handle);
    if (ret != ENONE){
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "matmul() error");
    }

#ifndef NDEBUG
    clock_gettime(CLOCK_REALTIME, &t4);
#endif

    memcpy(static_cast<uint8_t*>(y->MutableDataRaw()) + helper.OutputOffsets()[batch], y_neutron, y_size);

#ifndef NDEBUG
    clock_gettime(CLOCK_REALTIME, &t5);

    printf("NeutronEP: QlinearMatmul [%d,%d]*[%d,%d]: Prepare %f us, B_pack %f us, in_copy %f us, matmul %f us, out_copy %f us\n",
            neutron_a_rows, neutron_a_cols, neutron_b_cols, neutron_b_rows, time_diff(t0,t00), time_diff(t1,t2), time_diff(t2,t3),
            time_diff(t3,t4), time_diff(t4,t5));
#endif

#ifndef NDEBUG
    printf("Neutron: Computed QLinearMatmul of size %ld * %ld * %ld in %f us\n", helper.M(),helper.N(),helper.K(),time_diff(t1,t5));
#endif
    neutronAlloc->popMemoryState(m_handle);
  } //batch
  return Status::OK();

}
}  // namespace neutron
}  // namespace onnxruntime
