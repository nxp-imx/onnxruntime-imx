// Copyright 2025 NXP

#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <unistd.h>

#include "core/providers/neutron/ops/matmul_nbits.h"
#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#if NEUTRON_AARCH64
#include "neutron/NeutronDriver.h"
#endif

namespace onnxruntime {
namespace neutron {
extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;
#define USE_NBITS_KERNEL 1

ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
    MatMulNBits,                                                          \
    kMSDomain,                                                            \
    1,                                                                    \
    float,                                                                \
    kNeutronExecutionProvider,                                            \
    KernelDefBuilder()                                                    \
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())       \
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())     \
        .TypeConstraint("T3", {DataTypeImpl::GetTensorType<uint8_t>(),    \
                               DataTypeImpl::GetTensorType<float>(),      \
                               DataTypeImpl::GetTensorType<MLFloat16>()}) \
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),    \
    onnxruntime::neutron::MatMulNBits);

double DecimalToFixedPoint(double number, int integer_bits = 10, int fraction_bits = 6) {
  bool sign = false;
  if (number < 0) {
    sign = true;
    number = -number;
  }

  if (number > (1 << integer_bits) - 1) {
    number = (1 << integer_bits) - 1;
  }

  // Split integer and fractional parts
  int integer_part = static_cast<int>(number);
  double fractional_part = number - integer_part;

  bool first_bit_obtained = (integer_part > 0);
  int bits_obtained = first_bit_obtained ? static_cast<int>(log2(integer_part)) + 1 : 0;
  int bits_left = integer_bits - bits_obtained;
  int shift = 0;
  int fixed_point_scale = integer_part;

  if (first_bit_obtained) {
    fraction_bits = bits_left;
  } else {
    while (fractional_part < 0.5 && shift < (1 << fraction_bits) - 1) {
      fractional_part *= 2;
      shift++;
    }
 }

  // Convert fractional part to binary
  for (int i = 0; i < fraction_bits; i++) {
    if (shift >= (1 << fraction_bits) - 1) {
      break;
    }
    fractional_part *= 2;
    int bit = static_cast<int>(fractional_part);
    fixed_point_scale = (fixed_point_scale * 2) | bit;
    shift++;
    fractional_part -= bit;
    if (fractional_part == 0) {
      break;
    }
  }
  int scale_10bit = fixed_point_scale & 0b1111111111;
  int shift_6bit = shift & 0b111111;
  return (scale_10bit * pow(2, -shift_6bit)) * (sign ? -1 : 1);
}

// Number must be non-negative
int16_t DecimalToNeutron(double number, int integer_bits = 10, int fraction_bits = 6) {
  if (number > (1 << integer_bits) - 1) {
    number = (1 << integer_bits) - 1;
  }

  int integer_part = static_cast<int>(number);
  double fractional_part = number - integer_part;

  bool first_bit_obtained = (integer_part > 0);
  int bits_obtained = (integer_part > 0) ? static_cast<int>(log2(integer_part)) + 1 : 0;
  int bits_left = integer_bits - bits_obtained;

  int shift = 0;
  int fixed_point_scale = integer_part & 0b1111111111;

  if (first_bit_obtained) {
    fraction_bits = bits_left;
  } else {
    while (fractional_part > 0 && fractional_part < 0.5 && shift < (1 << fraction_bits) - 1) {
      fractional_part *= 2;
      shift++;
    }
  }

  for (int i = 0; i < fraction_bits; i++) {
    if (shift >= (1 << fraction_bits) - 1) {
      break;
    }
    fractional_part *= 2;
    int bit = static_cast<int>(fractional_part);
    fixed_point_scale = (fixed_point_scale << 1) | bit;
    shift++;
    fractional_part -= bit;
    if (fractional_part == 0) {
      break;
    }
  }

  int scale_10bit = fixed_point_scale & 0b1111111111;
  int shift_6bit = shift & 0b111111;

  int16_t result = static_cast<int16_t>((shift_6bit << 10) | scale_10bit);
  return static_cast<int16_t>(result - 65536 * (shift_6bit >= 32));
}

template <typename T>
void OrganizeDecodeData(const T* decodeData, T* output, void* tempBuffer,
                        int rowsB, int colsB, int channelDensity, int numNeutrons,
                        int groupSize, int divisions, int weightBits = 4, int MACs = 16) {
    T* reorganized = static_cast<T*>(tempBuffer);
    int counter = 0;

    for (int i = 0; i < rowsB; i += channelDensity * numNeutrons) {
        for (int division = 0; division < divisions; ++division) {
            for (int neutron = 0; neutron < numNeutrons; ++neutron) {
                for (int j = 0; j < (colsB / groupSize / divisions); ++j) {
                    for (int row = 0; row < channelDensity; ++row) {
                        int row_index = i + row + neutron * channelDensity;
                        int col_index = j + division * (colsB / groupSize / divisions);
                        int flat_index = row_index * (colsB / groupSize) + col_index;

                        reorganized[counter] = decodeData[flat_index];
                        ++counter;
                    }
                }
            }
        }
    }

    int da = 0;
    int sa = 0;

    int dstStride = channelDensity * colsB / groupSize / divisions;
    int inner_cnt = std::min(dstStride, (int)(8 * 1024 / sizeof(T)));

    int iters = dstStride / inner_cnt;
    int stride = dstStride - inner_cnt;
    int repeats = rowsB / channelDensity / numNeutrons;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int division = 0; division < divisions; ++division) {
            for (int iter = 0; iter < iters; ++iter) {
                int da_save = da;
                for (int idx = 0; idx < numNeutrons; ++idx) {
                    for (int jdx = 0; jdx < inner_cnt; ++jdx) {
                        output[sa++] = reorganized[da++];
                    }
                    da += stride;
                }
                da = da_save + inner_cnt;
            }
            da = da - inner_cnt * iters;
            da += channelDensity * numNeutrons * colsB / groupSize / divisions;
        }
    }
}

void CalculateDecodeData(const uint8_t *B, const float *scalesData, int16_t *decodeScales,
                         int8_t * decodeBias, uint32_t *bFactors, int32_t *bRowSum, int rowsB,
                         int blocksPerCol,int blockSize) {
  for (int i = 0; i < rowsB; i ++) {
      float maxScale = std::abs(scalesData[i * blocksPerCol]);
      for (int m = 1; m < blocksPerCol; m ++) {
        float data = scalesData[i * blocksPerCol + m];
        if (std::abs(data) > maxScale) {
          maxScale = std::abs(data);
        }
      }

      float channelScale = maxScale / 128 * 8;
      float sum = 0;
      for (int j = 0; j < blocksPerCol; j ++) {
        float groupScale = scalesData[i * blocksPerCol + j];
        float scale = groupScale / channelScale;
        if (scale < 0) {
          decodeBias[i * blocksPerCol + j] = 1;
        } else {
          decodeBias[i * blocksPerCol + j] = 0;
        }
        auto fixedPointScale = DecimalToNeutron(DecimalToFixedPoint(std::abs(scale)));
        decodeScales[i * blocksPerCol + j] = fixedPointScale;

        //Caculate weights row sum
        for (int m = 0; m < blockSize; m ++) {
          int idx = i * blocksPerCol * blockSize + j * blockSize + m;
          uint8_t value = B[idx / 2];
          if (idx % 2 == 0) {
            value = (value & 0x0F);
          } else {
            value = ((value >> 4) & 0x0F);
          }

          float temp = ((float)value - 8) * DecimalToFixedPoint(scale);
          sum = sum + (int8_t)std::clamp((int)std::floor(temp + 0.5), -128, 127);
        }
      }

      bRowSum[i] = (int32_t)sum;
      bFactors[i] = ScaleToNeutron(channelScale);
  }
}

//Only 4 bit support
void PackWeight(const uint8_t *B, const float *scalesData, int8_t *packedWeights,
                int rowsB, int colsB, int blocksPerCol, int blockSize, int channelDensity,
                int row_stride, int weightBits = 4, int MACs = 16) {
  int cell_pointer = 0;
  int bit_pointer = 0;

  for (int j = 0; j < rowsB; j += channelDensity) {
    cell_pointer = std::ceil(channelDensity * colsB * weightBits / 8.0) * (j / channelDensity);
    bit_pointer = 0;

    for (int i = 0; i < colsB; i += MACs) {
      for (int m = 0; m < channelDensity; ++m) {
        for (int k = 0; k < MACs; ++k) {
          uint8_t value = B[(j + m) * row_stride / 2  + (i + k) / 2];
          if ((i + k) % 2 == 0) {
            value = (value & 0x0F);
          } else {
            value = ((value >> 4) & 0x0F);
          }
          if (scalesData[(j + m) * blocksPerCol + (i + k) / blockSize] < 0) {
            value = 7 - value;
          } else {
            value = value - 8;
          }
          uint8_t extracted_bits = value & ((1 << weightBits) - 1);
          if (8 - bit_pointer >= weightBits) {
            packedWeights[cell_pointer] |= extracted_bits << bit_pointer;
            bit_pointer += weightBits;
            cell_pointer += bit_pointer / 8;
            bit_pointer %= 8;
          } else {
            int fitting_bits = 8 - bit_pointer;
            int remaining_bits = weightBits - fitting_bits;
            uint8_t rem_extracted_bits = (value >> fitting_bits) & ((1 << remaining_bits) - 1);

            packedWeights[cell_pointer] |= extracted_bits << bit_pointer;
            cell_pointer ++;
            bit_pointer = 0;
            packedWeights[cell_pointer] |= rem_extracted_bits;
            bit_pointer += remaining_bits;
          }
        }
      }
    }
  }

  return;
}

void CMatMul(const float* A, const float* B, float* C, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

void ReQuantizeWeight(const uint8_t *in, int8_t *out, const float *scalesData, uint32_t *bFactors, int rowsB,
                int blocksPerCol, int blockSize) {
  for (int i = 0; i < rowsB; i ++) {
    float maxScale = scalesData[i * blocksPerCol];
    for (int m = 1; m < blocksPerCol; m ++) {
      float data = scalesData[i * blocksPerCol + m];
      if (std::abs(data) > std::abs(maxScale)) {
        maxScale = data;
      }
    }

    float channelScale;
    if (maxScale > 0) {
      channelScale = maxScale / 128 * 8;
    } else {
      channelScale = maxScale / 128 * 8 * -1.0;
    }

    for (int j = 0; j < blocksPerCol; j ++) {
      float groupScale = scalesData[i * blocksPerCol + j];
      for (int m = 0; m < blockSize; m ++) {
        int idx = i * blocksPerCol * blockSize + j * blockSize + m;
        uint8_t value = in[idx / 2];
        if (idx % 2 == 0) {
          value = (value & 0x0F);
        } else {
          value = ((value >> 4) & 0x0F);
        }

        float temp = ((float)value - 8) * groupScale / channelScale;
        out[idx] = (int8_t)std::clamp((int)std::round(temp), -128, 127);
      }
    }

    bFactors[i] = ScaleToNeutron(channelScale);
  }
}

void DequantizeWeight(const uint8_t *in, float *out, const float *scalesData,
                      int rowsB, int blocksPerCol, int blockSize) {
  for (int i = 0; i < rowsB; i ++) {
    for (int j = 0; j < blocksPerCol; j ++) {
      for (int m = 0; m < blockSize; m ++) {
        int idx = i * blocksPerCol * blockSize + j * blockSize + m;

        uint8_t value = in[idx / 2];
        if (idx % 2 == 0) {
          value = (value & 0x0F);
        } else {
          value = ((value >> 4) & 0x0F);
        }

        int outidx = (j * blockSize + m) * rowsB + i;
        out[outidx] = ((float)value - 8) * scalesData[i * blocksPerCol + j];
      }
    }
  }
}

static inline int32x4_t roundq_s32_f32(float32x4_t x) {
    float32x4_t half = vdupq_n_f32(0.5f);

    uint32x4_t is_positive = vcgeq_f32(x, vdupq_n_f32(0.0f));

    float32x4_t offset = vbslq_f32(is_positive, half, vnegq_f32(half));
    float32x4_t adjusted = vaddq_f32(x, offset);

    return vcvtq_s32_f32(adjusted);
}

// Function to perform per-tensor quantization with zero-point = 0
void  QuantizeInput(const float *in, uint8_t* out, float *scales,
                    uint32_t a_rows, uint32_t a_cols) {
  for (uint32_t i = 0; i < a_rows; i ++) {
    float max_abs = std::abs(in[i * a_cols]);
    // Loop through the array of pointers and calculate min/max
    for (uint32_t j = 0; j < a_cols; j ++) {
      if (max_abs < std::abs(in[i * a_cols + j])) {
        max_abs = std::abs(in[i * a_cols + j]);
      }
    }

    scales[i] = max_abs / 127;

    static constexpr int32_t min_val = std::numeric_limits<uint8_t>::min();
    static constexpr int32_t max_val = std::numeric_limits<uint8_t>::max();
    const int32_t zero_point = 128;

    uint32_t j = 0;
    const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scales[i]);
    const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
    const int32x4_t min_val_dup = vdupq_n_s32(min_val);
    const int32x4_t max_val_dup = vdupq_n_s32(max_val);

    for (; j <= a_cols - 8; j += 8) {
      const float* src_data_ptr = in + i * a_cols + j;
      float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
      float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

      input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
      input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

      int32x4_t casted_val_0 = roundq_s32_f32(input_val_0);
      int32x4_t casted_val_1 = roundq_s32_f32(input_val_1);

      casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
      casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

      // Clamp the values to fit the target type's range.
      casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
      casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
      casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
      casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

      const uint16x4_t narrowed_val_0 = vqmovun_s32(casted_val_0);
      const uint16x4_t narrowed_val_1 = vqmovun_s32(casted_val_1);
      const uint16x8_t combined_val = vcombine_u16(narrowed_val_0, narrowed_val_1);
      const uint8x8_t combined_val_narrowed = vmovn_u16(combined_val);
      vst1_u8(out + i * a_cols + j, combined_val_narrowed);
    }
  }

  return;
}

void DequantizeOutput(const int32_t *in, float* out, float* scales,
                      uint32_t a_batch, uint32_t a_rows, uint32_t b_rows) {
  for (uint32_t b = 0; b < a_batch; b ++) {
    for (uint32_t i = 0; i < a_rows; i ++) {
      for (uint32_t j = 0; j <= b_rows - 4; j += 4) {
        uint32_t idx = b * a_rows * b_rows + i * b_rows + j;
        int32x4_t vq = vld1q_s32(in + idx);

        float32x4_t vf = vcvtq_f32_s32(vq);
        float32x4_t vscale = vdupq_n_f32(scales[i]);
        float32x4_t vres = vmulq_f32(vf, vscale);

        vst1q_f32(out + idx, vres);
      }
    }
  }
}

Status MatMulNBits::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                            /*out*/ bool& is_packed,
                            /*out*/ PrePackedWeights* prepacked_weights) {
  try {
    switch (input_idx) {
      case InputIndex::IN_A:
        break;
      case InputIndex::IN_B:{

        m_handle = neutronAlloc->getMemoryHandle();

        //unpacked_b_ data
        unpacked_b_ = static_cast<const uint8_t*>(tensor.DataRaw());
        b_size_ = tensor.SizeInBytes();
        break;
      }
      case InputIndex::SCALES:{
#ifdef USE_NBITS_KERNEL
        const uint32_t MAGIC_WORD = 0x20250918;
        uint32_t magic = *((uint32_t*)unpacked_b_);

        int decode_bias_len = N_ * blocks_per_col_ * sizeof(int8_t);
        int decode_scale_len = N_ * blocks_per_col_ * sizeof(int16_t);
        int bias_len =  N_ * sizeof(int32_t);
        int factor_len = N_ * sizeof(int32_t);
        int ilength = 16 * sizeof(uint8_t);

        if (offline_packed_ || magic == MAGIC_WORD) {
          // ---- Offline prepacked path ----
          uint32_t weight_len = *((uint32_t*)unpacked_b_ + 1);
          uint32_t compress_num = *((uint32_t*)unpacked_b_ + 2);
          compress_num_ = compress_num;

          uint32_t total_len = ALIGN16_SIZE(16*sizeof(uint32_t)) + ALIGN16_SIZE(decode_bias_len) + ALIGN16_SIZE(decode_scale_len)
                             + ALIGN16_SIZE(bias_len) + ALIGN16_SIZE(factor_len) + ALIGN16_SIZE(weight_len)
                             + ALIGN16_SIZE(compress_num*4) + ALIGN16_SIZE(ilength);

          m_buffer = neutronAlloc->Alloc(total_len, m_handle);

          m_header = (uint32_t*)m_buffer;
          memset(m_header, 0, 16 * sizeof(uint32_t));
          clean_cache(m_header,  16 * sizeof(uint32_t));

          m_decode_bias  =   (int8_t*)((int8_t*)m_buffer       + ALIGN16_SIZE(16*sizeof(uint32_t)));
          m_decode_scale =  (int16_t*)((int8_t*)m_decode_bias  + ALIGN16_SIZE(decode_bias_len));
          m_b_bias       =  (int32_t*)((int8_t*)m_decode_scale + ALIGN16_SIZE(decode_scale_len));
          m_b_factors    = (uint32_t*)((int8_t*)m_b_bias       + ALIGN16_SIZE(bias_len));
          m_b_neutron    =   (int8_t*)((int8_t*)m_b_factors    + ALIGN16_SIZE(factor_len));
          m_compress_len =  (int32_t*)((int8_t*)m_b_neutron    + ALIGN16_SIZE(weight_len));
          m_decode_input =  (uint8_t*)((int8_t*)m_compress_len + ALIGN16_SIZE(compress_num*4));

          // weight layout
          // magic_word      weight_length   compress_lengths_number
          // decode_biases   decode_scales   bias   factors
          // compress_weight compress_lengths
          int offset = 12;
          memcpy(m_decode_bias,  unpacked_b_ + offset, decode_bias_len);

          offset += decode_bias_len;
          memcpy(m_decode_scale, unpacked_b_ + offset, decode_scale_len);

          offset += decode_scale_len;
          memcpy(m_b_bias,       unpacked_b_ + offset, bias_len);

          offset += bias_len;
          memcpy(m_b_factors,    unpacked_b_ + offset, factor_len);

          offset += factor_len;
          memcpy(m_b_neutron,    unpacked_b_ + offset, weight_len);

          offset += weight_len;
          memcpy(m_compress_len, unpacked_b_ + offset, compress_num * 4);
          compress_offset_ = offset + weight_len;

          clean_cache(m_b_neutron,    weight_len);
          clean_cache(m_decode_bias,  decode_bias_len);
          clean_cache(m_decode_scale, decode_scale_len);
          clean_cache(m_b_factors,    factor_len);
          clean_cache(m_b_bias,       bias_len);
          clean_cache(m_compress_len, compress_num * 4);
          clean_cache(m_decode_input, ilength);
        } else {
          // ---- Inline prepack for raw ONNX models ----
          const float* scalesData = static_cast<const float*>(tensor.DataRaw());

          // Allocate temp buffers for decode data
          int16_t*  decodeScales = (int16_t*)malloc(N_ * blocks_per_col_ * sizeof(int16_t));
          int8_t*   decodeBias   = (int8_t*)malloc(N_ * blocks_per_col_ * sizeof(int8_t));
          uint32_t* bFactorsTmp  = (uint32_t*)malloc(N_ * sizeof(uint32_t));
          int32_t*  bRowSum      = (int32_t*)malloc(N_ * sizeof(int32_t));

          // Calculate decode scales, biases, factors, and row sums
          CalculateDecodeData(unpacked_b_, scalesData, decodeScales, decodeBias,
                              bFactorsTmp, bRowSum, N_, blocks_per_col_, block_size_);

          // Unpack 4-bit B to int8 (zero point = 8)
          int8_t* B_int8 = (int8_t*)malloc(N_ * K_);
          for (size_t i = 0; i < N_ * K_; i++) {
            uint8_t byte_val = unpacked_b_[i / 2];
            uint8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
            B_int8[i] = (int8_t)nibble - 8;
          }

          // Adjust B_int8 for negative scales (where decodeBias == 1)
          for (size_t i = 0; i < N_; i++) {
            for (size_t j = 0; j < K_; j++) {
              size_t block_idx = j / block_size_;
              if (decodeBias[i * blocks_per_col_ + block_idx] == 1) {
                B_int8[i * K_ + j] = -B_int8[i * K_ + j] - 1;
              }
            }
          }

          // Call prepack to pack weights for Neutron
          PrepackCfg cfg;
          cfg.rearrange     = true;
          cfg.miniWeights   = true;
          cfg.weightBits    = static_cast<int>(nbits_);
          cfg.groupSize     = static_cast<int>(block_size_);
          cfg.useDecodeBias = true;
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

          PrePackWeight(B_int8, static_cast<int>(N_), static_cast<int>(K_), &cfg, &pckOut);

          // Extract tiling info and packed sizes
          int channelDensity = pckOut.tilingInfo.channelDensity;
          int pNumNeutrons   = pckOut.tilingInfo.numNeutrons;
          int divisions      = pckOut.tilingInfo.divisions;
          size_t weight_len  = pckOut.Bpacked->size;
          int32_t compress_num = static_cast<int32_t>(pckOut.lengths->size);
          compress_num_ = compress_num;

          // Organize decode scales into Neutron layout
          int16_t* organized_scales = (int16_t*)malloc(decode_scale_len);
          void* temp_scales = malloc(decode_scale_len);
          OrganizeDecodeData<int16_t>(decodeScales, organized_scales, temp_scales,
                                      static_cast<int>(N_), static_cast<int>(K_),
                                      channelDensity, pNumNeutrons,
                                      static_cast<int>(block_size_), divisions,
                                      static_cast<int>(nbits_), 16);
          free(temp_scales);
          free(decodeScales);

          // Organize decode biases into Neutron layout
          int8_t* organized_biases = (int8_t*)malloc(decode_bias_len);
          void* temp_biases = malloc(decode_bias_len);
          OrganizeDecodeData<int8_t>(decodeBias, organized_biases, temp_biases,
                                     static_cast<int>(N_), static_cast<int>(K_),
                                     channelDensity, pNumNeutrons,
                                     static_cast<int>(block_size_), divisions,
                                     static_cast<int>(nbits_), 16);
          free(temp_biases);
          free(decodeBias);

          // Allocate Neutron memory using actual sizes from prepack output
          uint32_t total_len = ALIGN16_SIZE(16*sizeof(uint32_t)) + ALIGN16_SIZE(decode_bias_len) + ALIGN16_SIZE(decode_scale_len)
                             + ALIGN16_SIZE(bias_len) + ALIGN16_SIZE(factor_len) + ALIGN16_SIZE(weight_len)
                             + ALIGN16_SIZE(compress_num*4) + ALIGN16_SIZE(ilength);

          m_buffer = neutronAlloc->Alloc(total_len, m_handle);

          m_header = (uint32_t*)m_buffer;
          memset(m_header, 0, 16 * sizeof(uint32_t));
          clean_cache(m_header,  16 * sizeof(uint32_t));

          m_decode_bias  =   (int8_t*)((int8_t*)m_buffer       + ALIGN16_SIZE(16*sizeof(uint32_t)));
          m_decode_scale =  (int16_t*)((int8_t*)m_decode_bias  + ALIGN16_SIZE(decode_bias_len));
          m_b_bias       =  (int32_t*)((int8_t*)m_decode_scale + ALIGN16_SIZE(decode_scale_len));
          m_b_factors    = (uint32_t*)((int8_t*)m_b_bias       + ALIGN16_SIZE(bias_len));
          m_b_neutron    =   (int8_t*)((int8_t*)m_b_factors    + ALIGN16_SIZE(factor_len));
          m_compress_len =  (int32_t*)((int8_t*)m_b_neutron    + ALIGN16_SIZE(weight_len));
          m_decode_input =  (uint8_t*)((int8_t*)m_compress_len + ALIGN16_SIZE(compress_num*4));

          // Compute bias from row sums: m_b_bias[i] = -bRowSum[i] * 128
          for (size_t i = 0; i < N_; i++) {
            m_b_bias[i] = -bRowSum[i] * 128;
          }

          // Copy all data into Neutron layout
          memcpy(m_decode_bias,  organized_biases, decode_bias_len);
          memcpy(m_decode_scale, organized_scales, decode_scale_len);
          memcpy(m_b_factors,    bFactorsTmp, factor_len);
          memcpy(m_b_neutron,    pckOut.Bpacked->data, weight_len);
          memcpy(m_compress_len, pckOut.lengths->data, compress_num * 4);

          clean_cache(m_b_neutron,    weight_len);
          clean_cache(m_decode_bias,  decode_bias_len);
          clean_cache(m_decode_scale, decode_scale_len);
          clean_cache(m_b_factors,    factor_len);
          clean_cache(m_b_bias,       bias_len);
          clean_cache(m_compress_len, compress_num * 4);
          clean_cache(m_decode_input, ilength);

          // Free temp buffers
          free(organized_scales);
          free(organized_biases);
          free(bFactorsTmp);
          free(bRowSum);
          free(B_int8);
          dyn8_free(pckOut.Bpacked);
          dyn32_free(pckOut.lengths);
        }
#endif

#ifdef USE_8BITS_KERNEL
        int blength = N_ * blocks_per_col_ * block_size_;
        int flength = N_ * sizeof(int32_t);
        int biaslength =  N_ * sizeof(int32_t);
        m_b_neutron = (int8_t*) neutronAlloc->Alloc(blength, m_handle);
        m_b_factors = (uint32_t *)neutronAlloc->Alloc(flength, m_handle);
        int8_scale_ = (float *)neutronAlloc->Alloc(flength, m_handle);
        m_b_row_sum = (int32_t *)neutronAlloc->Alloc(biaslength, m_handle);
        ReQuantizeWeight(unpacked_b_, m_b_neutron, scalesData, m_b_factors, N_,
                         blocks_per_col_, block_size_);
        clean_cache(m_b_neutron, blength);
        clean_cache(m_b_factors, flength);
        clean_cache(int8_scale_, flength);

        for (uint32_t i = 0; i < N_; i ++){
          int32_t row_sum = 0;
          for (uint32_t j = 0; j < K_; j ++) {
            row_sum += *(m_b_neutron + i * K_ + j);
          }
          m_b_row_sum[i] = row_sum;
        }

        clean_cache(m_b_row_sum, biaslength);
        m_b_bias = (int32_t *)neutronAlloc->Alloc(biaslength, m_handle);
        for (uint32_t i = 0; i < N_; i++){
          m_b_bias[i] = (int32_t)( - m_b_row_sum[i] * 128 );
        }

        clean_cache(m_b_bias, biaslength);
#endif

#ifdef USE_PURE_C
        int blength2 = N_ * blocks_per_col_ * block_size_ * sizeof(float);
        float_b = (float *)neutronAlloc->Alloc(blength2, m_handle);
        DequantizeWeight(unpacked_b_, float_b, scalesData, N_, blocks_per_col_, block_size_);
        clean_cache(float_b, blength2);
#endif
        break;
      }
      case InputIndex::ZERO_POINTS:
        throw std::invalid_argument("NeutronEP:MatMulNBits don't support zero-points");
        break;
    }
  } catch (const std::exception &e) {
    // Do not delegate this instance if out of memory
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }
  return Status::OK();
}

Status MatMulNBits::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                                  /*out*/ bool& used_shared_buffers) {
  return Status::OK();
}

Status MatMulNBits::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(InputIndex::IN_A);

  TensorShape b_shape({static_cast<int64_t>(N_), static_cast<int64_t>(K_)});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  uint32_t a_cols = static_cast<uint32_t>(helper.K());
  uint32_t a_rows = static_cast<uint32_t>(helper.M());
  uint32_t b_rows = static_cast<uint32_t>(helper.N());
  uint32_t num_matmuls = helper.OutputOffsets().size();;

  if (!m_header) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "NeutronEP:MatMulNBits falied to init.");
  }

  Tensor* y = ctx->Output(OutputIndex::OUT_Y, helper.OutputShape());
  auto  a_data = static_cast<const float*>(a->DataRaw());
  float* y_data = static_cast<float*>(y->MutableDataRaw());

  for (size_t batch = 0; batch < num_matmuls; batch++) {
    neutronAlloc->pushMemoryState(m_handle);

#if defined(USE_NBITS_KERNEL) || defined(USE_8BITS_KERNEL)
    uint32_t a_size = a_rows * a_cols * sizeof(uint8_t);
    uint8_t *a_neutron = (uint8_t *) neutronAlloc->AllocReserved(a_size, m_handle);
    uint32_t y_size = a_rows * b_rows * sizeof(int32_t);
    int32_t *y_neutron = (int32_t *) neutronAlloc->AllocReserved(y_size, m_handle);
    float *input_scales = (float *) neutronAlloc->AllocReserved(a_rows * sizeof(float), m_handle);

    QuantizeInput(a_data + helper.LeftOffsets()[batch], a_neutron, input_scales, a_rows, a_cols);
    clean_cache(a_neutron, a_size);
#endif

#ifdef USE_NBITS_KERNEL
    //TODO: verify if need to set to 1 every invoke
    memset(m_decode_input, 1, 16);
    clean_cache(m_decode_input, 16);

    m_header[0] = (uint8_t *)m_compress_len  - (uint8_t *)m_header;
    m_header[1] = (uint8_t *)m_decode_bias - (uint8_t *)m_header;
    m_header[2] = a_rows;
    m_header[3] = a_cols;
    m_header[4] = b_rows | (1 << 18);
    m_header[5] = (uint8_t *)a_neutron - (uint8_t *)m_header;
    m_header[6] = (uint8_t *)m_b_neutron - (uint8_t *)m_header;
    m_header[7] = (uint8_t *)m_b_bias - (uint8_t *)m_header;
    m_header[8] = (uint8_t *)m_b_factors - (uint8_t *)m_header;
    m_header[9] = (uint8_t *)y_neutron - (uint8_t *)m_header;
    m_header[10] = GetMatmulTypeFlag(true, a->IsDataType<int8_t>()); // packed
    m_header[11] = 4; // result num bytes
    m_header[12] = nbits_; // Weight Bits
    m_header[13] = block_size_; // Group Size
    m_header[14] = (uint8_t *)m_decode_scale - (uint8_t *)m_header;
    m_header[15] = (uint8_t *)m_decode_input - (uint8_t *)m_header;
#endif

#ifdef USE_8BITS_KERNEL
    m_header[0] = GetMatmulTypeFlag(true, a->IsDataType<int8_t>());
    m_header[1] = 0;
    m_header[2] = a_rows;
    m_header[3] = a_cols;
    m_header[4] = b_rows;
    m_header[5] = (uint8_t *)a_neutron - (uint8_t *)m_header;
    m_header[6] = (uint8_t *)m_b_neutron - (uint8_t *)m_header;
    m_header[7] = (uint8_t *)m_b_bias - (uint8_t *)m_header;
    m_header[8] = (uint8_t *)m_b_factors - (uint8_t *)m_header;
    m_header[9] = (uint8_t *)y_neutron - (uint8_t *)m_header;
    m_header[10] = 0; // m_y_zp;
    m_header[11] = 4; // result num bytes
    m_header[12] = 8; // Weight Bits
    m_header[13] = -1; // Group Size
    m_header[14] = 0;
    m_header[15] = 0;
#endif

#if defined(USE_NBITS_KERNEL) || defined(USE_8BITS_KERNEL)
    NeutronError ret = ENONE;
    ret = matmul((const void *)m_header, 16 * sizeof(uint32_t), (const void *)a_neutron, a_size, 
		 (const void *)y_neutron, y_size, m_handle);
    if (ret != ENONE) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "NeutronEP:MatMulNBits falied to call kernel");
    }
    clean_cache(y_neutron, y_size);
    DequantizeOutput(y_neutron, y_data + helper.OutputOffsets()[batch], input_scales,
         num_matmuls, a_rows, b_rows);
#endif
#ifdef USE_PURE_C
    CMatMul(a_data, float_b, y_data + helper.OutputOffsets()[batch], a_rows, K_, N_);
#endif

    neutronAlloc->popMemoryState(m_handle);
  } //batch

  return Status::OK();
}
}  // namespace neutron
}  // namespace onnxruntime
