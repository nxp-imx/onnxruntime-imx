// Copyright 2026 NXP

#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

#ifdef __cplusplus
extern "C" {
#endif
/* compress_weight_tensor_grouped function */
int* CompressWeightTensorGrouped(
    int8_t* data,
    int data_size,
    int channelDensity,
    int numColsA,
    int weightBits,
    int num_decomp,
    int word_size,
    int buffer_size,
    int packet_size,
    bool compress,
    int8_t** outCompressed,
    size_t* outCompressedN,
    int* outListSize) {

    const int max_output_size = data_size * 2;
    int8_t* compressed_stream = (int8_t*)malloc(max_output_size);
    *outCompressed = compressed_stream;

    int unit_size_outer = channelDensity * numColsA * weightBits / 8;
    int unit_size = unit_size_outer;
    int splits_per_channelC = 1;

    if (compress) {
        int splits_per_channelC_tmp = (int)ceil(unit_size_outer / (256.0 * 1024));
        splits_per_channelC = 1;
        while (splits_per_channelC < splits_per_channelC_tmp) {
            splits_per_channelC *= 2;
        }
        unit_size = unit_size_outer / splits_per_channelC;
    }

    int cycles = unit_size_outer / unit_size;
    bool compressed_one = false;
    int iters = data_size / unit_size;
    int compressed_offset = 0;

    int* len_compressed_stream_list = (int*)malloc(iters * sizeof(int));
    *outListSize = iters;

    CodeTable current_code_table;
    init_code_table(&current_code_table, 0);

    for (int iter = 0; iter < iters; iter++) {
        if (iter > 0 && iter % cycles == 0) {
            if (!compressed_one) {
                /* replace_last_m_with_sum */
                if (cycles > 0 && cycles <= iter) {
                    int sum = 0;
                    for (int i = iter - cycles; i < iter; i++) {
                        sum += len_compressed_stream_list[i];
                    }
                    len_compressed_stream_list[iter - 1] = sum;
                }
            }
            compressed_one = false;
        }

        const int8_t* current_data_ptr = data + iter * unit_size;

        /* Find minimum */
        int8_t min_val = find_min_int8(current_data_ptr, unit_size);

        if (min_val > -128 && compress) {
            compressed_one = true;

            uintptr_t current_data_addr = (uintptr_t)current_data_ptr;
            CoreCompResult result = compress_weight_tensor(current_data_addr, unit_size, current_code_table,
                                                 num_decomp, word_size, buffer_size, packet_size);

            // NOTE: compress_weight_tensor takes CodeTable by value (shallow copy).
            // When num_decomp > 1, it internally frees the code_table's pointers via
            // free_code_table(&code_table). Since those pointers are shared with
            // current_code_table, we must NOT call free_code_table(&current_code_table)
            // here — that would be a double-free. Instead, reset pointers to NULL.
            current_code_table.code = NULL;
            current_code_table.len = NULL;
            current_code_table.data_len = NULL;
            current_code_table.size = 0;
            current_code_table.capacity = 0;
            copy_code_table(&current_code_table, &result.code_table);

            int8_t* current_compressed = (int8_t*)malloc(unit_size * 2);
            int current_compressed_offset = 0;

            /* Process code words */
            for (int i = 0; i < current_code_table.size; i++) {
                int code = current_code_table.code[i];
                int len = current_code_table.len[i];
                current_compressed[current_compressed_offset++] = (int8_t)(code << (8 - len));
            }

            /* Process code lengths */
            for (int i = 0; i < current_code_table.size; i++) {
                int len = current_code_table.len[i];
                current_compressed[current_compressed_offset++] = (int8_t)(((1 << len) - 1) << (8 - len));
            }

            /* Add compressed data */
            for (size_t i = 0; i < result.length; i++) {
                uint8_t val = result.compressed[i];
                current_compressed[current_compressed_offset++] = val > 127 ?
                    (int8_t)(val - 256) : (int8_t)val;
            }

            /* Add padding */
            int padding_needed = (16 - current_compressed_offset % 16) % 16;
            for (int i = 0; i < padding_needed; i++) {
                current_compressed[current_compressed_offset++] = 0;
            }

            /* Copy to main stream */
            memcpy(compressed_stream + compressed_offset, current_compressed, current_compressed_offset);
            compressed_offset += current_compressed_offset;

            len_compressed_stream_list[iter] = current_compressed_offset;

            free(current_compressed);
            free_core_comp_result(&result);

        } else {
            /* No compression - just copy data */
            memcpy(compressed_stream + compressed_offset, current_data_ptr, unit_size);
            compressed_offset += unit_size;

            free_code_table(&current_code_table);
            init_code_table(&current_code_table, 0);
            len_compressed_stream_list[iter] = unit_size;
        }
    }

    free_code_table(&current_code_table);
    *outCompressedN = compressed_offset;

    return len_compressed_stream_list;
}

// Prepack
void PrePackWeight(const int8_t* B, int rb, int cb, PrepackCfg* pckCfg, PrepackOut* pckOut)
{
    int ca = cb;
    int weightBits = pckCfg->weightBits;
    int groupSize = pckCfg->groupSize;
    bool useDecodeBias = pckCfg->useDecodeBias;
    bool rearrange = pckCfg->rearrange;
    bool miniWeights = pckCfg->miniWeights;
    bool compress = pckCfg->compress;
    int numMacs = pckCfg->numMacs;
    int numNeutrons = pckCfg->numNeutrons;
    int tcmSize = pckCfg->tcmSize;
    int numBanks = pckCfg->numBanks;
    TilingResult t = tiling_solver(1, ca, rb, 4, numMacs, numNeutrons, tcmSize, numBanks, weightBits, (groupSize>0), useDecodeBias, groupSize);
    pckOut->tilingInfo = t;
    int MACs = numMacs;

    int cd = t.channelDensity, nn = t.numNeutrons, divisions = t.divisions;
    Dyn8* B_stream = pckOut->Bpacked;
    Dyn32* lengths_all = pckOut->lengths;

    if (rearrange) {
        // REARRANGE
        for (int rows = 0; rows < rb; rows += cd*nn) {
            for (int cols = 0; cols < ca; cols += ca/divisions) {
                int rowsB = (cd*nn < rb - rows) ? cd*nn : rb - rows;
                int colsB = (ca/divisions < ca - cols) ? ca/divisions : ca - cols;
                // slice
                size_t sliceN = (size_t)rowsB * colsB;
                int8_t* slice = (int8_t*)malloc(sliceN);
                for (int r = 0; r < rowsB; ++r)
                    for (int c = 0; c < colsB; ++c)
                        slice[IDX(r,c,colsB)] = B[IDX(rows+r, cols+c, ca)];

                // pack -> organize -> compress
                int8_t *packed=NULL, *organized=NULL, *compressed=NULL;
                size_t packedN=0, organizedN=0, compressedN=0;

                // [VICTOR] Converted to C
                weight_packer(slice, rowsB, colsB, cd, MACs, weightBits, &packed, &packedN);

                // [VICTOR] Converted to C
                fetch_unp_organize(packed, packedN, rowsB, colsB, cd, MACs, weightBits, nn, &organized, &organizedN);
                int outListSize;
                int32_t* lenList = CompressWeightTensorGrouped(organized, organizedN, cd*nn, colsB,
                                               weightBits,
                                               16,8,96,32,
                                               compress, &compressed, &compressedN, &outListSize);


                // [VICTOR] Converted to C
                dyn8_append(B_stream, compressed, compressedN);
                for (int i = 0; i < outListSize; ++i) dyn32_push(lengths_all, lenList[i]);

                free(slice); free(packed); free(organized); free(compressed);
            }
        }
    } else {
        // NO REARRANGE
        if (miniWeights) {
            fprintf(stderr, "error. Cannot have miniweights without rearrange\n");
            return;
        }
        for (int rows = 0; rows < rb; rows += cd*nn) {
                const int8_t* slice = B + rows * ca;
                for (int div_count = 0; div_count < divisions; div_count ++) {
                    // pack -> organize -> compress
                    int8_t *packed=NULL, *organized=NULL, *compressed=NULL;
                    size_t packedN=0, organizedN=0, compressedN=0;

                    extract_patterned_rows(slice, rb, ca, div_count*cd/divisions,
                        cd, cd/divisions, nn, &packed, &packedN);
                    fetch_unp_organize(packed, packedN, cd*nn/divisions, ca, cd/divisions, MACs, weightBits, nn, &organized, &organizedN);
                    int outListSize;
                    int32_t* lenList = CompressWeightTensorGrouped(organized, organizedN, cd*nn/divisions, ca,
                                weightBits,
                            16,8,96,32,
                                compress, &compressed, &compressedN, &outListSize);

                    dyn8_append(B_stream, compressed, compressedN);
                    for (int i = 0; i < outListSize; ++i) dyn32_push(lengths_all, lenList[i]);

                    free(packed); free(organized); free(compressed);
                }
        }
    }
}
#ifdef __cplusplus
}
#endif

namespace onnxruntime {
namespace neutron {

/*
    From CPU Provider implementation
*/

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
