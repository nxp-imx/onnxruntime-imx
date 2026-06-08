// Copyright 2026 NXP

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

typedef struct {
    int* len;
    int* code;
    int* data_len;
    int size;
    int capacity;
} CodeTable;

typedef struct {
    uint8_t* compressed;
    int compressed_capacity;
    CodeTable code_table;
    size_t length;
} CoreCompResult;

typedef struct {
    int channelDensity;
    int lineDensity;
    int numNeutrons;
    bool bPingPong;
    int divisions;
} TilingResult;

typedef struct {
    int8_t* data;
    size_t  size;
    size_t  cap;
} Dyn8;
typedef struct {
    int32_t* data;
    size_t   size;
    size_t   cap;
} Dyn32;

typedef struct {
    Dyn8* Bpacked;
    Dyn32* lengths;
    TilingResult tilingInfo;
} PrepackOut;

typedef struct {
    bool rearrange;
    bool miniWeights;
    int weightBits;
    int groupSize;
    bool useDecodeBias;
    bool compress;
    int numMacs;
    int numNeutrons;
    int tcmSize;
    int numBanks;
} PrepackCfg;

#define IDX(i,j,cols) ( ((size_t)(i) * (size_t)(cols)) + (size_t)(j) )

#ifdef __cplusplus
extern "C" {
#endif

void dyn8_append(Dyn8* b, const int8_t* src, size_t n);
void dyn8_free(Dyn8* b);
void dyn32_push(Dyn32* b, int32_t v);
void dyn32_free(Dyn32* b);

void init_code_table(CodeTable* ct, int capacity);
void free_code_table(CodeTable* ct);
void copy_code_table(CodeTable* dest, const CodeTable* src);
int8_t find_min_int8(const int8_t* arr, int size);
void free_core_comp_result(CoreCompResult* result);
void extract_patterned_rows(const int8_t* matrix,
                        size_t rows,
                        size_t cols,
                        size_t start,
                        size_t step,
                        size_t length,
                        size_t count,
                        int8_t **out_ptr,
                        size_t *out_len);
void fetch_unp_organize(const int8_t* packed, size_t packedN, int rowsB, int colsB,
                        int channelDensity, int MACs, int weightBits, int numNeutrons,
                        int8_t** outOrganized, size_t* outOrganizedN);
void weight_packer(const int8_t* B, int rowsB, int colsB,
                   int channelDensity, int MACs, int weightBits,
                   int8_t** outPacked, size_t* outPackedN);
void fetch_unp_organize(const int8_t* packed, size_t packedN, int rowsB, int colsB,
                        int channelDensity, int MACs, int weightBits, int numNeutrons,
                        int8_t** outOrganized, size_t* outOrganizedN);
TilingResult tiling_solver(int numTokens, int embeddings_in, int embeddings_out,
              int resNumBytes, int MACS, int neutrons,
              int tcm_size, int tcm_banks, int weightBits,
              bool decodeWeights, bool useDecodeBias, int groupSize);
CoreCompResult compress_weight_tensor(
    uintptr_t data_ptr,
    int data_size,
    CodeTable code_table,
    int num_decomp,
    int word_size,
    int buffer_size,
    int packet_size);
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
    int* outListSize);

void PrePackWeight(const int8_t* B, int rb, int cb, PrepackCfg* pckCfg, PrepackOut* pckOut);

#ifdef __cplusplus
}
#endif

namespace onnxruntime {
namespace neutron {
#define ALIGN16_SIZE(size) ((size + 0xf) & (~0xf))

void PrepareForQDQ(const TensorShape& input_shape,
                   const Tensor& scale,
                   const Tensor* zero_point_ptr,
                   int64_t axis,
                   int64_t& block_count,
                   int64_t& broadcast_dim,
                   int64_t& block_size);

uint32_t ScaleToNeutron(float scale_data);

int32_t
GetMatmulTypeFlag(bool packed, bool signedData);


}  // namespace neutron
}  // namespace onnxruntime
