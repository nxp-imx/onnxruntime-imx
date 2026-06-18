// Copyright 2026 NXP

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

#ifndef NEUTRON_HAS_DRIVER_BMM
#define NEUTRON_HAS_DRIVER_BMM 1
#endif
#if NEUTRON_HAS_DRIVER_BMM
#include "neutron/NeutronDriverBmm.h"
#endif

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

#ifdef __cplusplus
extern "C" {
#endif

void init_code_table(CodeTable* ct, int capacity);
void free_code_table(CodeTable* ct);
void copy_code_table(CodeTable* dest, const CodeTable* src);
int8_t find_min_int8(const int8_t* arr, int size);
void free_core_comp_result(CoreCompResult* result);

CoreCompResult compress_weight_tensor(
    uintptr_t data_ptr,
    int data_size,
    CodeTable code_table,
    int num_decomp,
    int word_size,
    int buffer_size,
    int packet_size);

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
