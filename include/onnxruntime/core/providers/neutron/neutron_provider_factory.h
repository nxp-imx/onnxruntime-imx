// Copyright (c) NXP. All rights reserved.

#pragma once

#include "onnxruntime_c_api.h"


enum NeutronFlags {
  NEUTRON_FLAG_USE_NONE = 0x000,
  NEUTRON_FLAG_USE_ARENA = 0x001,

  NEUTRON_FLAG_LAST = NEUTRON_FLAG_USE_ARENA,
};

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Neutron,
                          _In_ OrtSessionOptions* options, uint32_t neutron_flags);

#ifdef __cplusplus
}
#endif
