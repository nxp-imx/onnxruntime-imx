// Copyright (c) NXP. All rights reserved.

#pragma once

#include "core/framework/allocator.h"

#include <cstdlib>

#define DEFAULT_NEUTRON_ALLOCATOR_DEVICE_ID 0

namespace onnxruntime {


constexpr size_t kDefaultTensorAlignment = 64;
constexpr size_t kFullNeutronBufferSize = 945 * 1024 * 1024 - 15;

class NeutronAllocator : public IAllocator {
 public:
  NeutronAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;

  private:
    size_t neutron_buffer_size_{0};
    size_t neutron_used_size_{0};
    size_t neutron_last_chunk_{0};
    uint8_t* neutron_ptr_;
};

/* Placeholder for page-locked allocation */

class NeutronPinnedAllocator : public IAllocator {
 public:
  NeutronPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::NEUTRON_PINNED, device_id),
                          device_id, OrtMemTypeCPU)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  size_t allocated_{0};
};

}  // namespace onnxruntime
