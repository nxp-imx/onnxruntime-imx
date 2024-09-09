// Copyright (c) NXP. All rights reserved.

#pragma once

#include <cstdlib>
#include <stdint.h>
#include <vector>

#define NEUTRON_DDR_GB 1.5

namespace onnxruntime {

constexpr size_t kDefaultTensorAlignment = 64;
constexpr size_t kFullNeutronBufferSize =  NEUTRON_DDR_GB * 1024 * 1024 * 1024LL;
constexpr size_t kBoundaryNeutronBufferSize = 512 * 1024 * 1024;
constexpr size_t kReservedNeutronBufferSize = 128 * 1024 * 1024;
constexpr size_t kNeutronNumHandles = NEUTRON_DDR_GB * 2;

class NeutronStackAllocator {
public:
  // Constructor.
  NeutronStackAllocator();

  // The first operation. Picks the memory slot with most free space.
  size_t getMemoryHandle();

  // Memory allocation within given
  void* Alloc(size_t size, size_t handle);
  void* AllocReserved(size_t size, size_t handle);

  // Remember current allocations.
  void pushMemoryState(size_t handle);

  // Releases all allocations since the last push.
  void popMemoryState(size_t handle);

  ~NeutronStackAllocator();

private:
  uint8_t* p_{NULL};
  uint8_t* neutron_ptr_[kNeutronNumHandles];
  size_t   neutron_size_[kNeutronNumHandles];
  std::vector<uint8_t*> past_ptrs_;
  std::vector<size_t> past_sizes_;
};

}  // namespace onnxruntime
