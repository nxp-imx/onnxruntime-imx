// Copyright (c) NXP. All rights reserved.

#pragma once

#include <cstdlib>
#include <stdint.h>
#include <vector>

namespace onnxruntime {

constexpr size_t kDefaultTensorAlignment = 64;
constexpr size_t kFullNeutronBufferSize = 945 * 1024 * 1024 - 15;
constexpr size_t kBoundaryNeutronBufferSize = 512 * 1024 * 1024;
constexpr size_t kReservedNeutronBufferSize = 96 * 1024 * 1024;
constexpr size_t kNeutronNumHandles = 2;

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
  uint8_t* p_;
  uint8_t* neutron_ptr_[kNeutronNumHandles];
  size_t   neutron_size_[kNeutronNumHandles];
  std::vector<uint8_t*> past_ptrs_;
  std::vector<size_t> past_sizes_;
};

}  // namespace onnxruntime
