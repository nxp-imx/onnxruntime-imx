// Copyright (c) NXP. All rights reserved.

#include <sys/mman.h>
#include <cstddef>

#include "core/providers/neutron/neutron_allocator.h"
#include "core/common/logging/logging.h"
#if NEUTRON_AARCH64
#include "core/providers/neutron/platform/NeutronDriver.h"
#endif

namespace onnxruntime {

inline size_t getAlignedSize(uint64_t size) {
  uint64_t mod = size % kDefaultTensorAlignment;
  return mod ? (size + kDefaultTensorAlignment - mod) : size;
}

NeutronStackAllocator::NeutronStackAllocator() {
  NeutronError ret = allocateBuffer((uint32_t)kFullNeutronBufferSize, (void **)&p_);
  if (ret != ENONE) {
    throw std::bad_alloc();
  }
  neutron_ptr_[0] = p_;
  neutron_ptr_[1] = p_ + kBoundaryNeutronBufferSize;
  neutron_size_[0] = kBoundaryNeutronBufferSize;
  neutron_size_[1] = kFullNeutronBufferSize - kBoundaryNeutronBufferSize;
  printf("[NeutronStackAllocator::NeutronStackAllocator] Allocating buffers from %p of %ld and %ld Bytes\n",p_,neutron_size_[0],neutron_size_[1]);
}

size_t NeutronStackAllocator::getMemoryHandle() {
  size_t largest_pos = 0;
  for (size_t i = 1; i < kNeutronNumHandles; i++)
    if (neutron_size_[i] > neutron_size_[largest_pos])
      largest_pos = i;
  return largest_pos;
}

void* NeutronStackAllocator::Alloc(size_t size, size_t handle) {
  printf("[NeutronStackAllocator::Alloc] Allocating %ld bytes from handle %ld\n",size,handle);
  size = getAlignedSize(size);
  if (neutron_size_[handle] < (kReservedNeutronBufferSize + size)) {
    throw std::bad_alloc();
  }
  printf("[NeutronStackAllocator::Alloc] Allocated %ld bytes from handle %ld\n",size,handle);
  void* tmp = neutron_ptr_[handle];
  neutron_ptr_[handle] += size;
  neutron_size_[handle] -= size;
  return tmp;
}  

void* NeutronStackAllocator::AllocReserved(size_t size, size_t handle) {
  printf("[NeutronStackAllocator::Alloc] Allocating %ld reserved bytes from handle %ld\n",size,handle);
  size = getAlignedSize(size);
  if (neutron_size_[handle] < size) {
    throw std::bad_alloc();
  }
  printf("[NeutronStackAllocator::Alloc] Allocated %ld reserved bytes from handle %ld\n",size,handle);
  void* tmp = neutron_ptr_[handle];
  neutron_ptr_[handle] += size;
  neutron_size_[handle] -= size;
  return tmp;
}  

void NeutronStackAllocator::pushMemoryState(size_t handle) {
  past_ptrs_.push_back(neutron_ptr_[handle]);
  past_sizes_.push_back(neutron_size_[handle]);
}

void NeutronStackAllocator::popMemoryState(size_t handle) {
  neutron_ptr_[handle] = past_ptrs_.back();
  past_ptrs_.pop_back();
  neutron_size_[handle] = past_sizes_.back();
  past_sizes_.pop_back();
}

NeutronStackAllocator::~NeutronStackAllocator() {
  releaseBuffer(p_);
}


}  // namespace onnxruntime
