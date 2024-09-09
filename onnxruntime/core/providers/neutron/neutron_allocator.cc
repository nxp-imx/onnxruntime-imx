// Copyright (c) NXP. All rights reserved.

#include <sys/mman.h>
#include <cstddef>

#include "core/providers/neutron/neutron_allocator.h"
#include "core/common/logging/logging.h"
#if NEUTRON_AARCH64
#include "core/providers/neutron/platform/NeutronDriver.h"
#endif

namespace onnxruntime {

inline size_t getAlignedSize(size_t size) {
  size_t mod = size % kDefaultTensorAlignment;
  return mod ? (size + kDefaultTensorAlignment - mod) : size;
}

NeutronStackAllocator::NeutronStackAllocator() {
  _Bool user = true;

  printf("NeutronEP: start %s memory allocation %ld MB\n", user ? "userspace" : "kernel", (kFullNeutronBufferSize/1024/1024));

  NeutronError ret = allocateBuffer(kFullNeutronBufferSize, (void **)&p_, true);
  if (ret != ENONE) {
    throw std::bad_alloc();
    return;
  }

#ifndef NDEBUG
  printf("NeutronEP: allocated memory %p %ld MB\n", p_ , (kFullNeutronBufferSize/1024/1024));
#endif

  for (size_t i = 0; i < kNeutronNumHandles; i++) {
      neutron_ptr_[i] = p_ + i * kBoundaryNeutronBufferSize;
      size_t rest = kFullNeutronBufferSize - i * kBoundaryNeutronBufferSize;
      neutron_size_[i] =  rest >=  kBoundaryNeutronBufferSize ? kBoundaryNeutronBufferSize : (uint32_t) rest;

      printf("NeutronEP: allocated handle[%ld] %p %ld MB\n", i, neutron_ptr_[i], (long)(neutron_size_[i]/1024/1024));
  }
}

size_t NeutronStackAllocator::getMemoryHandle() {
  size_t largest_pos = 0;
  for (size_t i = 1; i < kNeutronNumHandles; i++)
    if (neutron_size_[i] > neutron_size_[largest_pos])
      largest_pos = i;
  return largest_pos;
}

void* NeutronStackAllocator::Alloc(size_t size, size_t handle) {

  if (p_ == NULL) {
    throw std::bad_alloc();
  }

  size = getAlignedSize(size);
  if (neutron_size_[handle] < (kReservedNeutronBufferSize + size)) {
#ifndef NDEBUG
    printf("NeutronEP: %s handle[%02ld] %p 0x%08lx bytes, remaining 0x%08lx bytes\n",
      "ToCPU", handle, neutron_ptr_[handle], size, neutron_size_[handle] - kReservedNeutronBufferSize);
#endif
    throw std::bad_alloc();
  }

#ifndef NDEBUG
    static int i = 1;
    printf("NeutronEP: %d allocated handle[%02ld] %p 0x%08lx bytes, remaining 0x%08lx bytes\n",
       i, handle, neutron_ptr_[handle], size, neutron_size_[handle] - size - kReservedNeutronBufferSize);
    i++;
#endif

    void* tmp = neutron_ptr_[handle];
    neutron_ptr_[handle] += size;
    neutron_size_[handle] -= size;

    return tmp;
}

void* NeutronStackAllocator::AllocReserved(size_t size, size_t handle) {
  size = getAlignedSize(size);
  if (neutron_size_[handle] < size) {
#ifndef NDEBUG
    printf("NeutronEP: reservation exception handle[%ld] %p %10ld bytes, space %12ld bytes\n",
       handle, neutron_ptr_[handle], size, neutron_size_[handle]);
#endif
    throw std::bad_alloc();
  }

#ifndef NDEBUG
  printf("NeutronEP: reserved handle[%ld] %p %10ld bytes, remaining handle %12ld bytes\n",
      handle, neutron_ptr_[handle], size, neutron_size_[handle] - size);
#endif

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

#ifndef NDEBUG
  printf("NeutronEP: restored handle[%ld] %p %10ld bytes, remaining %12ld bytes\n",
      handle, neutron_ptr_[handle], (long)0, neutron_size_[handle]);
#endif
}

NeutronStackAllocator::~NeutronStackAllocator() {
  releaseBuffer(p_);
}


}  // namespace onnxruntime
