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


void* NeutronAllocator::Alloc(size_t size) {
#if NEUTRON_AARCH64
  NeutronError ret;
  auto aligned_size = getAlignedSize(size);
  if (!neutron_buffer_size_) {
    ret = allocateBuffer((uint32_t)kFullNeutronBufferSize, (void **)&neutron_ptr_);
    if (ret != ENONE) {
      ORT_THROW_EX(std::bad_alloc);
    }
    neutron_buffer_size_ = kFullNeutronBufferSize;
    neutron_used_size_ = aligned_size;
    neutron_last_chunk_ = aligned_size;
    printf("[Neutron Allocator] Initial allocation of %zu bytes in: %p \n", aligned_size, neutron_ptr_);
    return neutron_ptr_;

  } else if (neutron_used_size_ + aligned_size < neutron_buffer_size_) {
    auto ptr = neutron_ptr_ + neutron_used_size_;
    neutron_used_size_ += aligned_size;
    neutron_last_chunk_ = aligned_size;
    printf("[Neutron Allocator] Allocated %zu bytes in: %p \n", aligned_size, ptr);
    return ptr;

  } else {
    ORT_THROW_EX(std::bad_alloc);
  }
#else
  (void)size;
  return NULL;
#endif
}


void NeutronAllocator::Free(void* p) {
#if NEUTRON_AARCH64

  if (neutron_buffer_size_ && neutron_last_chunk_ ) {
    printf("[Neutron Allocator] Freeing last chunk of size %zu \n", neutron_used_size_);
    neutron_used_size_ -= neutron_used_size_;
  } else {
    printf("[Neutron Allocator] Releasing buffer at: %p \n", p);
    releaseBuffer(p);
    neutron_buffer_size_ = 0;
    neutron_used_size_ = 0;
    neutron_ptr_ = 0;
  }


#else
  (void)p;
#endif
}

/*
  NeutronPinned:
    Allocate locked memory. In case copies are required,
    we might be able to implement faster transfers (DMA goes broom).
    This approach is used by other EPs to manage memory transfers (CUDA).
    Locking requires page size, not really optimal.

    Note: proper pinned memory might be implemented in Neutron driver,
    using MAP_FIXED and explicit addresses
*/

void* NeutronPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;

  if (allocated_) {
    ORT_THROW_EX(std::bad_alloc);
  }
  p = mmap(NULL, size,
      PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED,
      -1, 0);

  if (p == MAP_FAILED) {
    ORT_THROW_EX(std::bad_alloc);
  }
  allocated_ = size;
  LOGS_DEFAULT(VERBOSE) <<
      "[NeutronPinned] Mapping " << size << " bytes in " << p ;
  return p;
}

void NeutronPinnedAllocator::Free(void* p) {
    if(!allocated_) {
      ORT_THROW_EX(std::bad_alloc);
    }
    LOGS_DEFAULT(VERBOSE) << "[NeutronPinned] Unmapping " << allocated_ << " bytes in " << p ;
    munmap(p, allocated_);
}

}  // namespace onnxruntime
