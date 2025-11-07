// Copyright (c) NXP. All rights reserved.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_allocator.h"
#include "core/providers/neutron/neutron_execution_provider.h"

namespace onnxruntime {
namespace neutron {

/*
  Remember: using AllocatorPtr = std::shared_ptr<IAllocator>;
*/

class NeutronKernel : public OpKernel {
 public:
  explicit NeutronKernel(const OpKernelInfo& info)
      : OpKernel(info),
        provider_(const_cast<NeutronExecutionProvider*>(static_cast<const NeutronExecutionProvider*>(info.GetExecutionProvider()))) {
          
        }


 private:
  NeutronExecutionProvider* provider_;

};

}  // namespace neutron
}  // namespace onnxruntime
