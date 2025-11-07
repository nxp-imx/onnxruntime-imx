// Copyright (c) NXP. All rights reserved.

#pragma once

#include "core/framework/execution_provider.h"


namespace onnxruntime {

class NeutronExecutionProvider : public IExecutionProvider {
 public:
  explicit NeutronExecutionProvider(uint32_t neutron_flags);
  virtual ~NeutronExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
  AllocatorPtr CreateNeutronAllocator(OrtDevice::DeviceId device_id);
  [[nodiscard]] OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

 private:
  uint32_t neutron_flags_;

};

Status RegisterNeutronKernels(KernelRegistry& kernel_registry);

}  // namespace onnxruntime
