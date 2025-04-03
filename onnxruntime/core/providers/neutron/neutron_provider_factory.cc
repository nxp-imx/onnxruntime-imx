// Copyright (c) NXP. All rights reserved.

#include "core/providers/neutron/neutron_provider_factory.h"
#include "neutron_execution_provider.h"
#include "neutron_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct NeutronProviderFactory : IExecutionProviderFactory {
  NeutronProviderFactory(uint32_t neutron_flags)
      : neutron_flags_(neutron_flags) {}
  ~NeutronProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  uint32_t neutron_flags_;
};

std::unique_ptr<IExecutionProvider> NeutronProviderFactory::CreateProvider() {
  return std::make_unique<NeutronExecutionProvider>(neutron_flags_);
}

std::shared_ptr<IExecutionProviderFactory> NeutronProviderFactoryCreator::Create(uint32_t neutron_flags) {
  return std::make_shared<onnxruntime::NeutronProviderFactory>(neutron_flags);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Neutron,
                    _In_ OrtSessionOptions* options, uint32_t neutron_flags) {
  options->provider_factories.push_back(onnxruntime::NeutronProviderFactoryCreator::Create(neutron_flags));
  return nullptr;
}
