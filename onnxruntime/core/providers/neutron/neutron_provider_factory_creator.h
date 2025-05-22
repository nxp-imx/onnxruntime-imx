// Copyright 2025 NXP

#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/providers/neutron/neutron_provider_factory.h"

namespace onnxruntime {
struct NeutronProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(NeutronProviderOptions neutron_options);
};
}  // namespace onnxruntime
