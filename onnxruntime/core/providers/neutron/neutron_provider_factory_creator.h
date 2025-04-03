// Copyright (c) NXP. All rights reserved.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct NeutronProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(uint32_t neutron_flags);
};
}  // namespace onnxruntime
