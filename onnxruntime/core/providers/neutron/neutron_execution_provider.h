// Copyright (c) NXP. All rights reserved.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

class NeutronExecutionProvider : public IExecutionProvider {
 public:
  NeutronExecutionProvider(uint32_t neutron_flags);
  virtual ~NeutronExecutionProvider();

  const uint32_t neutron_flags_;

 private:
};
}  // namespace onnxruntime
