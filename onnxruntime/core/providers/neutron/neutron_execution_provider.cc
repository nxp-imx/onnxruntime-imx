// Copyright (c) NXP. All rights reserved.

#include "core/providers/neutron/neutron_execution_provider.h"


namespace onnxruntime {

NeutronExecutionProvider::NeutronExecutionProvider(uint32_t neutron_flags)
    : IExecutionProvider{onnxruntime::kNeutronExecutionProvider, true},
      neutron_flags_(neutron_flags) {
}

NeutronExecutionProvider::~NeutronExecutionProvider() {}



}  // namespace onnxruntime
