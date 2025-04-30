// Copyright (c) NXP. All rights reserved.

#pragma once

#include "core/framework/execution_provider.h"


namespace onnxruntime {

class NeutronExecutionProvider : public IExecutionProvider {
 public:
  explicit NeutronExecutionProvider(uint32_t neutron_flags);
  virtual ~NeutronExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* /* resource_accountant */) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

 private:
  uint32_t neutron_flags_;
  uint32_t node_number_{0};
};

Status RegisterNeutronKernels(KernelRegistry& kernel_registry);

}  // namespace onnxruntime
