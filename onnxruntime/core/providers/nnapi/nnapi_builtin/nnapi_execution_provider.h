// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/optional.h"
#include "core/framework/execution_provider.h"
#include "core/providers/nnapi/nnapi_provider_factory.h"

#include "core/common/inlined_containers_fwd.h"

namespace onnxruntime {
namespace nnapi {
class Model;
}

using Shape = InlinedVector<uint32_t>;

class NnapiExecutionProvider : public IExecutionProvider {
 public:
  explicit NnapiExecutionProvider(uint32_t nnapi_flags,
                                  const optional<std::string>& partitioning_stop_ops_list = {},
                                  const std::string& bypass_output_shape_str = "");

  virtual ~NnapiExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const IKernelLookup& /*kernel_lookup*/) const override;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
#endif

  uint32_t GetNNAPIFlags() const { return nnapi_flags_; }

  DataLayout GetPreferredLayout() const override;

 private:
  // The bit flags which define bool options for NNAPI EP, bits are defined as
  // NNAPIFlags in include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h
  const uint32_t nnapi_flags_;

  const std::unordered_set<std::string> partitioning_stop_ops_;

  // Bypass shape as inlined vector
  std::string bypass_output_shape_str_;
  Shape bypass_output_shape_;

  std::unordered_map<std::string, std::unique_ptr<onnxruntime::nnapi::Model>> nnapi_models_;
};
}  // namespace onnxruntime
