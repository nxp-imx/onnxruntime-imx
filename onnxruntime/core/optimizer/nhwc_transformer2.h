// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020-2021 NXP
// Licensed under the MIT License.

#if defined(USE_ACL) || defined(USE_ARMNN)
#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class NhwcTransformer2

Transformer that optimizes the graph by using NHWC nodes instead of NCHW nodes
and inserts nodes to reorder tensors as needed.
*/
class NhwcTransformer2 : public GraphTransformer {
 public:
  NhwcTransformer2(const std::vector<std::string>& registered_execution_providers) noexcept :
	GraphTransformer("NhwcTransformer2"),  registered_execution_providers_(registered_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  const std::vector<std::string>& registered_execution_providers_;
};

}  // namespace onnxruntime
#endif