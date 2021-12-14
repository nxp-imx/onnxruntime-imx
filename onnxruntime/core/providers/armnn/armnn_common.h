// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020-2021 NXP
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "armnn/ArmNN.hpp"


namespace onnxruntime {
namespace armnn_ep {

armnn::TensorShape ArmNNTensorShape(const TensorShape& tensorShape, unsigned int extDim = 0);

typedef struct
{
  std::map<OpKernel*, armnn::NetworkId> layers;
  armnn::IRuntimePtr run = armnn::IRuntimePtr(nullptr, nullptr);
} Runtime;

}  // namespace armnn_ep
}  // namespace onnxruntime
