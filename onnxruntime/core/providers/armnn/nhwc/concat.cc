// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
// Licensed under the MIT License.

#include "core/providers/armnn/nhwc/concat.h"
#include "core/providers/common.h"
#include "core/framework/TensorSeq.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "core/providers/armnn/armnn_fwd.h"

#include <iostream>

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T>
thread_local Runtime* NHWCConcat<T>::rt = nullptr;

template <typename T>
Status NHWCConcat<T>::Compute(OpKernelContext* ctx) const {

  NHWCConcat<T>::initRuntime();

  // Number of input tensors to concatenate
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  std::vector<const Tensor*> input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  std::vector<int64_t> output_dims = input_tensors[0]->Shape().GetDims();

  // 'Concat' mode
  if (!is_stack_) {
    // While concating, the rank of the output is the same as the input rank(s)

    // Calculate the size of the concatenated axis
    size_t concat_axis_size = 0;
    for (int64_t index = 0; index < input_count; index++) {
      concat_axis_size += input_tensors[index]->Shape()[static_cast<int>(axis_)];
    }

    output_dims[axis_] = concat_axis_size;
  } else { // 'Stack' mode
    // While stacking, the rank of the output is one more than the input rank(s).
    // Stacking may be thought of as adding an unit dimension (of value 1) in the input tensors,
    // and concatenating them on thie new axis.
    // The value in the corresponding axis of the output will be the number of inputs that are being stacked.
    output_dims.insert(output_dims.begin() + axis_, static_cast<int64_t>(input_count));
  }

  int axis = axis_;
  if(axis_ == 1) // channel
    axis = 3;
  if(axis_ == 2) // height
    axis = 1;
  if(axis_ == 3) // width
    axis = 2;

  if(output_dims.size() > 4 || axis_ > 3)
    return onnxruntime::Concat::Compute(ctx);

  TensorShape output_shape(output_dims);
  Tensor* Y = ctx->Output(0, output_shape);

  armnn::NetworkId* pNetworkId;
  ConcatIterator it = NHWCConcat::rt->layers.find((OpKernel*)this);
  if (it == NHWCConcat::rt->layers.end()) {

    armnn::NetworkId networkId;
    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    const unsigned int supportedNumDims = 4;
    unsigned int numConcatViews = input_count;
    armnn::OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatViews), supportedNumDims);
    concatDescriptor.SetConcatAxis(axis);
    armnn::TensorShape mergeDims(supportedNumDims);
    unsigned int mergeDim = 0;
    for (unsigned int viewIndex = 0; viewIndex < numConcatViews; ++viewIndex) {
      // Copy the input tensor shape to mergeDimSizes and initialize the view origin coordinates for the current input
      mergeDims = ArmNNTensorShape(input_tensors[viewIndex]->Shape(), PREF_DIM);
      mergeDims = { mergeDims[0],
                    mergeDims[2],
                    mergeDims[3],
                    mergeDims[1] };

      unsigned int* viewOrigin = const_cast<unsigned int*>(concatDescriptor.GetViewOrigin(viewIndex));
      std::fill(viewOrigin, viewOrigin + supportedNumDims, 0);

      // Update the view origin coordinates and the merge dimension value
      concatDescriptor.SetViewOriginCoord(viewIndex, axis, mergeDim);
      mergeDim += mergeDims[axis];
    }

    // Update the output shape
    mergeDims[axis] = mergeDim;
    armnn::IConnectableLayer *layer = myNetwork->AddConcatLayer(concatDescriptor, "concat_armnn");

    for (unsigned int viewIndex = 0; viewIndex < numConcatViews; ++viewIndex) {
      armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(viewIndex);
      InputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(viewIndex));

      armnn::TensorShape inputShape = ArmNNTensorShape(input_tensors[viewIndex]->Shape(), PREF_DIM);
      inputShape = { inputShape[0],
                     inputShape[2],
                     inputShape[3],
                     inputShape[1] };
      InputLayer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(inputShape, armnn::DataType::Float32));
    }

    armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);
    layer->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(mergeDims, armnn::DataType::Float32)); 

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, NHWCConcat::rt->run->GetDeviceSpec());

    if (optNet == nullptr) {
      return onnxruntime::Concat::Compute(ctx);
    }

    // Load graph into runtime
    NHWCConcat::rt->run->LoadNetwork(networkId, std::move(optNet));

    std::pair<ConcatIterator, bool> ret;
    ret = NHWCConcat::rt->layers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;
    
  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{};
  for (int index = 0; index < input_count; ++index)
    inputTensors.push_back({index, armnn::ConstTensor(NHWCConcat::rt->run->GetInputTensorInfo(*pNetworkId, index),
                                                       input_tensors[index]->template Data<T>())});
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(NHWCConcat::rt->run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       Y->template MutableData<T>())}};

  NHWCConcat::rt->run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Concat,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NHWCConcat<float>);

}  // namespace contrib
}  // namespace onnxruntime
