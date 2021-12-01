// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
// Licensed under the MIT License.

#include "core/providers/armnn/tensor/transpose.h"
#include "core/providers/common.h"

#include "core/providers/armnn/armnn_fwd.h"

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T>
thread_local Runtime* Transpose<T>::rt = nullptr;

std::vector<unsigned int> armnnPermutation(std::vector<size_t> permutations, unsigned int rank) {
  // permuteShape assumes Tf/Np permute vectors, we must translate to armnn expected form
  // to do so we find the perm vector which would invert what a tf perm vector would do (ex 3,0,1,2 -> 1,2,3,0)
  std::vector<unsigned int> armnnPermuteShape(rank);
  std::vector<size_t>::iterator it;
  for (unsigned int i = 0; i < rank; ++i) {
      it = std::find(permutations.begin(), permutations.end(), i);
      armnnPermuteShape[i] = static_cast<unsigned int>(std::distance(permutations.begin(), it));
  }
  return armnnPermuteShape;
}

template <typename T>
Status Transpose<T>::Compute(OpKernelContext* ctx) const {

  Transpose<T>::initRuntime();

  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  const Tensor& X = *input_tensor_ptr;
  const TensorShape& input_shape = X.Shape();


  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  unsigned int rank = input_dims.size();

  if(rank > 4) {
      LOGS_DEFAULT(WARNING) << "Arbitrary permutation vectors are supported with rank not greater than 4";
      return onnxruntime::Transpose::Compute(ctx);
  }

  std::vector<int64_t> output_dims(rank);
  const std::vector<size_t>* p_perm;
  std::vector<size_t> default_perm(rank);
  Status status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor& Y = *ctx->Output(0, output_shape);

  const T* x_data = X.template Data<T>();
  T* y_data = Y.template MutableData<T>();

  armnn::NetworkId* pNetworkId;
  TransposeIterator it = Transpose::rt->layers.find((OpKernel*)this);
  if (it == Transpose::rt->layers.end()) {

    armnn::NetworkId networkId;
    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    armnn::TensorShape inputShape = ArmNNTensorShape(X.Shape());
    armnn::TensorShape outputShape = inputShape;
    for (unsigned int i = 0; i < rank; ++i) {
      outputShape[i] = output_dims[i];
    }

    armnn::PermuteDescriptor permuteDescriptor;

    armnn::PermutationVector permutationVector(armnnPermutation(*p_perm,rank).data(), rank);

    permuteDescriptor = armnn::PermuteDescriptor(permutationVector);

    armnn::IConnectableLayer *layer = myNetwork->AddPermuteLayer(permuteDescriptor, "transpose_armnn");

    armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
    armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

    InputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    //Set the tensors in the network.
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Transpose::rt->run->GetDeviceSpec());

    if (optNet == nullptr) {
      LOGS_DEFAULT(WARNING) << "Got invalid operation; defaulting to cpu implementation";
      return onnxruntime::Transpose::Compute(ctx);
    }

    // Load graph into runtime
    Transpose::rt->run->LoadNetwork(networkId, std::move(optNet));

    std::pair<TransposeIterator, bool> ret;
    ret = Transpose::rt->layers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;
    
  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Transpose::rt->run->GetInputTensorInfo(*pNetworkId, 0),
                                                          x_data)}};
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(Transpose::rt->run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       y_data)}};

  Transpose::rt->run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    1, 12,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose<float>);

ONNX_OPERATOR_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    13,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime
