// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/armnn/nhwc/batch_norm.h"
#include "core/providers/armnn/armnn_fwd.h"

#include <thread>
#include <mutex>

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T>
thread_local Runtime* NHWCBatchNorm<T>::rt = nullptr;

template <typename T>
Status NHWCBatchNorm<T>::Compute(OpKernelContext* context) const {

  NHWCBatchNorm<T>::initRuntime();

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* S = context->Input<Tensor>(1);//scale
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* M = context->Input<Tensor>(3);//mean
  const Tensor* V = context->Input<Tensor>(4);//var

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, S, B, M, V));

  const T* x_data = X->template Data<T>();

  Tensor* Y = context->Output(0, X->Shape());

  T* y_data = Y->template MutableData<T>();

  armnn::NetworkId* pNetworkId;
  BatchNormLayersIterator it = NHWCBatchNorm::rt->layers.find((OpKernel*)this);
  if (it == NHWCBatchNorm::rt->layers.end()) {
    
    armnn::NetworkId networkId;
    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    armnn::TensorShape inputShape = ArmNNTensorShape(X->Shape());
    armnn::TensorShape outputShape = ArmNNTensorShape(Y->Shape());

    inputShape = { inputShape[0],
                   inputShape[2],
                   inputShape[3],
                   inputShape[1] };

    outputShape = { outputShape[0],
                    outputShape[2],
                    outputShape[3],
                    outputShape[1] };

    armnn::BatchNormalizationDescriptor desc;
    desc.m_Eps = epsilon_;
    desc.m_DataLayout  = armnn::DataLayout::NHWC;

    const T* mean_data = M->template Data<T>();
    const T* var_data = V->template Data<T>();
    const T* b_data = B->template Data<T>();
    const T* scale_data = S->template Data<T>();

    armnn::TensorInfo meanDesc(ArmNNTensorShape(M->Shape()), armnn::DataType::Float32);
    armnn::ConstTensor mean(meanDesc, mean_data);
    armnn::TensorInfo varianceDesc(ArmNNTensorShape(V->Shape()), armnn::DataType::Float32);
    armnn::ConstTensor variance(varianceDesc, var_data);
    armnn::TensorInfo betaDesc(ArmNNTensorShape(B->Shape()), armnn::DataType::Float32);
    armnn::ConstTensor beta(betaDesc, b_data);
    armnn::TensorInfo gammaDesc(ArmNNTensorShape(S->Shape()), armnn::DataType::Float32);
    armnn::ConstTensor gamma(gammaDesc, scale_data);

    armnn::IConnectableLayer* layer = myNetwork->AddBatchNormalizationLayer(desc, mean, variance, beta, gamma, "batchnorm_armnn");

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
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, NHWCBatchNorm::rt->run->GetDeviceSpec());

    if (optNet == nullptr) {
      ORT_NOT_IMPLEMENTED("Something went wrong when creating the layer");
    }

    // Load graph into runtime
    NHWCBatchNorm::rt->run->LoadNetwork(networkId, std::move(optNet));

    std::pair<BatchNormLayersIterator, bool> ret;
    ret = NHWCBatchNorm::rt->layers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;

  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{{0, armnn::ConstTensor(NHWCBatchNorm::rt->run->GetInputTensorInfo(*pNetworkId, 0),
                                                          x_data)}};
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(NHWCBatchNorm::rt->run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       y_data)}};

  NHWCBatchNorm::rt->run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    BatchNormalization,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder()
      .TypeConstraint("X", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("scale", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("B", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("mean", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("var", DataTypeImpl::GetTensorType<float>()),
    NHWCBatchNorm<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime
