// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2020 NXP
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/math/gemm.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_common.h"

namespace onnxruntime {
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator GEMMLayersIterator;

template <typename T>
class Gemm : public onnxruntime::Gemm<T> {
 public:
  Gemm(const OpKernelInfo& info) : onnxruntime::Gemm<T>(info) {
    int64_t temp;

    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;
    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());

    provider_ = (const_cast<ArmNNExecutionProvider*>(
        static_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  Status Compute(OpKernelContext* context) const override {

    Gemm<T>::initRuntime();

    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);

    bool useBias = B != nullptr && beta_ != 0;
    bool FC = alpha_ == 1 && (beta_ == 1 || beta_ == 0);
    if (!FC) {
      LOGS_DEFAULT(WARNING) << "Implementation not supported ; defaulting to cpu implementation";
      return onnxruntime::Gemm<T>::Compute(context);
    }

    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans, useBias ? B->Shape() : TensorShape({}));

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    auto Y = context->Output(0, TensorShape({M, N}));

    if (trans_A_ == CblasTrans) { // transpose input
      LOGS_DEFAULT(WARNING) << "Transposed input not supported ; defaulting to cpu implementation";
      return onnxruntime::Gemm<T>::Compute(context);
    }

    int64_t K = helper.K();
    LOGS_DEFAULT(VERBOSE) << "Gemm ArmNN:";
    LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
    LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str();
    if (B != nullptr) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str();
    LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str();
    LOGS_DEFAULT(VERBOSE) << "M " << (int)M << ", N " << (int)N << ", K " << (int)K;
    LOGS_DEFAULT(VERBOSE) << "Alfa " << alpha_ << ", Beta " << beta_;
    LOGS_DEFAULT(VERBOSE) << "trans_A_ " << (trans_A_ == CblasTrans);
    LOGS_DEFAULT(VERBOSE) << "trans_B_ " << (trans_B_ == CblasTrans);

    const T* x_data = X->template Data<T>();
    const T* w_data = W->template Data<T>();
    const T* b_data;
    if (useBias)
      b_data = B->template Data<T>();
    T* y_data = Y->template MutableData<T>();

    armnn::NetworkId* pNetworkId;
    GEMMLayersIterator it = Gemm::rt->layers.find((OpKernel*)this);
    if (it == Gemm::rt->layers.end()) {
      
      armnn::NetworkId networkId;
      armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

      armnn::TensorShape inputShape = ArmNNTensorShape(X->Shape());
      armnn::TensorShape weightShape = ArmNNTensorShape(W->Shape());
      armnn::TensorShape outputShape = ArmNNTensorShape(Y->Shape());

      armnn::FullyConnectedDescriptor fcDescriptor;
      fcDescriptor.m_BiasEnabled = useBias;
      fcDescriptor.m_TransposeWeightMatrix = trans_B_ == CblasTrans;

      armnn::IConnectableLayer* fc_armnn;

      armnn::TensorInfo weightsInfo(weightShape, armnn::DataType::Float32);
      armnn::ConstTensor weights(weightsInfo, w_data);

      if (fcDescriptor.m_BiasEnabled) {
        armnn::TensorShape biasShape = ArmNNTensorShape(B->Shape());
        if(B->Shape().NumDimensions() == 2){
          if(B->Shape().GetDims()[0] == 1 && B->Shape().GetDims()[1] > 1) {
            biasShape = {(unsigned int)(B->Shape().GetDims()[1])};
            LOGS_DEFAULT(VERBOSE) << "Bias reshaped to: {" << B->Shape().GetDims()[1] << "}";
          }
        }
        armnn::TensorInfo biasDesc(biasShape, armnn::DataType::Float32);
        armnn::ConstTensor bias(biasDesc, b_data);
        fc_armnn = myNetwork->AddFullyConnectedLayer(fcDescriptor,
                                                     weights,
                                                     armnn::Optional<armnn::ConstTensor>(bias),
                                                     "fc_armnn");
      } else {
        fc_armnn = myNetwork->AddFullyConnectedLayer(fcDescriptor,
                                                     weights,
                                                     armnn::EmptyOptional(),
                                                     "fc_armnn");
      }

      armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
      armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

      InputLayer->GetOutputSlot(0).Connect(fc_armnn->GetInputSlot(0));
      fc_armnn->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

      //Set the tensors in the network.
      armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
      InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

      armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
      fc_armnn->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

      // Optimise ArmNN network
      armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Gemm::rt->run->GetDeviceSpec());

      if (optNet == nullptr) {
        LOGS_DEFAULT(WARNING) << "Got invalid operation; defaulting to cpu implementation";
        return onnxruntime::Gemm<T>::Compute(context);
      }

      // Load graph into runtime
      Gemm::rt->run->LoadNetwork(networkId, std::move(optNet));

      std::pair<GEMMLayersIterator, bool> ret;
      ret = Gemm::rt->layers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
      pNetworkId = &ret.first->second;

    } else {
      pNetworkId = &it->second;
    }

    armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Gemm::rt->run->GetInputTensorInfo(*pNetworkId, 0),
                                                          x_data)}};
    armnn::OutputTensors outputTensors{{0, armnn::Tensor(Gemm::rt->run->GetOutputTensorInfo(*pNetworkId, 0),
                                                         y_data)}};

    Gemm::rt->run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

    LOGS_DEFAULT(VERBOSE) << std::endl;

    return Status::OK();
  }

  ~Gemm() {
    GEMMLayersIterator it = Gemm::rt->layers.find((OpKernel*)this);
    if (it != Gemm::rt->layers.end()) {
      Gemm::rt->run->UnloadNetwork(it->second);
    }
    Gemm::rt->layers.erase(this);
  }

  static void initRuntime(){
    if(!Gemm::rt) {
      static thread_local Runtime runtime_obj;
      armnn::IRuntime::CreationOptions options;
      runtime_obj.run = std::move(armnn::IRuntime::Create(options));

      Gemm::rt =  &runtime_obj;
    }
  }

 private:
  static thread_local Runtime* rt;
  ArmNNExecutionProvider* provider_;

  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  float alpha_;
  float beta_;
};

template <typename T>
thread_local Runtime* Gemm<T>::rt = nullptr;

}  // namespace armnn_ep
}  // namespace onnxruntime
