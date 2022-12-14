// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2019-2021 NXP
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include <thread>
#include <mutex>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/acl/nn/conv.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"

// ACL
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#if defined(ACL_2008)
#include "arm_compute/core/AccessWindowStatic.h"
#endif
// NEON
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"

#ifdef ACL_1902
#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolutionLayer3x3Kernel.h"
#endif
#if defined(ACL_1905) || defined(ACL_1908)
#include "arm_compute/runtime/NEON/functions/assembly/NEDepthwiseConvolutionAssemblyDispatch.h"
#endif

#define CONV_ACL
#undef DEPTHWISE_CPU

#define PREF_DIM 4

namespace onnxruntime {
namespace acl {

template <typename T>
thread_local std::map<OpKernel*, ACLNEConv> Conv<T>::convLayers;

template <typename T>
arm_compute::TensorShape Conv<T>::ACLReshapeWeightsDepthwise(arm_compute::Tensor* kernel) const {
  arm_compute::TensorShape shape = arm_compute::TensorShape(kernel->info()->tensor_shape());
  shape[2] = shape[2] * shape[3];
  shape[3] = 1;

  return shape;
}

#if defined(ACL_2008)

bool validate_window(arm_compute::ITensorInfo *input, arm_compute::ITensorInfo *weights, arm_compute::ITensorInfo *output, const arm_compute::PadStrideInfo &conv_info, unsigned int depth_multiplier, const arm_compute::Size2D &dilation) {
    arm_compute::Window win;

    // Configure kernel window (generic)
    const unsigned int conv_stride_x = conv_info.stride().first;
    const unsigned int conv_stride_y = conv_info.stride().second;
    const unsigned int conv_pad_top  = conv_info.pad_top();
    const unsigned int conv_pad_left = conv_info.pad_left();

    unsigned int num_elems_written_per_iteration = 16 >> conv_stride_x;
    unsigned int num_elems_read_per_iteration = 12 + 11 * (dilation.x() - 1);

    win = arm_compute::calculate_max_window(*output, arm_compute::Steps(num_elems_written_per_iteration));

    arm_compute::AccessWindowRectangle  input_access(input, -conv_pad_left, -conv_pad_top, num_elems_read_per_iteration, 3 + 2 * (dilation.y() - 1), conv_stride_x, conv_stride_y);
    arm_compute::AccessWindowStatic     weights_access(weights, 0, 0, 3, 3);
    arm_compute::AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);

    bool window_changed = arm_compute::update_window_and_padding(win, input_access, weights_access, output_access);

    return window_changed;
}

bool is_optimized_supported(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *weights, const arm_compute::ITensorInfo *biases, const arm_compute::ITensorInfo *output, const arm_compute::PadStrideInfo &conv_info, unsigned int depth_multiplier, const arm_compute::Size2D &dilation) {

  if(dilation.x() < 1 || dilation.y() < 1)
    return false;

  const size_t idx_w = arm_compute::get_data_layout_dimension_index(input->data_layout(), arm_compute::DataLayoutDimension::WIDTH);
  const size_t idx_h = arm_compute::get_data_layout_dimension_index(input->data_layout(), arm_compute::DataLayoutDimension::HEIGHT);
  if( (weights->dimension(idx_w) + (weights->dimension(idx_w) - 1) * (dilation.x() - 1) > input->dimension(idx_w) + conv_info.pad_left() + conv_info.pad_right()) ||
      (weights->dimension(idx_h) + (weights->dimension(idx_h) - 1) * (dilation.y() - 1) > input->dimension(idx_h) + conv_info.pad_top() + conv_info.pad_bottom()) )
      return false;

  if(biases != nullptr) {
    const unsigned int channel_idx = arm_compute::get_data_layout_dimension_index(input->data_layout(), arm_compute::DataLayoutDimension::CHANNEL);
    if(biases->num_dimensions() > 1 || biases->dimension(0) != weights->dimension(channel_idx))
      return false;
  }

  // Reshape input shape if in NHWC format
  const arm_compute::DataLayout data_layout = input->data_layout();
  arm_compute::TensorShape in_shape{ input->tensor_shape() };
  if(data_layout == arm_compute::DataLayout::NHWC) {
    in_shape.set(arm_compute::Window::DimX, input->tensor_shape().y());
    in_shape.set(arm_compute::Window::DimY, input->tensor_shape().z());
    in_shape.set(arm_compute::Window::DimZ, input->tensor_shape().x());
  }

  // Check weighs size
  std::set<unsigned int> supported_kernel_sizes = { 3, 5 };
  const unsigned int     width_idx              = arm_compute::get_data_layout_dimension_index(data_layout, arm_compute::DataLayoutDimension::WIDTH);
  const unsigned int     height_idx             = arm_compute::get_data_layout_dimension_index(data_layout, arm_compute::DataLayoutDimension::HEIGHT);
  const unsigned int     kernel_w               = weights->dimension(width_idx);
  const unsigned int     kernel_h               = weights->dimension(height_idx);
  bool                   weights_supported      = (kernel_w == kernel_h) && (supported_kernel_sizes.count(kernel_w) != 0);

  // Check for supported strides
  const auto &strides           = conv_info.stride();
  bool        supported_strides = (strides.first == strides.second) && ((strides.first == 1) || (strides.first == 2));

  // Check for supported padding
  const auto    pad_top           = conv_info.pad_top();
  const auto    pad_right         = conv_info.pad_right();
  const auto    pad_bottom        = conv_info.pad_bottom();
  const auto    pad_left          = conv_info.pad_left();
  arm_compute::PadStrideInfo same_pad = arm_compute::calculate_same_pad(in_shape, weights->tensor_shape(), conv_info, arm_compute::DataLayout::NCHW, dilation);
  bool          is_same_padding   = (pad_top == same_pad.pad_top()) && (pad_right == same_pad.pad_right()) && (pad_bottom == same_pad.pad_bottom()) && (pad_left == same_pad.pad_left());
  bool          is_valid_padding  = (pad_top == 0) && (pad_right == 0) && (pad_bottom == 0) && (pad_left == 0);
  bool          supported_padding = is_same_padding || is_valid_padding;

  bool is_dilation_supported = ((dilation == arm_compute::Size2D(1U, 1U)) || ((dilation.x() == dilation.y()) && strides.first == 1));

  bool is_supported = weights_supported && supported_strides && supported_padding && is_dilation_supported;
  if(!is_supported) {
    if( (weights->dimension(width_idx) != 3 || weights->dimension(height_idx) != 3) ||
        (conv_info.stride().first < 1 || conv_info.stride().first > 3) )
        return false;

    bool window_changed = validate_window(input->clone().get(), weights->clone().get(), output->clone().get(), conv_info, depth_multiplier, dilation);

    if(window_changed)
      return false;
  }

  if(output->total_size() != 0) {
    const arm_compute::TensorShape output_shape = arm_compute::misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
    for(unsigned int i = 0; i < output_shape.num_dimensions(); i++)
      if(output->tensor_shape()[i] != output_shape[i])
        return false;
  }

  return true;
}
#endif

#ifdef CONV_ACL
template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();

  ACLNEConv* pConv;
  ConvLayersIterator it = Conv::convLayers.find((OpKernel*)this);
  if (it != Conv::convLayers.end()) {
    pConv = &it->second;
    if (pConv->isDepthwiseCPU == true) {
      Status s = onnxruntime::Conv<T>::Compute(context);
      return s;
    }
  }

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];

  LOGS_DEFAULT(VERBOSE) << "Conv ACL:";  
  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str();
  if (B != nullptr) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str();

  if (X->Shape().NumDimensions() != PREF_DIM) {
    LOGS_DEFAULT(WARNING) << "ACL does not have support for tensors with 4 or more dimensions; defaulting to cpu implementation";
    Status s = onnxruntime::Conv<T>::Compute(context);
    return s;
  }

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str();

  arm_compute::ActivationLayerInfo::ActivationFunction acl_activ_func;
  bool acl_activ_enabled = false;

  if (activation_type == "Relu") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-Relu fused implementation";
  } else if (activation_type == "LeakyRelu") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-LeakyRelu fused implementation";
  } else if (activation_type == "Tanh") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::TANH;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-Tanh fused implementation";
  } else if (activation_type == "Sigmoid") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-Sigmoid fused implementation";
  } else if (!activation_type.empty()) {
    ORT_NOT_IMPLEMENTED("Not implemented fused activation: ", activation_type);
  }

  if (it == Conv::convLayers.end()) {

    auto mm_layer = ACLCreateMemoryManager();

    ACLNEConv tconv = {0};
    tconv.mm_layer = std::move(mm_layer);

    tconv.in = std::make_shared<arm_compute::Tensor>();
    tconv.k = std::make_shared<arm_compute::Tensor>();
    if (B != nullptr)
      tconv.b = std::make_shared<arm_compute::Tensor>();
    tconv.out = std::make_shared<arm_compute::Tensor>();

    tconv.in->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape(), PREF_DIM), arm_compute::Format::F32));
    tconv.k->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape()), arm_compute::Format::F32));
    if (B != nullptr) {
      tconv.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(B->Shape()), arm_compute::Format::F32));
    }
    tconv.out->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(Y->Shape(), PREF_DIM), arm_compute::Format::F32));

    const arm_compute::DataLayout data_layout = tconv.in->info()->data_layout();
    const int idx_channel = arm_compute::get_data_layout_dimension_index(data_layout, arm_compute::DataLayoutDimension::CHANNEL);
    bool isDepthwise = (conv_attrs_.group > 1 && conv_attrs_.group == tconv.in->info()->tensor_shape()[idx_channel]);
    tconv.isDepthwiseCPU = isDepthwise;

    std::vector<int64_t> aclStrides(2);
    aclStrides[0] = (strides.size() == 2) ? strides[1] : 1;
    aclStrides[1] = strides[0];

    std::vector<int64_t> aclPads(4);
    // The pad order in acl is: pad_left, pad_right, pad_top, pad_bottom
    if (pads.size() == 2) {
      if (strides.size() == 1) {
        aclPads[0] = 0;
        aclPads[1] = 0;
        aclPads[2] = pads[1];
        aclPads[3] = pads[0];
      } else {
        aclPads[0] = pads[1];
        aclPads[1] = pads[0];
        aclPads[2] = pads[1];
        aclPads[3] = pads[0];
      }
    } else {
      aclPads[0] = pads[1];
      aclPads[1] = pads[3];
      aclPads[2] = pads[0];
      aclPads[3] = pads[2];
    }

    arm_compute::PadStrideInfo aclPadStride = arm_compute::PadStrideInfo(aclStrides[0], aclStrides[1],
                                                                         aclPads[0], aclPads[1], aclPads[2], aclPads[3], arm_compute::DimensionRoundingType::FLOOR);
    unsigned int aclDilation0 = (dilations.size() == 2) ? dilations[1] : 1;

    LOGS_DEFAULT(VERBOSE) << "padding: {" << aclPads[0] << "," << aclPads[1] << "," << aclPads[2] << "," << aclPads[3] << "}";
    LOGS_DEFAULT(VERBOSE) << "strides: {" << aclStrides[0] << "," << aclStrides[1] << "}";


    if (isDepthwise) {
      LOGS_DEFAULT(VERBOSE) << "Depthwise convolution";
#ifdef DEPTHWISE_CPU
      Status s = onnxruntime::Conv<T>::Compute(context);
      std::pair<ConvLayersIterator, bool> ret;
      ret = Conv::convLayers.insert(std::pair<OpKernel*, ACLNEConv>((OpKernel*)this, tconv));
      return s;
#else
      tconv.k->info()->set_tensor_shape(ACLReshapeWeightsDepthwise(tconv.k.get()));

      // in the configure function for NEDepthwiseConvolutionLayer3x3, there is a separation based on the optimization
#ifdef ACL_1902
      bool optimizable =
          arm_compute::NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(tconv.in->info()->tensor_shape(),
                                                                                             aclPadStride,
                                                                                             tconv.in->info()->data_type(),
                                                                                             1 /* depth multiplier */,
                                                                                             tconv.in->info()->data_layout());
#elif defined(ACL_1905) || defined(ACL_1908)
      bool optimizable =
          arm_compute::NEDepthwiseConvolutionAssemblyDispatch::is_optimized_supported(tconv.in->info(),
                                                                                      tconv.k->info(),
                                                                                      aclPadStride,
                                                                                      1 /* depth multiplier */,
                                                                                      arm_compute::Size2D(aclDilation0, dilations[0]));
#elif defined(ACL_2002)
      bool optimizable = bool(arm_compute::NEDepthwiseConvolutionLayerOptimized::validate(tconv.in->info(),
                                                                           tconv.k->info(),
                                                                           (B != nullptr) ? tconv.b->info() : nullptr,
                                                                           tconv.out->info(),
                                                                           aclPadStride,
                                                                           1 /* depth multiplier */,
                                                                           acl_activ_enabled ?
                                                                              arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) :
                                                                              arm_compute::ActivationLayerInfo(),
                                                                           arm_compute::Size2D(aclDilation0, dilations[0])));
#elif defined(ACL_2008)
      bool optimizable = is_optimized_supported(tconv.in->info(),
                                                tconv.k->info(),
                                                (B != nullptr) ? tconv.b->info() : nullptr,
                                                tconv.out->info(),
                                                aclPadStride,
                                                1 /* depth multiplier */,
                                                arm_compute::Size2D(aclDilation0, dilations[0]));
#elif defined(ACL_2102) || defined(ACL_2108)
      bool optimizable = bool(arm_compute::NEDepthwiseConvolutionLayer::validate(tconv.in->info(),
                                                                           tconv.k->info(),
                                                                           (B != nullptr) ? tconv.b->info() : nullptr,
                                                                           tconv.out->info(),
                                                                           aclPadStride,
                                                                           1 /* depth multiplier */,
                                                                           acl_activ_enabled ?
                                                                           arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) :
                                                                           arm_compute::ActivationLayerInfo(),
                                                                           arm_compute::Size2D(aclDilation0, dilations[0])));
#endif

      if (optimizable) {
        LOGS_DEFAULT(VERBOSE) << "ACL optimized depthwise convolution";
#if defined(ACL_1902) || defined(ACL_1905)
        auto layer = std::make_shared<arm_compute::NEDepthwiseConvolutionLayer3x3>();
#elif defined(ACL_1908)
        auto layer = std::make_shared<arm_compute::NEDepthwiseConvolutionLayerOptimized>();
#elif defined(ACL_2002) || defined(ACL_2008) || defined(ACL_2102) || defined(ACL_2108)
        auto layer = std::make_shared<arm_compute::NEDepthwiseConvolutionLayer>();
#endif

#ifdef ACL_1902
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                         aclPadStride, 1 /* depth multiplier */,
                         acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo());
#elif defined(ACL_1905) || defined(ACL_1908) || defined(ACL_2002) || defined(ACL_2008) || defined(ACL_2102) || defined(ACL_2108)
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                         aclPadStride, 1 /* depth multiplier */,
                         acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                         arm_compute::Size2D(aclDilation0, dilations[0]));
#endif
        tconv.layer = std::move(layer);
        tconv.isDepthwiseCPU = false;
      } else {
        LOGS_DEFAULT(VERBOSE) << "CPU depthwise convolution";
        Status s = onnxruntime::Conv<T>::Compute(context);
        std::pair<ConvLayersIterator, bool> ret;
        ret = Conv::convLayers.insert(std::pair<OpKernel*, ACLNEConv>((OpKernel*)this, tconv));
        return s;
      }
#endif  //DEPTHWISE_CPU
    } else {
      if(tconv.k->info()->tensor_shape()[0] == 1 && tconv.k->info()->tensor_shape()[1] == 1) {
        LOGS_DEFAULT(VERBOSE) << "CPU pointwise convolution";
        Status s = onnxruntime::Conv<T>::Compute(context);
        return s;
      } else {
        if(tconv.k->info()->tensor_shape()[0] == 9 && tconv.k->info()->tensor_shape()[1] == 9) {
          LOGS_DEFAULT(WARNING) << "9x9 DirectConvolution does not have an implementation in NCHW layout; defaulting to cpu implementation";
          Status s = onnxruntime::Conv<T>::Compute(context);
          return s;
        }
        LOGS_DEFAULT(VERBOSE) << "ACL 2D convolution";
        auto layer = std::make_shared<arm_compute::NEConvolutionLayer>(mm_layer);
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                         aclPadStride,
                         arm_compute::WeightsInfo(), arm_compute::Size2D(aclDilation0, dilations[0]),
                         acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                         false, conv_attrs_.group);
        tconv.layer = std::move(layer);
      }
    }

    tconv.out->info()->set_format(tconv.in->info()->format());

    std::pair<ConvLayersIterator, bool> ret;
    ret = Conv::convLayers.insert(std::pair<OpKernel*, ACLNEConv>((OpKernel*)this, tconv));
    pConv = &ret.first->second;

    ACLPrintTensorShape("X", *tconv.in.get());
    ACLPrintTensorShape("Y", *tconv.out.get());

  } else {
    //TODO: valildate shapes
    pConv = &it->second;
  }

  const T* x_data = X->template Data<T>();
  if(X->Shape().Size() != 0 && pConv->in->info()->has_padding() ){
    pConv->in->allocator()->allocate();
    importDataToTensor<T>(pConv->in.get(), x_data);
  }else{
    ACLImportMemory(pConv->in->allocator(), (void*)x_data, X->Shape().Size() * 4);
  }

  const T* k_data = W->template Data<T>();
  ACLImportMemory(pConv->k->allocator(), (void*)k_data, W->Shape().Size() * 4);

  if (B != nullptr) {
    const T* b_data = B->template Data<T>();
    ACLImportMemory(pConv->b->allocator(), (void*)b_data, B->Shape().Size() * 4);
  }

  T* y_data = Y->template MutableData<T>();
  if(Y->Shape().Size() != 0 && pConv->out->info()->has_padding() ){
    pConv->out->allocator()->allocate();
  } else {
    ACLImportMemory(pConv->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);
  }

  arm_compute::Allocator alloc_mm{};
  pConv->mm_layer->populate(alloc_mm, 1);
  pConv->layer->run();
  pConv->mm_layer->clear();

  if(Y->Shape().Size() != 0 && pConv->out->info()->has_padding() ){
    importDataFromTensor<T>(pConv->out.get(), y_data);
  }

  pConv->in->allocator()->free();
  pConv->k->allocator()->free();
  if (B != nullptr)
    pConv->b->allocator()->free();
  pConv->out->allocator()->free();

  LOGS_DEFAULT(VERBOSE) << std::endl;

  return Status::OK();
}
#else
template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str();
  if (B != nullptr)
    LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str();

  LOGS_DEFAULT(VERBOSE) << std::endl;

  Status s = onnxruntime::Conv<T>::Compute(context);
  return s;
}
#endif

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace acl
}  // namespace onnxruntime
