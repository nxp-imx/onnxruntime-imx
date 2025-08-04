// Copyright 2025 NXP

#include "core/providers/neutron/ops/quantize_linear.h"
#include "core/framework/element_type_lists.h"
#include "core/util/qmath.h"
#include "core/providers/neutron/ops/common.h"
#include "core/providers/neutron/neutron_fwd.h"


namespace onnxruntime {
namespace neutron {


#define REGISTER_Q_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                 \
      QuantizeLinear,                                                      \
      kOnnxDomain,                                                         \
      13, 18,                                                              \
      T,                                                                   \
      kNeutronExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())      \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),         \
      QuantizeLinear<T>);


#define REGISTER_Q_KERNEL_TYPED_19(T)                                      \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                       \
      QuantizeLinear,                                                      \
      kOnnxDomain,                                                         \
      19,                                                                  \
      T, float,                                                            \
      kNeutronExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())      \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),         \
      QuantizeLinear<T>);


REGISTER_Q_KERNEL_TYPED(uint8_t)
REGISTER_Q_KERNEL_TYPED(int8_t)
REGISTER_Q_KERNEL_TYPED_19(int8_t)
REGISTER_Q_KERNEL_TYPED_19(uint8_t)

/*
    From CPU Provider implementation
*/

template <typename InputType, typename OutputType>
void ParQuantizeLinear(const InputType* Input,
                       OutputType* Output,
                       size_t N,
                       InputType Scale,
                       size_t bd,
                       const OutputType* ZeroPoint,
                       bool saturate,
                       concurrency::ThreadPool* thread_pool) {
  if constexpr (!boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputType>::value) {
    ORT_UNUSED_PARAMETER(saturate);
    ParQuantizeLinearStd(Input, Output, N, Scale,
                            ZeroPoint != nullptr ?
                                ZeroPoint[bd] :
                                (OutputType)0,
                            thread_pool);
  } else {
    ParQuantizeLinearSat(Input, Output, N, Scale,
                            ZeroPoint != nullptr ?
                                ZeroPoint[bd] :
                                OutputType(static_cast<InputType>(static_cast<float>(0)), true),
                            saturate, thread_pool);
  }
}

template <typename T, typename InT>
void ComputeLoop(OpKernelContext* ctx,
                 const InT* input, const InT* scale, const T* zero_point, T* output,
                 int64_t N, int64_t broadcast_dim, int64_t block_size, bool saturate) {
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      ParQuantizeLinear(input, output,
                        static_cast<size_t>(block_size),
                        scale[bd], bd, zero_point, saturate,
                        ctx->GetOperatorThreadPool());
      input += block_size;
      output += block_size;
    }
  }
}

// formula is Y = X / Scale + ZeroPoint
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, N, broadcast_dim, block_size);

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;

  /* Override output tensor's buffer */
  auto y_type = y.DataType();
  auto vec_x_shape = x_shape.AsShapeVector();
  auto out_size = std::accumulate(vec_x_shape.begin(), vec_x_shape.end(), sizeof(T), std::multiplies<uint32_t>());

#if NEUTRON_AARCH64
  auto allocator = Info().GetAllocator(OrtMemType::OrtMemTypeDefault);
#else
  auto allocator = Info().GetAllocator(OrtMemType::OrtMemTypeCPU);
#endif
  auto buffer = allocator->Alloc(out_size);

  auto out_tensor = Tensor(y_type, x_shape,
                    /* alloc*/  buffer,
                    /* deleter*/ allocator,
                    /* offset*/ 0);

  T* out_ptr = out_tensor.MutableData<T>();
#ifndef NDEBUG
  printf("[QuantizeLinear] Output ptr: %p \n", out_ptr);
#endif
  if (x.IsDataType<float>()) {
    ComputeLoop<T, float>(ctx, x.Data<float>(), y_scale.Data<float>(), zero_point,
                          out_ptr, N, broadcast_dim, block_size, saturate_);
  } else {
    ORT_THROW("Unsupported input type.");
  }

  OrtValue out_ort;
  Tensor::InitOrtValue(std::move(out_tensor), out_ort);
  ORT_RETURN_IF_ERROR(ctx->ForceMLValue(0, std::move(out_ort)));

  return Status::OK();
}


}  // namespace neutron
}  // namespace onnxruntime
