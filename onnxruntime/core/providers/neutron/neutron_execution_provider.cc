// Copyright (c) NXP. All rights reserved.

/* Include `provider_api.h` first to avoid funny inclusion issues */
//#include "core/providers/shared_library/provider_api.h"
/* other headers */
#include "core/providers/neutron/neutron_execution_provider.h"
#include "core/providers/neutron/neutron_allocator.h"
#include "core/providers/neutron/neutron_provider_factory.h"
#include "core/framework/kernel_registry.h"

#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"

using namespace onnxruntime::common;

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace

namespace onnxruntime {

namespace neutron {

std::shared_ptr<NeutronStackAllocator> neutronAlloc(new NeutronStackAllocator());

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 13, 18, uint8_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 13, 18, int8_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 13, 18, int32_t,
                                                      DequantizeLinear);

class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 19, uint8_t, float,
                                                      DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 19, int8_t, float,
                                                      DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 19, int32_t, float,
                                                      DequantizeLinear);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 13, 18, uint8_t,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 13, 18, int8_t,
                                                      QuantizeLinear);

class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 19, uint8_t, float,
                                                      QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 19, int8_t, float,
                                                      QuantizeLinear);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, int8_t,
                                                      QLinearMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t,
                                                      QLinearMatMul);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, uint8_t, MatMulIntegerToFloat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, int8_t, MatMulIntegerToFloat);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t, MatMulInteger);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);

static Status RegisterNeutronKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 13, 18, uint8_t,DequantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 13, 18, int8_t, DequantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 13, 18, int32_t, DequantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 19, uint8_t, float, DequantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 19, int8_t, float, DequantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 19, int32_t, float, DequantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 13, 18, uint8_t, QuantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 13, 18, int8_t, QuantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 19, uint8_t, float, QuantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 19, int8_t, float, QuantizeLinear)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 10, int8_t, QLinearMatMul)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t, QLinearMatMul)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kMSDomain, 1, uint8_t, MatMulIntegerToFloat)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kMSDomain, 1, int8_t, MatMulIntegerToFloat)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t, MatMulInteger)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}
} // namespace neutron

NeutronExecutionProvider::NeutronExecutionProvider(uint32_t neutron_flags)
    : IExecutionProvider{onnxruntime::kNeutronExecutionProvider, true},
      neutron_flags_(neutron_flags) {
   onnxruntime::neutron::neutronAlloc->Init();
}

NeutronExecutionProvider::~NeutronExecutionProvider() {}

/* Utils */

KernelRegistryAndStatus GetNeutronKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = ::onnxruntime::neutron::RegisterNeutronKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry> NeutronExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = GetNeutronKernelRegistry();
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

}  // namespace onnxruntime
