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

AllocatorPtr NeutronExecutionProvider::CreateNeutronAllocator(OrtDevice::DeviceId device_id) {
  AllocatorCreationInfo mem_info(
      [](OrtDevice::DeviceId id) {
        return std::make_unique<NeutronAllocator>(id, NEUTRON);
      },
      device_id,
      neutron_flags_ & NEUTRON_FLAG_USE_ARENA
  );
  return CreateAllocator(mem_info);
}

std::vector<AllocatorPtr> NeutronExecutionProvider::CreatePreferredAllocators() {
    AllocatorCreationInfo pinned_mem_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<NeutronPinnedAllocator>(device_id, NEUTRON_PINNED);
      },
      DEFAULT_CPU_ALLOCATOR_DEVICE_ID);
  return std::vector<AllocatorPtr>{
      CreateNeutronAllocator(DEFAULT_NEUTRON_ALLOCATOR_DEVICE_ID),
      CreateAllocator(pinned_mem_info),
  };
}

OrtDevice NeutronExecutionProvider::GetOrtDeviceByMemType(OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeCPUInput) {
    return {};
  }
  if (mem_type == OrtMemTypeCPUOutput) {
    return {OrtDevice::CPU, OrtDevice::MemType::NEUTRON_PINNED, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};
  }
  return default_device_;
}

}  // namespace onnxruntime
