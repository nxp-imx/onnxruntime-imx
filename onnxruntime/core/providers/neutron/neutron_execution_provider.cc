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
#include "core/framework/compute_capability.h"

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

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, float, MatMulNBits);

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
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
                          kNeutronExecutionProvider, kMSDomain, 1, float, MatMulNBits)>,
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
    : IExecutionProvider(onnxruntime::kNeutronExecutionProvider),
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

std::vector<std::unique_ptr<ComputeCapability>>
NeutronExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                        const IKernelLookup&,
	                                const GraphOptimizerRegistry&,
                                        IResourceAccountant*)const {
  InlinedVector<NodeIndex> candidates;

  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      continue;
    }

    if ("DequantizeLinear" == node.OpType() ||
        "QuantizeLinear" == node.OpType() ||
        "QLinearMatMul" == node.OpType() ||
        "MatMulIntegerToFloat" == node.OpType() ||
        "MatMulInteger" == node.OpType()) {
      candidates.push_back(node.Index());
    } else if ("MatMulNBits" == node.OpType()) {
      const auto& attributes = node.GetAttributes();
      const auto& input_defs = node.InputDefs();
      int64_t K = SafeInt<int64_t>(attributes.at("K").i());
      int64_t N = SafeInt<int64_t>(attributes.at("N").i());

      if (K % 16 != 0 || N % 128 != 0) {
        LOGS_DEFAULT(INFO) << "NeutronEP: MatMulNBits (" << node.Name() << ") not supported, invalid K or N.";
      } else if (input_defs.size() > 3 && input_defs[3]->Exists()) {
        LOGS_DEFAULT(INFO) << "NeutronEP: MatMulNBits (" << node.Name() << ") with zero-point not supported.";
      } else {
        candidates.push_back(node.Index());
      }
    }
  }

  // For ROCM EP, exclude the subgraph that is preferred to be placed in CPU
  // These are usually shape related computation subgraphs
  // Following logic can be extended for other EPs
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

std::shared_ptr<KernelRegistry> NeutronExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = GetNeutronKernelRegistry();
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

}  // namespace onnxruntime
