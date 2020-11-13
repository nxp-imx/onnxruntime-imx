// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/acl/tensor/transpose.h"
#include "core/providers/common.h"

#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"

#define PREF_DIM 4

namespace onnxruntime {
namespace acl {

template <typename T>
thread_local std::map<OpKernel*, ACLNEPermute> Transpose<T>::permuteLayers;

std::vector<unsigned int> armnnPermutation(std::vector<size_t> permutations) {
  // permuteShape assumes Tf/Np permute vectors, we must translate to armnn expected form
  // to do so we find the perm vector which would invert what a tf perm vector would do (ex 3,0,1,2 -> 1,2,3,0)
  std::vector<unsigned int> armnnPermuteShape(permutations.size());
  std::vector<size_t>::iterator it;
  for (unsigned int i = 0; i < permutations.size(); ++i) {
      it = std::find(permutations.begin(), permutations.end(), i);
      armnnPermuteShape[i] = static_cast<unsigned int>(std::distance(permutations.begin(), it));
  }
  return armnnPermuteShape;
}

arm_compute::PermutationVector generateACLPermutationVector(std::vector<size_t> perm, int dims) {

  std::vector<unsigned int> permutation = armnnPermutation(perm);
  arm_compute::PermutationVector aclPermutationVector;

  int padd = dims - permutation.size();

  unsigned int start = 0;
  while ((start < permutation.size()) && (start == permutation[start]))
    start++;

  for (unsigned int poz = 0; poz < padd; poz++)
    aclPermutationVector.set(poz, poz);

  for (unsigned int poz = start; poz < permutation.size(); poz++)
    aclPermutationVector.set(padd + poz - start, padd + permutation[poz] - start);

  return aclPermutationVector;
}

template <typename T>
Status Transpose<T>::Compute(OpKernelContext* ctx) const {

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

  ACLNEPermute* pPerm;
  permuteLayersIterator it = permuteLayers.find((OpKernel*)this);
  if (it == permuteLayers.end()) {

    auto mm_layer = ACLCreateMemoryManager();

    ACLNEPermute tperm;
    tperm.mm_layer = std::move(mm_layer);

    tperm.in = std::make_shared<arm_compute::Tensor>();
    tperm.out = std::make_shared<arm_compute::Tensor>();

    tperm.in->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X.Shape(), PREF_DIM), arm_compute::Format::F32));
    tperm.out->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(Y.Shape(), PREF_DIM), arm_compute::Format::F32));

    auto layer = std::make_shared<arm_compute::NEPermute>();

    layer->configure(tperm.in.get(), tperm.out.get(), generateACLPermutationVector(*p_perm, PREF_DIM));

    tperm.layer = std::move(layer);

    std::pair<permuteLayersIterator, bool> ret;
    ret = permuteLayers.insert(std::pair<OpKernel*, ACLNEPermute>((OpKernel*)this, tperm));
    pPerm = &tperm;

  } else {
    pPerm = &it->second;
  }

  const T* x_data = X.template Data<T>();
  ACLImportMemory(pPerm->in->allocator(), (void*)x_data, X.Shape().Size() * 4);

  T* y_data = Y.template MutableData<T>();
  ACLImportMemory(pPerm->out->allocator(), (void*)y_data, Y.Shape().Size() * 4);

  arm_compute::Allocator alloc_mm{};
  pPerm->mm_layer->populate(alloc_mm, 1);
  pPerm->layer->run();
  pPerm->mm_layer->clear();

  pPerm->in->allocator()->free();
  pPerm->out->allocator()->free();

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    1, 12,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose<float>);

ONNX_OPERATOR_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    13,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose<float>);

}  // namespace acl
}  // namespace onnxruntime
