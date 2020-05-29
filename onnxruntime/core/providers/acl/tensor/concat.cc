// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/acl/tensor/concat.h"
#include "core/providers/common.h"
#include "core/framework/TensorSeq.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include <iostream>
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

template <typename T>
void importDataFromTensor(arm_compute::Tensor* tensor, T* data){

  arm_compute::Window aclInpuWindow;
  aclInpuWindow.use_tensor_dimensions(tensor->info()->tensor_shape());

  arm_compute::Iterator aclInputIt(tensor, aclInpuWindow);
  const unsigned int aclWidth = tensor->info()->dimension(0);
  const unsigned int aclHeight = tensor->info()->dimension(1);

  // copy input tensor into the larger buffer
  arm_compute::execute_window_loop(
      aclInpuWindow,
      [&](const arm_compute::Coordinates& co) {
        data[co.z() * (aclWidth * aclHeight) + co.y() * aclWidth + co.x()] = *reinterpret_cast<float*>(aclInputIt.ptr());
      },
      aclInputIt);
}

namespace onnxruntime {
namespace acl {

template <typename T>
Status Concat<T>::Compute(OpKernelContext* ctx) const {
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

  TensorShape output_shape(output_dims);
  Tensor* Y = ctx->Output(0, output_shape);

  arm_compute::Tensor output;
  std::vector<arm_compute::ITensor*> inputs_vector;
  for(int i = 0; i < input_count; i++) {
    arm_compute::Tensor* input = new arm_compute::Tensor();
    auto X = input_tensors[i];
    input->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape()), arm_compute::Format::F32));
    
    inputs_vector.push_back(input);
  }

  arm_compute::NEConcatenateLayer layer;
  layer.configure(inputs_vector, &output, 3 - axis_);

  for(int i = 0; i < input_count; i++) {
    auto X = input_tensors[i];
    const T* x_data = X->template Data<T>();
    arm_compute::Tensor* in = static_cast<arm_compute::Tensor*>(inputs_vector[i]);

    if(X->Shape().Size() != 0 && in->info()->has_padding() ){
      in->allocator()->allocate();
      arm_compute::Window aclInpuWindow;
      aclInpuWindow.use_tensor_dimensions(in->info()->tensor_shape());

      arm_compute::Iterator aclInputIt(in, aclInpuWindow);
      const unsigned int aclWidth = in->info()->dimension(0);
      const unsigned int aclHeight = in->info()->dimension(1);

      // copy input tensor into the larger buffer
      arm_compute::execute_window_loop(
        aclInpuWindow,
        [&](const arm_compute::Coordinates& co) {
          *reinterpret_cast<float*>(aclInputIt.ptr()) = x_data[co.z() * (aclWidth * aclHeight) + co.y() * aclHeight + co.x()];
        },
        aclInputIt);
    }else{
      ACLImportMemory(in->allocator(), (void*)x_data, X->Shape().Size() * 4);
    }
  }

  T* y_data = Y->template MutableData<T>();

  if(Y->Shape().Size() != 0 && output.info()->has_padding() ){
    output.allocator()->allocate();
  } else {
    ACLImportMemory(output.allocator(), (void*)y_data, Y->Shape().Size() * 4);
  }

  layer.run();

  if(Y->Shape().Size() != 0 && output.info()->has_padding() ){
    importDataFromTensor<T>(&output, y_data);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat,
    kOnnxDomain,
    4, 10,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat<float>);

}  // namespace acl
}  // namespace onnxruntime
