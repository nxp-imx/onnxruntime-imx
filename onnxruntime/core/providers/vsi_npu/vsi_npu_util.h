/****************************************************************************
 *
 *    Copyright (c) 2020 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#pragma once
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "nnrt/types.hpp"

namespace onnxruntime {
namespace vsi_npu {

nnrt::OperandType convertToOperandType(const int32_t dtype);

nnrt::OperandType convertToOperandType(const ONNX_NAMESPACE::DataType type);

std::string PrintNode(const onnxruntime::NodeArg& node_arg);

std::string PrintNode(const std::vector<int64_t> shape);

size_t GetTensorElementSize(const ONNXTensorElementDataType type);

size_t GetTensorBytes(Ort::CustomOpApi& ort, OrtTensorTypeAndShapeInfo* info);

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg);

void SetTensorDims(const onnxruntime::NodeArg& node_arg, std::vector<uint32_t>& dims);

std::shared_ptr<uint8_t> UnpackTensor(const NodeArg* node,
                                      const ONNX_NAMESPACE::TensorProto& initializer);

template <typename T>
Status GetAttr(const OpNodeProtoHelper<ProtoHelperNodeContext>& attrs,
               const std::string& name,
               T* value) {
    return attrs.GetAttr<T>(name, value);
};

template <>
Status GetAttr(const OpNodeProtoHelper<ProtoHelperNodeContext>& attrs,
               const std::string& name,
               int32_t* value);

template <typename T>
Status GetAttrs(const OpNodeProtoHelper<ProtoHelperNodeContext>& attrs,
                const std::string& name,
                std::vector<T>& values,
                bool reverse) {
    if (reverse) {
        std::vector<T> v;
        Status status = attrs.GetAttrs<T>(name, v);
        values.assign(v.rbegin(), v.rend());
        return status;
    } else {
        return attrs.GetAttrs<T>(name, values);
    }
};

template <>
Status GetAttrs(const OpNodeProtoHelper<ProtoHelperNodeContext>& attrs,
                const std::string& name,
                std::vector<int32_t>& values,
                bool reverse);

template <typename T>
Status GetAttrs(const OrtApi* api,
                OrtKernelContext* context,
                const int32_t index,
                std::vector<T>& values) {
    Ort::CustomOpApi ort{*api};
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, index);
    const auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    const auto tensor_element_count = ort.GetTensorShapeElementCount(tensor_info);
    const auto tensor_value = (T*)ort.GetTensorData<void>(input_tensor);

    for (size_t i = 0; i < tensor_element_count; i++) {
        values.push_back(tensor_value[i]);
    }
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
    return Status::OK();
}

template <typename T>
Status GetAttrs(const OrtApi* api,
                OrtKernelContext* context,
                const int32_t index,
                std::vector<T>& values,
                bool reverse) {
    if (reverse) {
        std::vector<T> v;
        Status status = GetAttrs<T>(api, context, index, v);
        values.assign(v.rbegin(), v.rend());
        return status;
    } else {
        return GetAttrs<T>(api, context, index, values);
    }
};

template <>
Status GetAttrs(const OrtApi* api,
                OrtKernelContext* context,
                const int32_t index,
                std::vector<int32_t>& values,
                bool reverse);

nnrt::PadType GetPadType(const std::string type);

nnrt::PadMode GetPadMode(const std::string mode);

bool CheckMainInputType(const Node* node,
                        std::string& reason);

bool CheckZeroDim(const NodeArg* node_arg);

bool ExcludeType(const NodeArg* node_arg, std::string& reason);

bool CheckAllExcludeType(const Node* node,
                         std::string& reason);

bool CheckAllZeroDim(const Node* node, std::string& reason);

}  // namespace vsi_npu
}  // namespace onnxruntime
