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

#include "vsi_npu_util.h"

namespace onnxruntime {

template <typename T>
struct shared_array_deletor {
    void operator()(T const* ptr) { delete[] ptr; }
};
namespace vsi_npu {

nnrt::OperandType convertToOperandType(const int32_t dtype) {
    nnrt::OperandType type = nnrt::OperandType::NONE;
    switch (dtype) {
        case onnx::TensorProto_DataType_FLOAT:
            type = nnrt::OperandType::TENSOR_FLOAT32;
            break;
        case onnx::TensorProto_DataType_FLOAT16:
            type = nnrt::OperandType::TENSOR_FLOAT16;
            break;
        case onnx::TensorProto_DataType_INT8:
            type = nnrt::OperandType::TENSOR_QUANT8_SYMM;
            break;
        case onnx::TensorProto_DataType_UINT8:
            type = nnrt::OperandType::TENSOR_QUANT8_ASYMM;
            break;
        case onnx::TensorProto_DataType_INT32:
            type = nnrt::OperandType::TENSOR_INT32;
            break;
        case onnx::TensorProto_DataType_INT16:
            type = nnrt::OperandType::TENSOR_INT16;
            break;
        case onnx::TensorProto_DataType_BOOL:
            type = nnrt::OperandType::TENSOR_BOOL8;
            break;
        default:
            LOGS_DEFAULT(WARNING) << "Unsupported Operand type: " << dtype;
            break;
    }
    return type;
}

nnrt::OperandType convertToOperandType(const ONNX_NAMESPACE::DataType type) {
    static const std::map<std::string, nnrt::OperandType> type_table = {
        {"tensor(float)", nnrt::OperandType::TENSOR_FLOAT32},
        {"tensor(float16)", nnrt::OperandType::TENSOR_FLOAT16},
        {"tensor(int8)", nnrt::OperandType::TENSOR_QUANT8_SYMM},
        {"tensor(uint8)", nnrt::OperandType::TENSOR_QUANT8_ASYMM},
        {"tensor(int32)", nnrt::OperandType::TENSOR_INT32},
        {"tensor(int16)", nnrt::OperandType::TENSOR_INT16},
        {"tensor(bool)", nnrt::OperandType::TENSOR_BOOL8},
    };
    auto search = type_table.find(*type);
    if (search != type_table.end()) {
        return search->second;
    }
    LOGS_DEFAULT(WARNING) << "Unsupported Operand type: " << *type;
    return nnrt::OperandType::NONE;
}

std::string PrintNode(const onnxruntime::NodeArg& node_arg) {
    auto shape = node_arg.Shape();
    if (shape == nullptr || shape->dim_size() == 0) {
        return "<null>";
    }
    std::string s = node_arg.Name() + ":<";
    for (int i = 0; i < shape->dim_size(); i++) {
        auto dim = shape->dim(i);
        std::string s1;
        std::stringstream ss;
        ss << dim.dim_value();
        ss >> s1;
        s += s1;
        if (i < shape->dim_size() - 1) {
            s += ",";
        } else {
            s += ">";
        }
    }
    return s;
}

std::string PrintNode(const std::vector<int64_t> shape) {
    if (shape.size() == 0) {
        return "<null>";
    }
    std::string s = "<";
    for (std::size_t i = 0; i < shape.size(); i++) {
        auto dim = shape[i];
        std::string s1;
        std::stringstream ss;
        ss << dim;
        ss >> s1;
        s += s1;
        if (i < shape.size() - 1) {
            s += ",";
        } else {
            s += ">";
        }
    }
    return s;
}

size_t GetTensorElementSize(const ONNXTensorElementDataType type) {
    switch (type) {
        case onnx::TensorProto_DataType_INT64:
            return 8;
        case onnx::TensorProto_DataType_FLOAT:
        case onnx::TensorProto_DataType_INT32:
            return 4;
        case onnx::TensorProto_DataType_FLOAT16:
        case onnx::TensorProto_DataType_INT16:
        case onnx::TensorProto_DataType_UINT16:
            return 2;
        case onnx::TensorProto_DataType_INT8:
        case onnx::TensorProto_DataType_UINT8:
        case onnx::TensorProto_DataType_BOOL:
            return 1;
        default:
            break;
    }
    return 0;
}

size_t GetTensorBytes(Ort::CustomOpApi& ort, OrtTensorTypeAndShapeInfo* info) {
    return ort.GetTensorShapeElementCount(info) *
           GetTensorElementSize(ort.GetTensorElementType(info));
}

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg) {
    auto shape = node_arg.Shape();
    std::vector<int64_t> dims;
    if (shape != nullptr) {
        for (int i = 0; i < shape->dim_size(); i++) {
            auto dim = shape->dim(i);
            dims.push_back(dim.dim_value());
        }
    }
    if (dims.size() == 0) {
        dims.push_back(1);
    }
    TensorShape ts = dims;
    return ts;
}

void SetTensorDims(const onnxruntime::NodeArg& node_arg, std::vector<uint32_t>& tensor_dims) {
    auto shape = vsi_npu::GetTensorShape(node_arg);
    const std::vector<int64_t>& dims = shape.GetDims();
    for (auto dim : dims) {
        tensor_dims.push_back(static_cast<uint32_t>(dim));
    }
}

std::shared_ptr<uint8_t> UnpackTensor(const NodeArg* node,
                                      const ONNX_NAMESPACE::TensorProto& initializer) {
    std::shared_ptr<uint8_t> unpackedTensor;
    auto shape = vsi_npu::GetTensorShape(*node);
    size_t elementCount = shape.Size();

#define CASE_PROTO(X, Y)                                                                    \
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X: {                  \
        size_t tensorByteSize = elementCount * sizeof(Y);                                   \
        unpackedTensor.reset(new uint8_t[tensorByteSize], shared_array_deletor<uint8_t>()); \
        onnxruntime::utils::UnpackTensor(                                                   \
            initializer,                                                                    \
            initializer.has_raw_data() ? initializer.raw_data().data() : nullptr,           \
            initializer.has_raw_data() ? initializer.raw_data().size() : 0,                 \
            reinterpret_cast<Y*>(unpackedTensor.get()),                                     \
            shape.Size());                                                                  \
        break;                                                                              \
    }
    switch (initializer.data_type()) {
        CASE_PROTO(FLOAT, float);
        CASE_PROTO(DOUBLE, double);
        CASE_PROTO(BOOL, bool);
        CASE_PROTO(INT8, int8_t);
        CASE_PROTO(INT16, int16_t);
        CASE_PROTO(INT32, int32_t);
        CASE_PROTO(INT64, int64_t);
        CASE_PROTO(UINT8, uint8_t);
        CASE_PROTO(UINT16, uint16_t);
        CASE_PROTO(UINT32, uint32_t);
        CASE_PROTO(UINT64, uint64_t);
        CASE_PROTO(FLOAT16, onnxruntime::MLFloat16);
        default:
            return nullptr;
    }

    return unpackedTensor;
}

template <>
Status GetAttr(const OpNodeProtoHelper<ProtoHelperNodeContext>& attrs,
               const std::string& name,
               int32_t* value) {
    int64_t v;
    Status status = GetAttr<int64_t>(attrs, name, &v);
    *value = static_cast<int32_t>(v);
    return status;
}

template <>
Status GetAttrs(const OpNodeProtoHelper<ProtoHelperNodeContext>& attrs,
                const std::string& name,
                std::vector<int32_t>& values,
                bool reverse) {
    std::vector<int64_t> v;
    Status status = GetAttrs<int64_t>(attrs, name, v, reverse);
    for (auto iter = v.begin(); iter != v.end(); iter++) {
        values.push_back(static_cast<int32_t>(*iter));
    }
    return status;
}

template <>
Status GetAttrs(const OrtApi* api,
                OrtKernelContext* context,
                const int32_t index,
                std::vector<int32_t>& values,
                bool reverse) {
    std::vector<int64_t> v;
    Status status = GetAttrs<int64_t>(api, context, index, v, reverse);
    for (auto iter = v.begin(); iter != v.end(); iter++) {
        values.push_back(static_cast<int32_t>(*iter));
    }
    return status;
}

nnrt::PadType GetPadType(const std::string type) {
    static const std::map<std::string, nnrt::PadType> type_table = {
        {"NOTSET", nnrt::PadType::AUTO},
        {"SAME_UPPER", nnrt::PadType::SAME},
        {"SAME_LOWER", nnrt::PadType::SAME},
        {"VALID", nnrt::PadType::VALID},
    };
    auto search = type_table.find(type);
    if (search != type_table.end()) {
        return search->second;
    }
    return nnrt::PadType::AUTO;
}

nnrt::PadMode GetPadMode(const std::string mode) {
    static const std::map<std::string, nnrt::PadMode> mode_table = {
        {"constant", nnrt::PadMode::CONSTANT},
        {"reflect", nnrt::PadMode::REFLECT},
        {"edge", nnrt::PadMode::REPLICATE},
    };
    auto search = mode_table.find(mode);
    if (search != mode_table.end()) {
        return search->second;
    }
    LOGS_DEFAULT(WARNING) << "Unsupported mode: " << mode;
    return nnrt::PadMode::CONSTANT;
}

bool ExcludeType(const NodeArg* node_arg, std::string& reason) {
    // If type is excluded, return false. Otherwise return true.
    const auto* type_proto = node_arg->TypeAsProto();
    if (!type_proto) {
        return false;
    }

    switch (type_proto->tensor_type().elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
            reason += "## only support int64 tensor as attribute.";
            return false;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
            reason += "## only support int32 tensor as attribute.";
            return false;
        default:
            return true;
    }
}

bool CheckMainInputType(const Node* node, std::string& reason) {
    auto input_defs = node->InputDefs();
    return ExcludeType(input_defs[0], reason);
}

bool CheckZeroDim(const NodeArg* node_arg) {
    auto shape = node_arg->Shape();
    if (shape == nullptr || shape->dim_size() == 0) {
        return false;
    }
    for (int i = 0; i < shape->dim_size(); i++) {
        if (shape->dim(i).dim_value() == 0) {
            return false;
        }
    }
    return true;
}

bool CheckAllExcludeType(const Node* node, std::string& reason) {
    bool are_types_supported = true;
    node->ForEachDef(
        [&are_types_supported, &reason](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
            are_types_supported &= ExcludeType(&node_arg, reason);
        });
    return are_types_supported;
}

bool CheckAllZeroDim(const Node* node, std::string& reason) {
    bool are_zero_dim = true;

    node->ForEachDef([&are_zero_dim](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
        are_zero_dim &= vsi_npu::CheckZeroDim(&node_arg);
    });

    if (!are_zero_dim) {
        reason += "## dim with zero not supported.";
        return false;
    }
    return true;
}

}  // namespace vsi_npu
}  // namespace onnxruntime
