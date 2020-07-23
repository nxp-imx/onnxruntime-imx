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
#include "vsi_npu_math_op.h"

namespace onnxruntime {

void VsiOpCallbackInfoAbs::Setup(const Node* node,
                                 ModelShellPtr& model,
                                 const onnxruntime::GraphViewer* graph_viewer) {
    std::vector<uint32_t> in_operand_ids;
    auto op = std::make_shared<nnrt::op::AbsOperation>();
    auto input_defs = node->InputDefs();

    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    std::vector<uint32_t> out_operand_ids;
    auto output_defs = node->OutputDefs();
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    model->AddOperation(op, nullptr);

    const auto* type_proto = input_defs[0]->TypeAsProto();
    if (type_proto->tensor_type().elem_type() ==
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 ||
        type_proto->tensor_type().elem_type() ==
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
        auto input_tensor = model->GetModelPtr()->operand(input_operand_id);
        input_tensor->quant.scalar.zeroPoint = 0;
        input_tensor->quant.scalar.scale = 1.0f;

        auto output_tensor = model->GetModelPtr()->operand(output_operand_id);
        output_tensor->quant.scalar.zeroPoint = 0;
        output_tensor->quant.scalar.scale = 1.0f;
    }
}
}  // namespace onnxruntime