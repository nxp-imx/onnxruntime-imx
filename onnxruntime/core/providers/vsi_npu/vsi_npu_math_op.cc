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

void VsiOpCallbackInfoMatMul::Setup(const Node* node,
                                    ModelShellPtr& model,
                                    const onnxruntime::GraphViewer* graph_viewer) {
    auto op = std::make_shared<nnrt::op::MatrixMulOperation>();
    std::vector<uint32_t> in_operand_ids;
    auto input_defs = node->InputDefs();
    for (auto input_def : input_defs) {
        uint32_t input_operand_id = model->AddOperand(input_def, graph_viewer);
        in_operand_ids.push_back(input_operand_id);
    }

    std::vector<uint32_t> out_operand_ids;
    auto output_defs = node->OutputDefs();
    for (auto output_def : output_defs) {
        uint32_t output_operand_id = model->AddOperand(output_def, graph_viewer);
        out_operand_ids.push_back(output_operand_id);
    }
    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());
    op->transpose[0] = false;
    op->transpose[1] = false;

    model->AddOperation(op, nullptr);
}

bool VsiOpCallbackInfoMatMul::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                              const Node* node,
                                              std::string& reason) {
    auto input_defs = node->InputDefs();
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() != 2) {
        reason += "## Only support Matmul 2D now.";
        return false;
    }
    shape = vsi_npu::GetTensorShape(*input_defs[1]);
    if (shape.NumDimensions() != 2) {
        reason += "## Only support Matmul 2D now.";
        return false;
    }
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}
}  // namespace onnxruntime