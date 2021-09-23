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
#include "vsi_npu_pool_op.h"

namespace onnxruntime {

bool PoolIsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                         const Node* node,
                         std::string& reason) {
    auto input_defs = node->InputDefs();
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() != 4) {
        reason += "## Only support Pool2D now.";
        return false;
    }
    return vsi_npu::CheckAllExcludeType(node, reason) && vsi_npu::CheckAllZeroDim(node, reason);
}

bool VsiOpCallbackInfoMaxPool::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                               const Node* node,
                                               std::string& reason) {
    bool res = PoolIsNodeSupported(graph_viewer, node, reason);
    if (!res) {
        return false;
    }

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::vector<int32_t> dilations;
    bool status = vsi_npu::GetAttrs<int32_t>(attrs, "dilations", dilations, false).IsOK();
    if (status) {
        reason += "## Do NOT support dilations now.";
        return false;
    }

    auto output_defs = node->OutputDefs();
    if (output_defs.size() > 1) {
        reason += "## Do NOT support Indices output now.";
        return false;
    }
    return true;
}

void VsiOpCallbackInfoAveragePool::SetupAttribute(nnrt::op::OperationPtr op,
                                                  const Node* node,
                                                  ModelShellPtr& model,
                                                  const onnxruntime::GraphViewer* graph_viewer) {
    VsiOpCallbackInfoPoolOp<nnrt::op::AveragePool2DOperation>::SetupAttribute(
        op, node, model, graph_viewer);
    auto pool = std::dynamic_pointer_cast<nnrt::op::AveragePool2DOperation>(op);

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    int32_t count_include_pad;
    bool status = vsi_npu::GetAttr<int32_t>(attrs, "count_include_pad", &count_include_pad).IsOK();
    if (status) {
        if (count_include_pad == 0) {
            pool->poolMode = nnrt::PoolMode::VALID;
        } else {
            pool->poolMode = nnrt::PoolMode::FULL;
        }
    }
}

}