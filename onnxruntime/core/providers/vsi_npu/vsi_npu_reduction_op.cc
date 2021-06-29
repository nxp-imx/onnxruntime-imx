/****************************************************************************
 *
 *    Copyright (c) 2021 Vivante Corporation
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
#include "vsi_npu_reduction_op.h"

namespace onnxruntime {

void VsiOpCallbackInfoArgMax::SetupAttribute(nnrt::op::OperationPtr op,
                                          const Node* node,
                                          ModelShellPtr& model,
                                          const onnxruntime::GraphViewer* graph_viewer) {
    auto argmax = std::dynamic_pointer_cast<nnrt::op::ArgmaxOperation>(op);
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    int32_t axis = 0;
    bool status = vsi_npu::GetAttr<int32_t>(attrs, "axis", &axis).IsOK();
    ORT_ENFORCE(status);

    argmax->axis = axis;
}

bool VsiOpCallbackInfoArgMax::IsNodeSupported(
    const onnxruntime::GraphViewer& graph_viewer, const Node* node, std::string& reason) {
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}

void VsiOpCallbackInfoReduceMean::SetupAttribute(nnrt::op::OperationPtr op,
                                          const Node* node,
                                          ModelShellPtr& model,
                                          const onnxruntime::GraphViewer* graph_viewer) {
    auto reduce = std::dynamic_pointer_cast<nnrt::op::ReduceMeanOperation>(op);
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::vector<int32_t> axes;
    bool status = vsi_npu::GetAttrs<int32_t>(attrs, "axes", axes, false).IsOK();
    if (!status) {
        auto input_defs = node->InputDefs();
        int dim_size = input_defs[0]->Shape()->dim_size();
        axes.resize(dim_size);
        std::iota(axes.begin(), axes.end(), 0);
    }

    int32_t keepdims = 0;
    status = vsi_npu::GetAttr<int32_t>(attrs, "keepdims", &keepdims).IsOK();
    ORT_ENFORCE(status);

    reduce->axes = std::move(axes);
    reduce->keepDim = keepdims ? true : false;
}

bool VsiOpCallbackInfoReduceMean::IsNodeSupported(
    const onnxruntime::GraphViewer& graph_viewer, const Node* node, std::string& reason) {
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}

}  // namespace onnxruntime