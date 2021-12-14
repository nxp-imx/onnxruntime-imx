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
#include "vsi_npu_tensor_op.h"

namespace onnxruntime {

static bool PadIsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                               const Node* node,
                               std::string& reason) {
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::string mode;
    bool status = vsi_npu::GetAttr<std::string>(attrs, "mode", &mode).IsOK();
    ORT_ENFORCE(status);

    auto input_def = node->InputDefs()[0];
    const auto* type_proto = input_def->TypeAsProto();

    auto shape = vsi_npu::GetTensorShape(*input_def);
    if (shape.NumDimensions() != 4) {
        reason += "## Only support Pad2D now.";
        return false;
    }

    if (mode == "constant" ||
        (type_proto && type_proto->tensor_type().elem_type() !=
                           ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)) {
        reason += "## Only support constant mode for float32 tensor.";
        return vsi_npu::CheckAllExcludeType(node, reason) && vsi_npu::CheckAllZeroDim(node, reason);
    }
    return false;
}

void VsiOpCallbackInfoPad_1_10::Setup(const onnxruntime::Node* node,
                                      onnxruntime::ModelShellPtr& model,
                                      const onnxruntime::GraphViewer* graph_viewer) {
    auto input_defs = node->InputDefs();
    std::vector<uint32_t> in_operand_ids;
    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    auto output_defs = node->OutputDefs();
    std::vector<uint32_t> out_operand_ids;
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    auto pad = std::make_shared<nnrt::op::PadOperation>();
    pad->setInputs(in_operand_ids.data(), in_operand_ids.size());
    pad->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::string mode;
    bool status = vsi_npu::GetAttr<std::string>(attrs, "mode", &mode).IsOK();
    ORT_ENFORCE(status);
    pad->padMode = vsi_npu::GetPadMode(mode);

    std::vector<int32_t> pads;
    status = vsi_npu::GetAttrs<int32_t>(attrs, "pads", pads, false).IsOK();
    if (pads.size() > 0) {
        for (std::vector<int32_t>::size_type i = 0; i < pads.size() / 2; i++) {
            pad->padFront.push_back(pads[i]);
            pad->padBack.push_back(pads[i + pads.size() / 2]);
        }
    }
    float value;
    status = vsi_npu::GetAttr<float>(attrs, "value", &value).IsOK();
    ORT_ENFORCE(status);
    pad->padValue = value;

    model->AddOperation(pad, nullptr);
}

void VsiOpCallbackInfoPad_11_0::Setup(const onnxruntime::Node* node,
                                      onnxruntime::ModelShellPtr& model,
                                      const onnxruntime::GraphViewer* graph_viewer) {
    auto input_defs = node->InputDefs();
    std::vector<uint32_t> in_operand_ids;
    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    auto output_defs = node->OutputDefs();
    std::vector<uint32_t> out_operand_ids;
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    auto pad = std::make_shared<nnrt::op::PadOperation>();
    pad->setInputs(in_operand_ids.data(), in_operand_ids.size());
    pad->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::string mode;
    bool status = vsi_npu::GetAttr<std::string>(attrs, "mode", &mode).IsOK();
    ORT_ENFORCE(status);
    pad->padMode = vsi_npu::GetPadMode(mode);

    std::vector<int32_t> compute_input_index;
    const int32_t kIndexPads = 1;
    const int32_t kIndexConstantValue = 2;

    std::vector<int32_t> pads;
    model->GetInitializerAsParameters<int32_t>(input_defs[1], graph_viewer, pads);
    if (pads.size() > 0) {
        for (std::vector<int32_t>::size_type i = 0; i < pads.size() / 2; i++) {
            pad->padFront.push_back(pads[i]);
            pad->padBack.push_back(pads[i + pads.size() / 2]);
        }
    } else {
        compute_input_index.push_back(kIndexPads);
    }

    if (input_defs.size() == 3) {
        std::vector<float> constant_value;
        model->GetInitializerAsParameters<float>(input_defs[2], graph_viewer, constant_value);
        if (constant_value.size() > 0) {
            pad->padValue = constant_value[0];
        } else {
            compute_input_index.push_back(kIndexConstantValue);
        }
    }

    auto compute_info = std::make_shared<VsiComputeInfo>();
    compute_info->op = pad;
    model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);

    model->AddOperation(pad, nullptr);
}

bool VsiOpCallbackInfoPad_1_10::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                                const Node* node,
                                                std::string& reason) {
    return PadIsNodeSupported(graph_viewer, node, reason);
}

bool VsiOpCallbackInfoPad_11_0::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                                const Node* node,
                                                std::string& reason) {
    bool res = true;
    res &= vsi_npu::CheckMainInputType(node, reason);
    res &= PadIsNodeSupported(graph_viewer, node, reason);
    return res;
}

Status VsiOpCallbackInfoPad_11_0::Compute(FunctionState state,
                                          const OrtApi* api,
                                          OrtKernelContext* context,
                                          NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto local_model = model->GetModelPtr();

    auto compute_info = model->GetComputeInfo(node_index);

    auto compute_input_ids = model->GetComputeInputIds(compute_info->compute_input_names);

    auto pad = std::dynamic_pointer_cast<nnrt::op::PadOperation>(compute_info->op);

    const int32_t kIndexPads = 0;
    const int32_t kIndexConstantValue = 1;

    std::vector<int32_t> pads;
    Status status =
        vsi_npu::GetAttrs<int32_t>(api, context, compute_input_ids[kIndexPads], pads, false);
    if (pads.size() > 0) {
        for (std::vector<int32_t>::size_type i = 0; i < pads.size() / 2; i++) {
            pad->padFront.push_back(pads[i]);
            pad->padBack.push_back(pads[i + pads.size() / 2]);
        }
    }

    std::vector<float> constant_value;
    status = vsi_npu::GetAttrs<float>(
        api, context, compute_input_ids[kIndexConstantValue], constant_value, false);
    if (constant_value.size() > 0) {
        pad->padValue = constant_value[0];
    }

    return Status::OK();
}

void VsiOpCallbackInfoConcat::SetupAttribute(nnrt::op::OperationPtr op,
                                             const Node* node,
                                             ModelShellPtr& model,
                                             const onnxruntime::GraphViewer* graph_viewer) {
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    int32_t axis;
    bool status = vsi_npu::GetAttr<int32_t>(attrs, "axis", &axis).IsOK();
    ORT_ENFORCE(status);

    auto concat = std::dynamic_pointer_cast<nnrt::op::ConcatOperation>(op);
    concat->axis = axis;
}

void VsiOpCallbackInfoReshape::SetupIO(nnrt::op::OperationPtr op,
                                       const Node* node,
                                       ModelShellPtr& model,
                                       const onnxruntime::GraphViewer* graph_viewer) {
    std::vector<uint32_t> in_operand_ids;
    auto input_defs = node->InputDefs();
    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    std::vector<uint32_t> out_operand_ids;
    auto output_defs = node->OutputDefs();
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    auto reshape = std::dynamic_pointer_cast<nnrt::op::ReshapeOperation>(op);
    reshape->setInputs(in_operand_ids.data(), in_operand_ids.size());
    reshape->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    std::vector<int32_t> shape;
    model->GetInitializerAsParameters<int32_t>(input_defs[1], graph_viewer, shape);
    reshape->shape = std::move(shape);
}

bool VsiOpCallbackInfoReshape::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                               const Node* node,
                                               std::string& reason) {
    return vsi_npu::CheckMainInputType(node, reason) && vsi_npu::CheckAllZeroDim(node, reason);
}

void VsiOpCallbackInfoTranspose::Setup(const Node* node,
                                       ModelShellPtr& model,
                                       const onnxruntime::GraphViewer* graph_viewer) {
    std::vector<uint32_t> in_operand_ids;
    auto op = std::make_shared<nnrt::op::PermuteOperation>();
    auto input_defs = node->InputDefs();

    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    std::vector<uint32_t> out_operand_ids;
    auto output_defs = node->OutputDefs();
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::vector<int32_t> perm;
    bool status = vsi_npu::GetAttrs<int32_t>(attrs, "perm", perm, false).IsOK();
    if (status) {
        op->perm = std::move(perm);
    } else {
        std::vector<int32_t> perm_default = {1, 0};
        op->perm = std::move(perm_default);
    }

    model->AddOperation(op, nullptr);
}
}  // namespace onnxruntime