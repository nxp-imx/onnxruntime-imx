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
#include "vsi_npu_quantize_op.h"

namespace onnxruntime {

void VsiOpCallbackInfoDequantizeLinear::SetupIO(nnrt::op::OperationPtr op,
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

    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    const int32_t kIndexScale = 1;
    const int32_t kIndexZeroPoint = 2;
    std::vector<int32_t> compute_input_index({kIndexScale, kIndexZeroPoint});
    auto compute_info = std::make_shared<VsiComputeInfo>();
    compute_info->operand_ids.push_back(input_operand_id);
    model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
}

Status VsiOpCallbackInfoDequantizeLinear::Compute(FunctionState state,
                                                  const OrtApi* api,
                                                  OrtKernelContext* context,
                                                  NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto local_model = model->GetModelPtr();

    auto compute_info = model->GetComputeInfo(node_index);

    auto compute_input_ids = model->GetComputeInputIds(compute_info->compute_input_names);

    const int32_t kIndexScale = 0;
    const int32_t kIndexZeroPoint = 1;

    const OrtValue* input_tensor_scale =
        ort.KernelContext_GetInput(context, compute_input_ids[kIndexScale]);
    const auto input_tensor_scale_value = (float*)ort.GetTensorData<void>(input_tensor_scale);

    const OrtValue* input_tensor_zp =
        ort.KernelContext_GetInput(context, compute_input_ids[kIndexZeroPoint]);
    const auto input_tensor_zp_value = (uint8_t*)ort.GetTensorData<void>(input_tensor_zp);

    auto tensor = local_model->operand(compute_info->operand_ids[0]);
    tensor->quant.scalar.scale = *input_tensor_scale_value;
    tensor->quant.scalar.zeroPoint = *input_tensor_zp_value;

    return Status::OK();
}

bool VsiOpCallbackInfoDequantizeLinear::IsNodeSupported(
    const onnxruntime::GraphViewer& graph_viewer, const Node* node, std::string& reason) {
    return vsi_npu::CheckAllExcludeType(node, reason);
}

void VsiOpCallbackInfoQuantizeLinear::SetupIO(nnrt::op::OperationPtr op,
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

    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    const int32_t kIndexScale = 1;
    const int32_t kIndexZeroPoint = 2;
    std::vector<int32_t> compute_input_index({kIndexScale, kIndexZeroPoint});
    auto compute_info = std::make_shared<VsiComputeInfo>();
    compute_info->operand_ids.push_back(output_operand_id);
    model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
}

Status VsiOpCallbackInfoQuantizeLinear::Compute(FunctionState state,
                                                const OrtApi* api,
                                                OrtKernelContext* context,
                                                NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto local_model = model->GetModelPtr();

    auto compute_info = model->GetComputeInfo(node_index);

    auto compute_input_ids = model->GetComputeInputIds(compute_info->compute_input_names);

    const int32_t kIndexScale = 0;
    const int32_t kIndexZeroPoint = 1;

    const OrtValue* input_tensor_scale =
        ort.KernelContext_GetInput(context, compute_input_ids[kIndexScale]);
    const auto input_tensor_scale_value = (float*)ort.GetTensorData<void>(input_tensor_scale);

    const OrtValue* input_tensor_zp =
        ort.KernelContext_GetInput(context, compute_input_ids[kIndexZeroPoint]);
    const auto input_tensor_zp_value = (uint8_t*)ort.GetTensorData<void>(input_tensor_zp);

    auto tensor = local_model->operand(compute_info->operand_ids[0]);
    tensor->quant.scalar.scale = *input_tensor_scale_value;
    tensor->quant.scalar.zeroPoint = *input_tensor_zp_value;

    return Status::OK();
}

bool VsiOpCallbackInfoQuantizeLinear::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                                      const Node* node,
                                                      std::string& reason) {
    return vsi_npu::CheckAllExcludeType(node, reason);
}

void VsiOpCallbackInfoConvInteger::Setup(const onnxruntime::Node* node,
                                         onnxruntime::ModelShellPtr& model,
                                         const onnxruntime::GraphViewer* graph_viewer) {
    std::vector<uint32_t> conv_in_operand_ids;
    auto input_defs = node->InputDefs();
    uint32_t input_operand_id_x = model->AddOperand(input_defs[0], graph_viewer);
    conv_in_operand_ids.push_back(input_operand_id_x);

    uint32_t input_operand_id_w = model->AddOperand(input_defs[1], graph_viewer);
    conv_in_operand_ids.push_back(input_operand_id_w);

    std::vector<uint32_t> conv_out_operand_ids;
    auto output_defs = node->OutputDefs();

    std::string add_name = "@@ConvIntegerTmp";
    auto tmp_tensor_info = std::make_shared<VsiGraphTensorInfo>();
    tmp_tensor_info->name = output_defs[0]->Name() + add_name;
    tmp_tensor_info->is_initializer = true;
    model->GetGraphInputs().push_back(tmp_tensor_info);

    uint32_t tmp_output_operand_id = model->AddOperand(output_defs[0]->Name() + add_name);
    auto operand = model->GetOperand(tmp_output_operand_id);
    auto shape = vsi_npu::GetTensorShape(*output_defs[0]);
    vsi_npu::SetTensorDims(*output_defs[0], operand->dimensions);
    operand->type = nnrt::OperandType::TENSOR_FLOAT16;
    conv_out_operand_ids.push_back(tmp_output_operand_id);

    auto op = std::make_shared<nnrt::op::GroupedConv2DOperation>();
    op->setInputs(conv_in_operand_ids.data(), conv_in_operand_ids.size());
    op->setOutputs(conv_out_operand_ids.data(), conv_out_operand_ids.size());

    std::vector<uint32_t> trans_out_operand_ids;
    uint32_t trans_out_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    auto op_trans = std::make_shared<nnrt::op::QuantizeOperation>();
    trans_out_operand_ids.push_back(trans_out_operand_id);

    op_trans->setInputs(conv_out_operand_ids.data(), conv_out_operand_ids.size());
    op_trans->setOutputs(trans_out_operand_ids.data(), trans_out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::vector<int32_t> vpads(4, 0);
    std::vector<int32_t> pads;
    bool status = vsi_npu::GetAttrs<int32_t>(attrs, "pads", pads, false).IsOK();
    if (status) {
        vpads[0] = pads[1];
        vpads[1] = pads[3];
        vpads[2] = pads[0];
        vpads[3] = pads[2];
    }

    std::string auto_pad;
    status = vsi_npu::GetAttr<std::string>(attrs, "auto_pad", &auto_pad).IsOK();
    nnrt::PadType pad_type = nnrt::PadType::AUTO;
    if (status) {
        pad_type = vsi_npu::GetPadType(auto_pad);
    }

    std::vector<int32_t> strides(2, 1);
    status = vsi_npu::GetAttrs<int32_t>(attrs, "strides", strides, true).IsOK();

    std::vector<int32_t> vdilations(2, 1);
    std::vector<int32_t> dilations;
    status = vsi_npu::GetAttrs<int32_t>(attrs, "dilations", dilations, true).IsOK();
    if (status) {
        vdilations = std::move(dilations);
    }

    int32_t group = 1;
    status = vsi_npu::GetAttr<int32_t>(attrs, "group", &group).IsOK();

    op->pad = std::move(vpads);
    op->dilations = std::move(vdilations);
    op->groups = group;
    op->padType = pad_type;
    op->strides = std::move(strides);
    op->setDataLayout(nnrt::DataLayout::NCHW);
    op->setVxParam(
        nnrt::OverflowPolicy::SATURATE, nnrt::RoundingPolicy::TO_ZERO, nnrt::Rounding::FLOOR);

    model->AddOperation(op, nullptr);
    model->AddOperation(op_trans, nullptr);

    const int32_t kIndexW = 1;
    const int32_t kIndexZeroPoint = 2;
    std::vector<int32_t> compute_index({kIndexW, kIndexZeroPoint});
    auto compute_info = std::make_shared<VsiComputeInfo>();
    compute_info->operand_ids.push_back(input_operand_id_x);
    compute_info->operand_ids.push_back(input_operand_id_w);
    model->CollectComputeInfo(node, graph_viewer, compute_index, compute_info);
}

Status VsiOpCallbackInfoConvInteger::Compute(FunctionState state,
                                             const OrtApi* api,
                                             OrtKernelContext* context,
                                             NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto local_model = model->GetModelPtr();

    auto compute_info = model->GetComputeInfo(node_index);
    auto compute_input_ids = model->GetComputeInputIds(compute_info->compute_input_names);

    const OrtValue* inout_tensor_zp = ort.KernelContext_GetInput(context, compute_input_ids[1]);
    const auto inout_tensor_zp_value = (uint8_t*)ort.GetTensorData<void>(inout_tensor_zp);

    auto tensor_x = local_model->operand(compute_info->operand_ids[0]);

    tensor_x->quant.scalar.zeroPoint = *inout_tensor_zp_value;
    tensor_x->quant.scalar.scale = 1.0f;

    auto tensor_w = local_model->operand(compute_info->operand_ids[1]);
    tensor_w->quant.scalar.zeroPoint = 0;
    tensor_w->quant.scalar.scale = 1.0f;

    auto tensor_size_w = tensor_w->size();

    std::shared_ptr<uint8_t> tensorValue(new uint8_t[tensor_size_w]);
    const OrtValue* input_tensor_w = ort.KernelContext_GetInput(context, compute_input_ids[0]);
    const auto input_tensor_w_value = ort.GetTensorData<void>(input_tensor_w);
    memcpy(tensorValue.get(), input_tensor_w_value, tensor_size_w);

    const void* value_addr = reinterpret_cast<const void*>(tensorValue.get());
    model->GetModelPtr()->setOperandValue(
        compute_info->operand_ids[1], value_addr, tensor_w->bytes());

    return Status::OK();
}

bool VsiOpCallbackInfoConvInteger::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                                   const Node* node,
                                                   std::string& reason) {
    return vsi_npu::CheckMainInputType(node, reason);
}

}  // namespace onnxruntime