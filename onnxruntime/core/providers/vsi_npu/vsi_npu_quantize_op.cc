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

    std::vector<float> scales;
    std::vector<uint8_t> zps;
    model->GetInitializerAsParameters(input_defs[1], graph_viewer, scales);
    model->GetInitializerAsParameters(input_defs[2], graph_viewer, zps);
    if (scales.size() == 0) {
        const int32_t kIndexScale = 1;
        const int32_t kIndexZeroPoint = 2;
        std::vector<int32_t> compute_input_index({kIndexScale, kIndexZeroPoint});
        auto compute_info = std::make_shared<VsiComputeInfo>();
        compute_info->operand_ids.push_back(input_operand_id);
        model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
    } else {
        auto tensor = model->GetModelPtr()->operand(input_operand_id);
        tensor->quant.scalar.scale = scales[0];
        tensor->quant.scalar.zeroPoint = zps[0];
    }
}

Status VsiOpCallbackInfoDequantizeLinear::Compute(FunctionState state,
                                                  const OrtApi* api,
                                                  OrtKernelContext* context,
                                                  NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto local_model = model->GetModelPtr();

    auto compute_info = model->GetComputeInfo(node_index);
    if (compute_info != nullptr) {
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
    }
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

    std::vector<float> scales;
    std::vector<uint8_t> zps;
    model->GetInitializerAsParameters(input_defs[1], graph_viewer, scales);
    model->GetInitializerAsParameters(input_defs[2], graph_viewer, zps);

    if (scales.size() == 0) {
        const int32_t kIndexScale = 1;
        const int32_t kIndexZeroPoint = 2;
        std::vector<int32_t> compute_input_index({kIndexScale, kIndexZeroPoint});
        auto compute_info = std::make_shared<VsiComputeInfo>();
        compute_info->operand_ids.push_back(output_operand_id);
        model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
    } else {
        auto tensor = model->GetModelPtr()->operand(output_operand_id);
        tensor->quant.scalar.scale = scales[0];
        tensor->quant.scalar.zeroPoint = zps[0];
    }
}

Status VsiOpCallbackInfoQuantizeLinear::Compute(FunctionState state,
                                                const OrtApi* api,
                                                OrtKernelContext* context,
                                                NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto local_model = model->GetModelPtr();

    auto compute_info = model->GetComputeInfo(node_index);
    if (compute_info != nullptr) {
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
    }
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

void VsiOpCallbackInfoQLinearConv::Setup(const onnxruntime::Node* node,
                                         onnxruntime::ModelShellPtr& model,
                                         const onnxruntime::GraphViewer* graph_viewer) {
    auto op = std::make_shared<nnrt::op::GroupedConv2DOperation>();
    auto input_defs = node->InputDefs();

    uint32_t input_operand_id_x = model->AddOperand(input_defs[0], graph_viewer);
    uint32_t input_operand_id_w = -1;
    uint32_t input_operand_id_b = -1;
    uint32_t output_operand_id = -1;

    std::vector<uint32_t> in_operand_ids{input_operand_id_x};
    std::vector<uint32_t> out_operand_ids;

    std::vector<float> x_scales;
    model->GetInitializerAsParameters(input_defs[1], graph_viewer, x_scales);

    auto output_defs = node->OutputDefs();
    output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    if (x_scales.size() == 0) {
        input_operand_id_w = model->AddOperand(input_defs[3], graph_viewer);
        if (input_defs.size() < 9) {
            input_operand_id_b = model->AddOperand(input_defs[3]->Name() + "b");
            AddBiasOperand(node, model, input_operand_id_b);
        } else {
            input_operand_id_b = model->AddOperand(input_defs[8], graph_viewer);
        }

    } else {
        std::vector<uint8_t> x_zps;
        std::vector<float> w_scales;
        std::vector<uint8_t> w_zps;
        std::vector<float> y_scales;
        std::vector<uint8_t> y_zps;
        model->GetInitializerAsParameters(input_defs[2], graph_viewer, x_zps);
        model->GetInitializerAsParameters(input_defs[4], graph_viewer, w_scales);
        model->GetInitializerAsParameters(input_defs[5], graph_viewer, w_zps);
        model->GetInitializerAsParameters(input_defs[6], graph_viewer, y_scales);
        model->GetInitializerAsParameters(input_defs[7], graph_viewer, y_zps);

        input_operand_id_w = model->AddOperand(input_defs[3], graph_viewer);
        if (input_defs.size() < 9) {
            input_operand_id_b = model->AddOperand(input_defs[3]->Name() + "b");
            AddBiasOperand(node, model, input_operand_id_b);
        } else if (input_defs.size() == 9) {
            input_operand_id_b = model->AddOperand(input_defs[8], graph_viewer);
        }

        auto local_model = model->GetModelPtr();
        auto x_tensor = local_model->operand(input_operand_id_x);
        x_tensor->quant.scalar.scale = x_scales[0];
        x_tensor->quant.scalar.zeroPoint = x_zps[0];

        auto w_tensor = local_model->operand(input_operand_id_w);
        w_tensor->quant.scalar.scale = w_scales[0];
        w_tensor->quant.scalar.zeroPoint = w_zps[0];

        auto y_tensor = local_model->operand(output_operand_id);
        y_tensor->quant.scalar.scale = y_scales[0];
        y_tensor->quant.scalar.zeroPoint = y_zps[0];
    }
    in_operand_ids.push_back(input_operand_id_w);
    in_operand_ids.push_back(input_operand_id_b);
    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

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

    std::vector<int32_t> vstrides(2, 1);
    std::vector<int32_t> strides;
    status = vsi_npu::GetAttrs<int32_t>(attrs, "strides", strides, true).IsOK();
    if (status) {
        vstrides = std::move(strides);
    }

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
    op->strides = std::move(vstrides);
    op->setDataLayout(nnrt::DataLayout::NCHW);
    op->setVxParam(
        nnrt::OverflowPolicy::SATURATE, nnrt::RoundingPolicy::TO_ZERO, nnrt::Rounding::FLOOR);

    if (x_scales.size() == 0) {
        const int32_t xIndexScale = 1;
        const int32_t xIndexZeroPoint = 2;
        const int32_t wIndexScale = 4;
        const int32_t wIndexZeroPoint = 5;
        const int32_t yIndexScale = 6;
        const int32_t yIndexZeroPoint = 7;
        std::vector<int32_t> compute_input_index({xIndexScale,
                                                  xIndexZeroPoint,
                                                  wIndexScale,
                                                  wIndexZeroPoint,
                                                  yIndexScale,
                                                  yIndexZeroPoint});
        auto compute_info = std::make_shared<VsiComputeInfo>();
        compute_info->operand_ids.push_back(input_operand_id_x);
        compute_info->operand_ids.push_back(input_operand_id_w);
        compute_info->operand_ids.push_back(output_operand_id);
        model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
    }
    model->AddOperation(op, nullptr);
}

void VsiOpCallbackInfoQLinearConv::AddBiasOperand(const onnxruntime::Node* node,
                                                  onnxruntime::ModelShellPtr& model,
                                                  uint32_t operand_id) {
    auto input_defs = node->InputDefs();
    auto operand_b = model->GetOperand(operand_id);
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    const std::vector<int64_t>& dims = shape.GetDims();
    operand_b->dimensions.push_back(dims[0]);
    operand_b->type = nnrt::OperandType::TENSOR_INT32;

    auto operand_b_size = operand_b->size();
    auto value = new float[operand_b_size];
    for (size_t i = 0; i < operand_b_size; i++) {
        value[i] = 0;
    }
    std::shared_ptr<float> tensorValue(value);
    const void* value_addr = reinterpret_cast<const void*>(tensorValue.get());
    model->GetModelPtr()->setOperandValue(operand_b, value_addr, operand_b->bytes());
}

Status VsiOpCallbackInfoQLinearConv::Compute(FunctionState state,
                                             const OrtApi* api,
                                             OrtKernelContext* context,
                                             NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto local_model = model->GetModelPtr();

    auto compute_info = model->GetComputeInfo(node_index);
    if (compute_info != nullptr) {
        auto compute_input_ids = model->GetComputeInputIds(compute_info->compute_input_names);

        const OrtValue* x_tensor_scale = ort.KernelContext_GetInput(context, compute_input_ids[0]);
        const auto x_tensor_scale_value = (float*)ort.GetTensorData<void>(x_tensor_scale);

        const OrtValue* x_tensor_zp = ort.KernelContext_GetInput(context, compute_input_ids[1]);
        const auto x_tensor_zp_value = (uint8_t*)ort.GetTensorData<void>(x_tensor_zp);

        auto x_tensor = local_model->operand(compute_info->operand_ids[0]);
        x_tensor->quant.scalar.scale = *x_tensor_scale_value;
        x_tensor->quant.scalar.zeroPoint = *x_tensor_zp_value;

        const OrtValue* w_tensor_scale = ort.KernelContext_GetInput(context, compute_input_ids[2]);
        const auto w_tensor_scale_value = (float*)ort.GetTensorData<void>(w_tensor_scale);

        const OrtValue* w_tensor_zp = ort.KernelContext_GetInput(context, compute_input_ids[3]);
        const auto w_tensor_zp_value = (uint8_t*)ort.GetTensorData<void>(w_tensor_zp);

        auto w_tensor = local_model->operand(compute_info->operand_ids[1]);
        w_tensor->quant.scalar.scale = *w_tensor_scale_value;
        w_tensor->quant.scalar.zeroPoint = *w_tensor_zp_value;

        const OrtValue* y_tensor_scale = ort.KernelContext_GetInput(context, compute_input_ids[4]);
        const auto y_tensor_scale_value = (float*)ort.GetTensorData<void>(y_tensor_scale);

        const OrtValue* y_tensor_zp = ort.KernelContext_GetInput(context, compute_input_ids[5]);
        const auto y_tensor_zp_value = (uint8_t*)ort.GetTensorData<void>(y_tensor_zp);

        auto y_tensor = local_model->operand(compute_info->operand_ids[2]);
        y_tensor->quant.scalar.scale = *y_tensor_scale_value;
        y_tensor->quant.scalar.zeroPoint = *y_tensor_zp_value;
    }

    return Status::OK();
}

bool VsiOpCallbackInfoQLinearConv::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                                   const Node* node,
                                                   std::string& reason) {
    auto input_defs = node->InputDefs();
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() != 4) {
        reason += "## Only support Conv2D now.";
        return false;
    }
    return vsi_npu::CheckMainInputType(node, reason);
}
}  // namespace onnxruntime