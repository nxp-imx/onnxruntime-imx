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
#include "vsi_npu_ort_interpreter.h"
#include "vsi_npu_pool_op.h"
#include "vsi_npu_quantize_op.h"
#include "vsi_npu_tensor_op.h"

namespace onnxruntime {

int VsiOrtInterpreter::run(nnrt::Model*, bool*) {
    return 0;
}

VsiOrtInterpreter::VsiOrtInterpreter() {}

void VsiOpCallbackInfo::SetupIO(nnrt::op::OperationPtr op,
                                const Node* node,
                                ModelShellPtr& model,
                                const onnxruntime::GraphViewer* graph_viewer) {
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
}

bool VsiOpCallbackInfo::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                        const Node* node,
                                        std::string& reason) {
    return vsi_npu::CheckAllExcludeType(node, reason) && vsi_npu::CheckAllZeroDim(node, reason);
}

void VsiOpCallbackInfo::ConstInputOprands(FunctionState state,
                                          const OrtApi* api,
                                          OrtKernelContext* context,
                                          NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);

    auto compute_info = model->GetComputeInfo(node_index);
    auto compute_input_ids = model->GetComputeInputIds(compute_info->compute_input_names);

    for (size_t i = 0; i < compute_input_ids.size(); i++) {
        model->ConstInputOprand(api, context, compute_info, compute_input_ids, i);
    }
}

void VsiOpCallbackInfoConv::SetupAttribute(nnrt::op::OperationPtr op,
                                           const Node* node,
                                           ModelShellPtr& model,
                                           const onnxruntime::GraphViewer* graph_viewer) {
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

    // add stride
    std::vector<int32_t> strides;
    status = vsi_npu::GetAttrs<int32_t>(attrs, "strides", strides, true).IsOK();
    ORT_ENFORCE(status);

    std::vector<int32_t> vdilations(2, 1);
    std::vector<int32_t> dilations;
    status = vsi_npu::GetAttrs<int32_t>(attrs, "dilations", dilations, true).IsOK();
    if (status) {
        vdilations = std::move(dilations);
    }

    int32_t group;
    status = vsi_npu::GetAttr<int32_t>(attrs, "group", &group).IsOK();
    ORT_ENFORCE(status);

    auto conv2d_ = std::dynamic_pointer_cast<nnrt::op::GroupedConv2DOperation>(op);
    conv2d_->groups = group;
    conv2d_->pad = std::move(vpads);
    conv2d_->strides = std::move(strides);
    conv2d_->dilations = std::move(vdilations);
    conv2d_->padType = pad_type;
    conv2d_->setDataLayout(nnrt::DataLayout::NCHW);
    conv2d_->setVxParam(
        nnrt::OverflowPolicy::SATURATE, nnrt::RoundingPolicy::TO_ZERO, nnrt::Rounding::FLOOR);
}

bool VsiOpCallbackInfoConv::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                            const Node* node,
                                            std::string& reason) {
    auto input_defs = node->InputDefs();
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() != 4) {
        reason += "## Only support Conv2D now.";
        return false;
    }
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}

void VsiOpCallbackInfoSoftmax::SetupAttribute(nnrt::op::OperationPtr op,
                                              const Node* node,
                                              ModelShellPtr& model,
                                              const onnxruntime::GraphViewer* graph_viewer) {
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    auto softmax = std::dynamic_pointer_cast<nnrt::op::SoftmaxOperation>(op);
    int32_t axis;
    bool status = vsi_npu::GetAttr<int32_t>(attrs, "axis", &axis).IsOK();
    ORT_ENFORCE(status);
    softmax->axis = axis;
}

bool VsiOpCallbackInfoSoftmax::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                        const Node* node,
                                        std::string& reason) {
    auto input_defs = node->InputDefs();
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() != 2) {
        reason += "## Only support Softmax2D now.";
        return false;
    }
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}

void VsiOpCallbackInfoGemm::AddMulOp(const onnxruntime::Node* node,
                                     onnxruntime::ModelShellPtr& model,
                                     std::vector<uint32_t> mul_operand_ids,
                                     std::vector<std::string> mul_add_names,
                                     uint32_t num) {
    auto input_defs = node->InputDefs();
    auto output_defs = node->OutputDefs();
    auto mul = std::make_shared<nnrt::op::MulOperation>();

    std::vector<uint32_t> mul_in_operand_ids{mul_operand_ids[0], mul_operand_ids[1]};
    std::vector<uint32_t> mul_out_operand_ids{mul_operand_ids[2]};

    auto mul_input_tensor_info = std::make_shared<VsiGraphTensorInfo>();
    mul_input_tensor_info->name = input_defs[num]->Name() + mul_add_names[0 + num * 2];
    mul_input_tensor_info->is_initializer = true;
    model->GetGraphInputs().push_back(mul_input_tensor_info);
    model->GetGraphOutputs().push_back(mul_input_tensor_info);

    auto mul_output_tensor_info = std::make_shared<VsiGraphTensorInfo>();
    mul_output_tensor_info->name = input_defs[num]->Name() + mul_add_names[1 + num * 2];
    mul_output_tensor_info->is_initializer = true;
    model->GetGraphInputs().push_back(mul_output_tensor_info);

    auto mul_intput_operand_a = model->GetOperand(mul_operand_ids[0]);
    if (mul_intput_operand_a->ndim() == 0) {
        mul_intput_operand_a->type = nnrt::OperandType::TENSOR_FLOAT32;
        auto intput0_shape = vsi_npu::GetTensorShape(*input_defs[0]);
        auto intput1_shape = vsi_npu::GetTensorShape(*input_defs[1]);
        const std::vector<int64_t>& intput0_dims = intput0_shape.GetDims();
        const std::vector<int64_t>& intput1_dims = intput1_shape.GetDims();
        mul_intput_operand_a->dimensions.push_back(static_cast<uint32_t>(intput0_dims[0]));
        mul_intput_operand_a->dimensions.push_back(static_cast<uint32_t>(intput1_dims[1]));
    }
    auto mul_intput_operand_b = model->GetOperand(mul_operand_ids[1]);
    if (mul_intput_operand_b->ndim() == 0) {
        mul_intput_operand_b->type = nnrt::OperandType::TENSOR_FLOAT32;
        if (num == 1) {
            vsi_npu::SetTensorDims(*input_defs[2], mul_intput_operand_b->dimensions);
        } else if (num == 0) {
            auto intput0_shape = vsi_npu::GetTensorShape(*input_defs[0]);
            auto intput1_shape = vsi_npu::GetTensorShape(*input_defs[1]);
            const std::vector<int64_t>& intput0_dims = intput0_shape.GetDims();
            const std::vector<int64_t>& intput1_dims = intput1_shape.GetDims();
            mul_intput_operand_b->dimensions.push_back(static_cast<uint32_t>(intput0_dims[0]));
            mul_intput_operand_b->dimensions.push_back(static_cast<uint32_t>(intput1_dims[1]));
        }
    }

    auto mul_output_operand = model->GetOperand(mul_operand_ids[2]);
    if (mul_output_operand->ndim() == 0) {
        mul_output_operand->type = nnrt::OperandType::TENSOR_FLOAT32;
        vsi_npu::SetTensorDims(*output_defs[0], mul_output_operand->dimensions);
    }

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    float alpha{1.0};
    float beta{1.0};
    auto status_alpha = vsi_npu::GetAttr<float>(attrs, "alpha", &alpha).IsOK();
    auto status_beta = vsi_npu::GetAttr<float>(attrs, "beta", &beta).IsOK();

    auto tensor_scale = model->GetModelPtr()->operand(mul_operand_ids[1]);
    auto tensor_size_scale = tensor_scale->size();
    auto value = new float[tensor_size_scale];

    if (num == 0 && status_alpha) {
        for (uint16_t i = 0; i < tensor_size_scale; i++) {
            value[i] = alpha;
        }

    } else if (num == 1 && status_beta) {
        for (uint16_t i = 0; i < tensor_size_scale; i++) {
            value[i] = beta;
        }
    }

    std::shared_ptr<float> tensorValue(value);
    const void* value_addr = reinterpret_cast<const void*>(tensorValue.get());
    model->GetModelPtr()->setOperandValue(mul_operand_ids[1], value_addr, tensor_scale->bytes());

    mul->setInputs(mul_in_operand_ids.data(), mul_in_operand_ids.size());
    mul->setOutputs(mul_out_operand_ids.data(), mul_out_operand_ids.size());
    model->AddOperation(mul, nullptr);
}

void VsiOpCallbackInfoGemm::AddTransposeOp(const onnxruntime::Node* node,
                                           onnxruntime::ModelShellPtr& model,
                                           std::vector<uint32_t> trans_operand_ids,
                                           std::string trans_add_name) {
    auto trans = std::make_shared<nnrt::op::PermuteOperation>();

    std::vector<uint32_t> trans_in_operand_ids{trans_operand_ids[0]};
    std::vector<uint32_t> trans_output_operand_ids{trans_operand_ids[1]};

    auto input_defs = node->InputDefs();
    auto output_defs = node->OutputDefs();

    auto trans_output_tensor_info = std::make_shared<VsiGraphTensorInfo>();
    trans_output_tensor_info->name = output_defs[0]->Name() + trans_add_name;
    trans_output_tensor_info->is_initializer = true;
    model->GetGraphInputs().push_back(trans_output_tensor_info);
    model->GetGraphOutputs().push_back(trans_output_tensor_info);

    auto trans_operand = model->GetOperand(trans_operand_ids[1]);
    if (trans_operand->ndim() == 0) {
        trans_operand->type = nnrt::OperandType::TENSOR_FLOAT32;
        auto shape = vsi_npu::GetTensorShape(*input_defs[1]);
        const std::vector<int64_t>& dims = shape.GetDims();
        trans_operand->dimensions.push_back(static_cast<uint32_t>(dims[1]));
        trans_operand->dimensions.push_back(static_cast<uint32_t>(dims[0]));
    }

    std::vector<int32_t> perm_default = {1, 0};
    trans->perm = std::move(perm_default);

    trans->setInputs(trans_in_operand_ids.data(), trans_in_operand_ids.size());
    trans->setOutputs(trans_output_operand_ids.data(), trans_output_operand_ids.size());

    model->AddOperation(trans, nullptr);
}
void VsiOpCallbackInfoGemm::AddAddOp(const onnxruntime::Node* node,
                                     onnxruntime::ModelShellPtr& model,
                                     std::vector<uint32_t> add_operand_ids,
                                     std::string add_add_name) {
    auto add = std::make_shared<nnrt::op::AddOperation>();

    std::vector<uint32_t> add_in_operand_ids{add_operand_ids[0], add_operand_ids[1]};
    std::vector<uint32_t> add_output_operand_ids{add_operand_ids[2]};

    auto output_defs = node->OutputDefs();

    auto add_output_tensor_info = std::make_shared<VsiGraphTensorInfo>();
    add_output_tensor_info->name = output_defs[0]->Name() + add_add_name;
    add_output_tensor_info->is_initializer = true;
    model->GetGraphInputs().push_back(add_output_tensor_info);
    model->GetGraphOutputs().push_back(add_output_tensor_info);

    auto add_operand_a = model->GetOperand(add_operand_ids[0]);
    if (add_operand_a->ndim() == 0) {
        add_operand_a->type = nnrt::OperandType::TENSOR_FLOAT32;
        vsi_npu::SetTensorDims(*output_defs[0], add_operand_a->dimensions);
    }

    add->setInputs(add_in_operand_ids.data(), add_in_operand_ids.size());
    add->setOutputs(add_output_operand_ids.data(), add_output_operand_ids.size());

    model->AddOperation(add, nullptr);
}
void VsiOpCallbackInfoGemm::AddMatmulOp(const onnxruntime::Node* node,
                                        onnxruntime::ModelShellPtr& model,
                                        std::vector<uint32_t> matmul_operand_ids) {
    auto matmul = std::make_shared<nnrt::op::MatrixMulOperation>();

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    int64_t transA{false};
    auto status = vsi_npu::GetAttr<int64_t>(attrs, "transA", &transA).IsOK();
    if (status && transA == 1) {
        matmul->transpose[0] = true;
    } else {
        matmul->transpose[0] = false;
    }
    matmul->transpose[1] = false;

    std::vector<uint32_t> matmul_in_operand_ids{matmul_operand_ids[0], matmul_operand_ids[1]};
    std::vector<uint32_t> matmul_out_operand_ids{matmul_operand_ids[2]};

    matmul->setInputs(matmul_in_operand_ids.data(), matmul_in_operand_ids.size());
    matmul->setOutputs(matmul_out_operand_ids.data(), matmul_out_operand_ids.size());
    matmul->setVxParam(
        nnrt::OverflowPolicy::SATURATE, nnrt::RoundingPolicy::TO_ZERO, nnrt::Rounding::FLOOR);
    model->AddOperation(matmul, nullptr);
};

void VsiOpCallbackInfoGemm::Setup(const onnxruntime::Node* node,
                                  onnxruntime::ModelShellPtr& model,
                                  const onnxruntime::GraphViewer* graph_viewer) {
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    auto input_defs = node->InputDefs();
    auto output_defs = node->OutputDefs();

    const std::string matmul_add_name = "@@matmuladdTmp";
    const std::string trans_add_name = "@@transaddTmp";
    const std::string mul_a_input_add_name = "@@mulainputaddTmp";
    const std::string mul_a_output_add_name = "@@mulaoutputaddTmp";
    const std::string mul_b_input_add_name = "@@mulbinputaddTmp";
    const std::string mul_b_output_add_name = "@@mulboutputaddTmp";

    int64_t is_trans{0};
    bool is_scale{false};
    bool is_add{false};

    int64_t transB{0};
    vsi_npu::GetAttr<int64_t>(attrs, "transB", &transB).IsOK();
    if (transB == 1) is_trans = true;

    float alpha{1.0};
    float beta{1.0};
    vsi_npu::GetAttr<float>(attrs, "alpha", &alpha).IsOK();
    vsi_npu::GetAttr<float>(attrs, "beta", &beta).IsOK();
    if (alpha != 1.0 || beta != 1.0) is_scale = true;

    if (input_defs.size() == 3) is_add = true;

    uint32_t trans_input_operand_id{0};
    uint32_t trans_output_operand_id{0};
    uint32_t mul_a_input_operand_id_a{0};
    uint32_t mul_a_input_operand_id_b{0};
    uint32_t mul_b_input_operand_id_a{0};
    uint32_t mul_b_input_operand_id_b{0};
    uint32_t mul_a_output_operand_id{0};
    uint32_t mul_b_output_operand_id{0};
    uint32_t add_input_operand_id_a{0};
    uint32_t add_input_operand_id_b{0};
    uint32_t add_output_operand_id{0};
    uint32_t matmul_input_operand_id_a{0};
    uint32_t matmul_input_operand_id_b{0};
    uint32_t matmul_output_operand_id{0};

    if (is_trans && is_scale && is_add) {
        // TO DO
    } else if (!is_trans && is_scale && is_add) {
        matmul_input_operand_id_a = model->AddOperand(input_defs[0], graph_viewer);
        matmul_input_operand_id_b = model->AddOperand(input_defs[1], graph_viewer);
        matmul_output_operand_id = model->AddOperand(output_defs[0]->Name() + matmul_add_name);
        mul_a_input_operand_id_a = matmul_output_operand_id;
        mul_a_input_operand_id_b = model->AddOperand(input_defs[0]->Name() + mul_a_input_add_name);
        mul_a_output_operand_id = model->AddOperand(output_defs[0]->Name() + mul_a_output_add_name);
        mul_b_input_operand_id_a = model->AddOperand(input_defs[2], graph_viewer);
        mul_b_input_operand_id_b = model->AddOperand(input_defs[2]->Name() + mul_b_input_add_name);
        mul_b_output_operand_id = model->AddOperand(output_defs[0]->Name() + mul_b_output_add_name);
        add_input_operand_id_a = mul_a_output_operand_id;
        add_input_operand_id_b = mul_b_output_operand_id;
        add_output_operand_id = model->AddOperand(output_defs[0], graph_viewer);

        std::vector<uint32_t> matmul_operand_ids{
            matmul_input_operand_id_a, matmul_input_operand_id_b, matmul_output_operand_id};
        std::vector<uint32_t> add_operand_ids{
            add_input_operand_id_a, add_input_operand_id_b, add_output_operand_id};
        std::vector<uint32_t> mul_a_operand_ids{
            mul_a_input_operand_id_a, mul_a_input_operand_id_b, mul_a_output_operand_id};
        std::vector<uint32_t> mul_b_operand_ids{
            mul_b_input_operand_id_a, mul_b_input_operand_id_b, mul_b_output_operand_id};
        std::vector<std::string> mul_add_names{mul_a_input_add_name,
                                               mul_a_output_add_name,
                                               mul_b_input_add_name,
                                               mul_b_output_add_name};

        AddMatmulOp(node, model, matmul_operand_ids);
        AddMulOp(node, model, mul_a_operand_ids, mul_add_names, 0);
        AddMulOp(node, model, mul_b_operand_ids, mul_add_names, 1);
        AddAddOp(node, model, add_operand_ids, matmul_add_name);
    } else if (is_trans && !is_scale && is_add) {
        trans_input_operand_id = model->AddOperand(input_defs[1], graph_viewer);
        trans_output_operand_id = model->AddOperand(input_defs[1]->Name() + trans_add_name);
        matmul_input_operand_id_a = model->AddOperand(input_defs[0], graph_viewer);
        matmul_input_operand_id_b = trans_output_operand_id;
        matmul_output_operand_id = model->AddOperand(output_defs[0]->Name() + matmul_add_name);
        add_input_operand_id_a = matmul_output_operand_id;
        add_input_operand_id_b = model->AddOperand(input_defs[2], graph_viewer);
        add_output_operand_id = model->AddOperand(output_defs[0], graph_viewer);

        std::vector<uint32_t> matmul_operand_ids{
            matmul_input_operand_id_a, matmul_input_operand_id_b, matmul_output_operand_id};
        std::vector<uint32_t> add_operand_ids{
            add_input_operand_id_a, add_input_operand_id_b, add_output_operand_id};
        std::vector<uint32_t> trans_operand_ids{trans_input_operand_id, trans_output_operand_id};

        AddTransposeOp(node, model, trans_operand_ids, trans_add_name);
        AddMatmulOp(node, model, matmul_operand_ids);
        AddAddOp(node, model, add_operand_ids, matmul_add_name);
    } else if (is_trans && is_scale && !is_add) {
        // TO DO
    } else if (!is_trans && !is_scale && is_add) {
        matmul_input_operand_id_a = model->AddOperand(input_defs[0], graph_viewer);
        matmul_input_operand_id_b = model->AddOperand(input_defs[1], graph_viewer);
        matmul_output_operand_id = model->AddOperand(output_defs[0]->Name() + matmul_add_name);
        add_input_operand_id_a = matmul_output_operand_id;
        add_input_operand_id_b = model->AddOperand(input_defs[2], graph_viewer);
        add_output_operand_id = model->AddOperand(output_defs[0], graph_viewer);

        std::vector<uint32_t> matmul_operand_ids{
            matmul_input_operand_id_a, matmul_input_operand_id_b, matmul_output_operand_id};
        std::vector<uint32_t> add_operand_ids{
            add_input_operand_id_a, add_input_operand_id_b, add_output_operand_id};

        AddMatmulOp(node, model, matmul_operand_ids);
        AddAddOp(node, model, add_operand_ids, matmul_add_name);
    } else if (!is_trans && is_scale && !is_add) {
        // TO DO
    } else if (is_trans && !is_scale && !is_add) {
        // TO DO
    } else if (!is_trans && !is_scale && !is_add) {
        matmul_input_operand_id_a = model->AddOperand(input_defs[0], graph_viewer);
        matmul_input_operand_id_b = model->AddOperand(input_defs[1], graph_viewer);
        matmul_output_operand_id = model->AddOperand(output_defs[0], graph_viewer);

        std::vector<uint32_t> matmul_operand_ids{
            matmul_input_operand_id_a, matmul_input_operand_id_b, matmul_output_operand_id};

        AddMatmulOp(node, model, matmul_operand_ids);
    }
}

bool VsiOpCallbackInfoGemm::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                            const Node* node,
                                            std::string& reason) {
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}

void VsiOpCallbackInfoLRN::SetupAttribute(nnrt::op::OperationPtr op,
                                          const Node* node,
                                          ModelShellPtr& model,
                                          const onnxruntime::GraphViewer* graph_viewer) {
    auto lrn = std::dynamic_pointer_cast<nnrt::op::LocalResponseNormOperation>(op);
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    int32_t size;
    bool status = vsi_npu::GetAttr<int32_t>(attrs, "size", &size).IsOK();
    ORT_ENFORCE(status);

    lrn->radius = (size - 1) / 2;

    float bias;
    status = vsi_npu::GetAttr<float>(attrs, "bias", &bias).IsOK();
    ORT_ENFORCE(status);
    lrn->bias = bias;

    float alpha;
    status = vsi_npu::GetAttr<float>(attrs, "alpha", &alpha).IsOK();
    ORT_ENFORCE(status);
    lrn->scale = alpha / size;

    float beta;
    status = vsi_npu::GetAttr<float>(attrs, "beta", &beta).IsOK();
    ORT_ENFORCE(status);
    lrn->exponent = beta;

    lrn->axis = 1;
}

void VsiOpCallbackInfoLeakyRelu::SetupAttribute(nnrt::op::OperationPtr op,
                                                const Node* node,
                                                ModelShellPtr& model,
                                                const onnxruntime::GraphViewer* graph_viewer) {
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    float alpha;
    bool status = vsi_npu::GetAttr<float>(attrs, "alpha", &alpha).IsOK();
    ORT_ENFORCE(status);
    auto leakyrelu = std::dynamic_pointer_cast<nnrt::op::LeakyReluOperation>(op);
    leakyrelu->ratio = alpha;
}

void VsiOpCallbackInfoUpsample::Setup(const Node* node,
                                      ModelShellPtr& model,
                                      const onnxruntime::GraphViewer* graph_viewer) {
    auto input_defs = node->InputDefs();
    std::vector<uint32_t> in_operand_ids;
    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    auto output_defs = node->OutputDefs();
    std::vector<uint32_t> out_operand_ids;
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::string mode;
    bool status = vsi_npu::GetAttr<std::string>(attrs, "mode", &mode).IsOK();
    ORT_ENFORCE(status);

    auto compute_info = std::make_shared<VsiComputeInfo>();
    std::vector<float> scales;
    if (input_defs.size() > 1) {
        model->GetInitializerAsParameters<float>(input_defs[1], graph_viewer, scales);
        if (scales.size() == 0) {
            const int32_t kIndexScale = 1;
            std::vector<int32_t> attrs_index({kIndexScale});
            compute_info->backup_names.push_back(mode);
            model->CollectComputeInfo(node, graph_viewer, attrs_index, compute_info);
        }
    } else {
        bool status = vsi_npu::GetAttrs<float>(attrs, "scales", scales, false).IsOK();
        ORT_ENFORCE(status);
    }

    int32_t outputHeight = 0;
    int32_t outputWidth = 0;
    auto shape = vsi_npu::GetTensorShape(*(input_defs[0]));
    const std::vector<int64_t>& dims = shape.GetDims();
    if (scales.size() == 4) {
        outputHeight = dims[2] * scales[2];
        outputWidth = dims[3] * scales[3];
    } else if (scales.size() == 2) {
        outputHeight = dims[2] * scales[0];
        outputWidth = dims[3] * scales[1];
    } else {
        outputHeight = dims[2];
        outputWidth = dims[3];
    }

    if (mode == "nearest") {
        auto op = std::make_shared<nnrt::op::ResizeNearestNeighborOperation>();
        op->setInputs(in_operand_ids.data(), in_operand_ids.size());
        op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

        op->outputHeight = outputHeight;
        op->outputWidth = outputWidth;
        model->AddOperation(op, nullptr);
        compute_info->op = op;
    } else if (mode == "linear") {
        auto op = std::make_shared<nnrt::op::ResizeBilinearOperation>();
        op->setInputs(in_operand_ids.data(), in_operand_ids.size());
        op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

        op->outputHeight = outputHeight;
        op->outputWidth = outputWidth;
        model->AddOperation(op, nullptr);
        compute_info->op = op;
    }

    const auto* type_proto = input_defs[0]->TypeAsProto();
    if (type_proto->tensor_type().elem_type() ==
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
        auto tensor = model->GetModelPtr()->operand(input_operand_id);
        tensor->quant.scalar.zeroPoint = 0;
        tensor->quant.scalar.scale = 1.0f;
    }
}

Status VsiOpCallbackInfoUpsample::Compute(FunctionState state,
                                          const OrtApi* api,
                                          OrtKernelContext* context,
                                          NodeIndex node_index) {
    Ort::CustomOpApi ort{*api};
    ModelShell* model = reinterpret_cast<ModelShell*>(state);
    auto compute_info = model->GetComputeInfo(node_index);
    if (compute_info == nullptr) return Status::OK();

    auto attributes_input_ids = model->GetComputeInputIds(compute_info->compute_input_names);

    const OrtValue* input_tensor_scale =
        ort.KernelContext_GetInput(context, attributes_input_ids[0]);
    const auto input_tensor_scale_value = (float*)ort.GetTensorData<void>(input_tensor_scale);

    std::string mode = compute_info->backup_names[0];
    auto op = compute_info->op;
    if (mode == "nearest") {
        auto upsample = std::dynamic_pointer_cast<nnrt::op::ResizeNearestNeighborOperation>(op);
        upsample->outputHeight *= input_tensor_scale_value[2];
        upsample->outputWidth *= input_tensor_scale_value[3];
    } else if (mode == "linear") {
        auto upsample = std::dynamic_pointer_cast<nnrt::op::ResizeBilinearOperation>(op);
        upsample->outputHeight *= input_tensor_scale_value[2];
        upsample->outputWidth *= input_tensor_scale_value[3];
    }
    return Status::OK();
}

bool VsiOpCallbackInfoUpsample::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                                const Node* node,
                                                std::string& reason) {
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    std::vector<float> scales;
    vsi_npu::GetAttrs<float>(attrs, "scales", scales, false).IsOK();

    if (scales.size() != 4 && scales.size() != 2 && scales.size() != 0) return false;
    if (scales.size() == 4 && (scales[0] != 1 || scales[1] != 1)) return false;
    return vsi_npu::CheckMainInputType(node, reason) && vsi_npu::CheckAllZeroDim(node, reason);
}

void VsiOpCallbackInfoInstanceNormalization::Setup(const onnxruntime::Node* node,
                                                   onnxruntime::ModelShellPtr& model,
                                                   const onnxruntime::GraphViewer* graph_viewer) {
    auto input_defs = node->InputDefs();
    std::vector<uint32_t> in_operand_ids;

    auto output_defs = node->OutputDefs();
    std::vector<uint32_t> out_operand_ids;

    const int32_t kIndexScale = 2;
    const int32_t kIndexB = 1;
    std::vector<int32_t> compute_input_index({kIndexScale, kIndexB});

    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() == 4) {
        uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
        in_operand_ids.push_back(input_operand_id);

        uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
        out_operand_ids.push_back(output_operand_id);
    } else if (shape.NumDimensions() == 3) {
        // reshape input
        uint32_t input_pre_reshape_operand_id = model->AddOperand(input_defs[0], graph_viewer);
        uint32_t output_pre_reshape_operand_id =
            model->AddOperand(input_defs[0]->Name() + "@@reshape_0");
        auto operand_pre_reshape = model->GetOperand(output_pre_reshape_operand_id);
        operand_pre_reshape->type = vsi_npu::convertToOperandType(input_defs[0]->Type());
        vsi_npu::SetTensorDims(*input_defs[0], operand_pre_reshape->dimensions);
        operand_pre_reshape->dimensions.push_back(1);

        std::vector<uint32_t> in_operand_ids_pre_reshape;
        std::vector<uint32_t> out_operand_ids_pre_reshape;
        in_operand_ids_pre_reshape.push_back(input_pre_reshape_operand_id);
        out_operand_ids_pre_reshape.push_back(output_pre_reshape_operand_id);

        auto pre_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        pre_reshape->setInputs(in_operand_ids_pre_reshape.data(),
                               in_operand_ids_pre_reshape.size());
        pre_reshape->setOutputs(out_operand_ids_pre_reshape.data(),
                                out_operand_ids_pre_reshape.size());

        for (auto dim : operand_pre_reshape->dimensions) {
            pre_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(pre_reshape, nullptr);

        in_operand_ids.push_back(output_pre_reshape_operand_id);

        // reshape output
        auto output_post_reshape_operand_id = model->AddOperand(output_defs[0], graph_viewer);
        auto input_post_reshape_operand_id =
            model->AddOperand(output_defs[0]->Name() + "@@reshape_1");
        auto operand_post_reshape = model->GetOperand(input_post_reshape_operand_id);
        operand_post_reshape->type = vsi_npu::convertToOperandType(output_defs[0]->Type());
        vsi_npu::SetTensorDims(*output_defs[0], operand_post_reshape->dimensions);
        operand_post_reshape->dimensions.push_back(1);

        std::vector<uint32_t> in_operand_ids_post_reshape;
        std::vector<uint32_t> out_operand_ids_post_reshape;
        in_operand_ids_post_reshape.push_back(input_post_reshape_operand_id);
        out_operand_ids_post_reshape.push_back(output_post_reshape_operand_id);

        auto post_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        post_reshape->setInputs(in_operand_ids_post_reshape.data(),
                                in_operand_ids_post_reshape.size());
        post_reshape->setOutputs(out_operand_ids_post_reshape.data(),
                                 out_operand_ids_post_reshape.size());

        for (auto dim : operand_post_reshape->dimensions) {
            post_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(post_reshape, nullptr);

        out_operand_ids.push_back(input_post_reshape_operand_id);
    } else if (shape.NumDimensions() > 4) {
        // reshape input
        uint32_t input_pre_reshape_operand_id = model->AddOperand(input_defs[0], graph_viewer);
        uint32_t output_pre_reshape_operand_id =
            model->AddOperand(input_defs[0]->Name() + "@@reshape_0");
        auto operand_pre_reshape = model->GetOperand(output_pre_reshape_operand_id);
        operand_pre_reshape->type = vsi_npu::convertToOperandType(input_defs[0]->Type());
        const std::vector<int64_t>& dims_pre_reshape = shape.GetDims();
        for (size_t i = 0; i < 3; i++) {
            operand_pre_reshape->dimensions.push_back(static_cast<uint32_t>(dims_pre_reshape[i]));
        }
        uint32_t val = 1;
        for (size_t i = 3; i < dims_pre_reshape.size(); i++) {
            val *= static_cast<uint32_t>(dims_pre_reshape[i]);
        }
        operand_pre_reshape->dimensions.push_back(val);

        std::vector<uint32_t> in_operand_ids_pre_reshape;
        std::vector<uint32_t> out_operand_ids_pre_reshape;
        in_operand_ids_pre_reshape.push_back(input_pre_reshape_operand_id);
        out_operand_ids_pre_reshape.push_back(output_pre_reshape_operand_id);

        auto pre_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        pre_reshape->setInputs(in_operand_ids_pre_reshape.data(),
                               in_operand_ids_pre_reshape.size());
        pre_reshape->setOutputs(out_operand_ids_pre_reshape.data(),
                                out_operand_ids_pre_reshape.size());

        for (auto dim : operand_pre_reshape->dimensions) {
            pre_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(pre_reshape, nullptr);

        in_operand_ids.push_back(output_pre_reshape_operand_id);

        // reshape output
        auto output_post_reshape_operand_id = model->AddOperand(output_defs[0], graph_viewer);
        auto input_post_reshape_operand_id =
            model->AddOperand(output_defs[0]->Name() + "@@reshape_1");
        auto operand_post_reshape = model->GetOperand(input_post_reshape_operand_id);
        operand_post_reshape->type = vsi_npu::convertToOperandType(output_defs[0]->Type());
        shape = vsi_npu::GetTensorShape(*output_defs[0]);
        const std::vector<int64_t>& dims_post_reshape = shape.GetDims();
        for (size_t i = 0; i < 3; i++) {
            operand_post_reshape->dimensions.push_back(static_cast<uint32_t>(dims_post_reshape[i]));
        }
        val = 1;
        for (size_t i = 3; i < dims_post_reshape.size(); i++) {
            val *= static_cast<uint32_t>(dims_post_reshape[i]);
        }
        operand_post_reshape->dimensions.push_back(val);

        std::vector<uint32_t> in_operand_ids_post_reshape;
        std::vector<uint32_t> out_operand_ids_post_reshape;
        in_operand_ids_post_reshape.push_back(input_post_reshape_operand_id);
        out_operand_ids_post_reshape.push_back(output_post_reshape_operand_id);

        auto post_reshape = std::make_shared<nnrt::op::ReshapeOperation>();
        post_reshape->setInputs(in_operand_ids_post_reshape.data(),
                                in_operand_ids_post_reshape.size());
        post_reshape->setOutputs(out_operand_ids_post_reshape.data(),
                                 out_operand_ids_post_reshape.size());

        for (auto dim : operand_post_reshape->dimensions) {
            post_reshape->shape.push_back(static_cast<int32_t>(dim));
        }

        model->AddOperation(post_reshape, nullptr);

        out_operand_ids.push_back(input_post_reshape_operand_id);
    }

    for (auto index : compute_input_index) {
        uint32_t input_operand_id = model->AddOperand(input_defs[index], graph_viewer);
        in_operand_ids.push_back(input_operand_id);
    }

    auto op = std::make_shared<nnrt::op::InstanceNormOperation<float>>();
    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    float epsilon;
    bool status = vsi_npu::GetAttr<float>(attrs, "epsilon", &epsilon).IsOK();
    ORT_ENFORCE(status);
    op->eps = epsilon;

    model->AddOperation(op, nullptr);

    auto compute_info = std::make_shared<VsiComputeInfo>();
    model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
}

Status VsiOpCallbackInfoInstanceNormalization::Compute(FunctionState state,
                                                       const OrtApi* api,
                                                       OrtKernelContext* context,
                                                       NodeIndex node_index) {
    ConstInputOprands(state, api, context, node_index);
    return Status::OK();
}

void VsiOpCallbackInfoBatchNormalization::Setup(const onnxruntime::Node* node,
                                                onnxruntime::ModelShellPtr& model,
                                                const onnxruntime::GraphViewer* graph_viewer) {
    auto input_defs = node->InputDefs();
    std::vector<uint32_t> in_operand_ids;
    const int32_t kIndexScale = 3;
    const int32_t kIndexB = 4;
    const int32_t kIndexMean = 1;
    const int32_t kIndexVar = 2;
    std::vector<int32_t> compute_input_index({kIndexScale, kIndexB, kIndexMean, kIndexVar});
    uint32_t input_operand_id = model->AddOperand(input_defs[0], graph_viewer);
    in_operand_ids.push_back(input_operand_id);

    for (auto index : compute_input_index) {
        uint32_t input_operand_id = model->AddOperand(input_defs[index], graph_viewer);
        in_operand_ids.push_back(input_operand_id);
    }

    auto output_defs = node->OutputDefs();
    std::vector<uint32_t> out_operand_ids;
    uint32_t output_operand_id = model->AddOperand(output_defs[0], graph_viewer);
    out_operand_ids.push_back(output_operand_id);

    auto op = std::make_shared<nnrt::op::BatchNormalization>();
    op->setInputs(in_operand_ids.data(), in_operand_ids.size());
    op->setOutputs(out_operand_ids.data(), out_operand_ids.size());

    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    float epsilon;
    bool status = vsi_npu::GetAttr<float>(attrs, "epsilon", &epsilon).IsOK();
    ORT_ENFORCE(status);
    op->eps = epsilon;

    model->AddOperation(op, nullptr);

    auto compute_info = std::make_shared<VsiComputeInfo>();
    model->CollectComputeInfo(node, graph_viewer, compute_input_index, compute_info);
}

Status VsiOpCallbackInfoBatchNormalization::Compute(FunctionState state,
                                                    const OrtApi* api,
                                                    OrtKernelContext* context,
                                                    NodeIndex node_index) {
    ConstInputOprands(state, api, context, node_index);
    return Status::OK();
}

bool VsiOpCallbackInfoBatchNormalization::IsNodeSupported(
    const onnxruntime::GraphViewer& graph_viewer, const Node* node, std::string& reason) {
    auto input_defs = node->InputDefs();
    auto shape = vsi_npu::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() != 4) {
        reason += "## Only support BN2D now.";
        return false;
    }
    return VsiOpCallbackInfo::IsNodeSupported(graph_viewer, node, reason);
}

#define REGISTER_OP(name)                               \
    std::pair<std::string, std::shared_ptr<VsiOpInfo>>( \
        #name, std::dynamic_pointer_cast<VsiOpInfo>(std::make_shared<VsiOpInfo##name>()))

std::map<std::string, std::shared_ptr<VsiOpInfo>> vsi_npu_supported_ops = {
    REGISTER_OP(Relu),
    REGISTER_OP(Abs),
    REGISTER_OP(Add),
    REGISTER_OP(Sub),
    REGISTER_OP(Mul),
    REGISTER_OP(Div),
    REGISTER_OP(Sum),
    REGISTER_OP(Conv),
    REGISTER_OP(Concat),
    REGISTER_OP(MaxPool),
    REGISTER_OP(AveragePool),
    REGISTER_OP(GlobalMaxPool),
    REGISTER_OP(GlobalAveragePool),
    REGISTER_OP(Softmax),
    REGISTER_OP(Reshape),
    REGISTER_OP(Gemm),
    REGISTER_OP(Transpose),
    REGISTER_OP(LRN),
    REGISTER_OP(DequantizeLinear),
    REGISTER_OP(QuantizeLinear),
    REGISTER_OP(LeakyRelu),
    REGISTER_OP(Upsample),
    REGISTER_OP(InstanceNormalization),
    REGISTER_OP(Pad),
    REGISTER_OP(BatchNormalization),
    REGISTER_OP(ConvInteger),
    REGISTER_OP(MatMul),
};

bool VsiSupported(const std::string& opName) {
    return vsi_npu_supported_ops.find(opName) != vsi_npu_supported_ops.end();
}

std::shared_ptr<VsiOpInfo> getVsiFunc(const std::string& opName) {
    return vsi_npu_supported_ops[opName];
}
}  // namespace onnxruntime
