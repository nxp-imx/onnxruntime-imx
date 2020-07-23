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

void VsiOpCallbackInfoGemm::SetupAttribute(nnrt::op::OperationPtr op,
                                           const Node* node,
                                           ModelShellPtr& model,
                                           const onnxruntime::GraphViewer* graph_viewer) {
    op->setVxParam(
        nnrt::OverflowPolicy::SATURATE, nnrt::RoundingPolicy::TO_ZERO, nnrt::Rounding::FLOOR);
}

bool VsiOpCallbackInfoGemm::IsNodeSupported(const onnxruntime::GraphViewer& graph_viewer,
                                            const Node* node,
                                            std::string& reason) {
    ProtoHelperNodeContext ctx(*node);
    OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

    int64_t transA;
    bool status = vsi_npu::GetAttr<int64_t>(attrs, "transA", &transA).IsOK();
    if (status && transA == 1) return false;

    int64_t transB;
    status = vsi_npu::GetAttr<int64_t>(attrs, "transB", &transB).IsOK();
    if (status && transB == 1) return false;

    float alpha;
    status = vsi_npu::GetAttr<float>(attrs, "alpha", &alpha).IsOK();
    if (status && alpha != 1.0) return false;

    float beta;
    status = vsi_npu::GetAttr<float>(attrs, "beta", &beta).IsOK();
    if (status && beta != 1.0) return false;

    auto input_defs = node->InputDefs();
    if (input_defs.size() == 3) {
        auto input_c_shape = vsi_npu::GetTensorShape(*input_defs[2]);
        auto output_defs = node->OutputDefs();
        auto output_shape = vsi_npu::GetTensorShape(*output_defs[0]);
        if (input_c_shape != output_shape) {
            reason += "## add input and output should have same shape.";
            return false;
        }
    }

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
        const std::vector<int64_t>& dims_pre_reshape = shape.GetDims();
        for (auto dim : dims_pre_reshape) {
            operand_pre_reshape->dimensions.push_back(static_cast<uint32_t>(dim));
        }
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
        shape = vsi_npu::GetTensorShape(*output_defs[0]);
        const std::vector<int64_t>& dims_post_reshape = shape.GetDims();
        for (auto dim : dims_post_reshape) {
            operand_post_reshape->dimensions.push_back(static_cast<uint32_t>(dim));
        }
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
};

bool VsiSupported(const std::string& opName) {
    return vsi_npu_supported_ops.find(opName) != vsi_npu_supported_ops.end();
}

std::shared_ptr<VsiOpInfo> getVsiFunc(const std::string& opName) {
    return vsi_npu_supported_ops[opName];
}
}  // namespace onnxruntime
