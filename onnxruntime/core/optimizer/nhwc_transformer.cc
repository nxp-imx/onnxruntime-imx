// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_ACL) || defined(USE_ARMNN)
#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"

#include "core/providers/acl/acl_common.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEPermute.h"


using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

class NhwcTransformerImpl {
 public:
  NhwcTransformerImpl(Graph& graph) noexcept : graph_(graph) {}
  void Transform(Node& node);
  void Finalize(bool& modified);

 private:
  void InsertPermuteNhwcNode(Node& node, const Node* parentNode);
  void InsertPermuteNchwNode(Node& node, const Node* childNode);
  long unsigned int ReplaceNode(Node& node);
  void PermuteWeights(NodeArg* src, NodeArg** dst, const std::string&);
  bool RequiresWeightsPermutation(const Node& node);

  Graph& graph_;
};

bool NhwcTransformerImpl::RequiresWeightsPermutation(const Node& node) {
   return ((node.GetExecutionProviderType() == kAclExecutionProvider || 
            node.GetExecutionProviderType() == kArmNNExecutionProvider) &&
           (node.OpType() == "Conv" ||
            node.OpType() == "FusedConv"));
} 

void NhwcTransformerImpl::InsertPermuteNhwcNode(__attribute__ ((unused)) Node& node, __attribute__ ((unused)) const Node* parent_node) {

  auto& input_defs = node.MutableInputDefs();
  std::string new_input_def_name = graph_.GenerateNodeArgName("input");
  auto* new_input_arg = &graph_.GetOrCreateNodeArg(new_input_def_name, input_defs[0]->TypeAsProto());
 
  Node& permute_node = graph_.AddNode(graph_.GenerateNodeName("PermuteNHCW"), 
                                     "ReorderInput",
                                     "ReorderInput",
                                     {input_defs[0]},
                                     {new_input_arg},
                                     nullptr,
                                     kMSNhwcDomain);
  permute_node.SetExecutionProviderType(node.GetExecutionProviderType());

  if (parent_node) {
    graph_.RemoveEdge(parent_node->Index(), node.Index(), 0, 0);
  }
  input_defs[0] = new_input_arg;

  graph_.AddEdge(permute_node.Index(), node.Index(), 0, 0);
  if (parent_node) {
    graph_.AddEdge(parent_node->Index(), permute_node.Index() , 0, 0);
  }

}

long unsigned int NhwcTransformerImpl::ReplaceNode(Node& node) {

  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  Node& newNode = graph_.AddNode(graph_.GenerateNodeName(node.Name()), 
                                     node.OpType(),
                                     node.OpType(),
                                     input_defs,
                                     output_defs,
                                     &node.GetAttributes(),
                                     kMSNhwcDomain);
  newNode.SetExecutionProviderType(node.GetExecutionProviderType());

  // Permute weights
  if (RequiresWeightsPermutation(node)) {
    PermuteWeights(input_defs[1], &newNode.MutableInputDefs()[1], node.GetExecutionProviderType());
  }
  
  const std::vector<std::reference_wrapper<Node>> replacedNode({node});
  graph_utils::FinalizeNodeFusion(graph_, replacedNode, newNode);

  return newNode.Index();
}

void NhwcTransformerImpl::PermuteWeights(NodeArg *input_def, NodeArg** nhwc_conv_W_arg, const std::string& execution_provider) {

  // Require that the weights tensor be static.
  const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph_, *input_def) ||
      !graph_.GetInitializedTensor(input_def->Name(), conv_W_tensor_proto) ||
      (conv_W_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
      (conv_W_tensor_proto->dims_size() != 4) ||
      (execution_provider == kArmNNExecutionProvider && conv_W_tensor_proto->dims(1) == 1)) {
    return;
  }

  auto conv_W = onnxruntime::make_unique<Initializer>(*conv_W_tensor_proto);

  arm_compute::Tensor weights;
  arm_compute::Tensor new_weights;

  arm_compute::TensorShape initial_shape(conv_W->dims()[3], conv_W->dims()[2], conv_W->dims()[1], conv_W->dims()[0]);

  weights.allocator()->init(arm_compute::TensorInfo(initial_shape, arm_compute::Format::F32));

  arm_compute::NEPermute permutationLayer;
  permutationLayer.configure(&weights, &new_weights,
    (conv_W_tensor_proto->dims(1) == 1) ? arm_compute::PermutationVector(3,2,0,1) : arm_compute::PermutationVector(2,0,1));

  weights.allocator()->import_memory(conv_W->data<float>());


  new_weights.allocator()->allocate();

  permutationLayer.run();

  weights.allocator()->free();

  float* reordered_filter = reinterpret_cast<float*>(new_weights.buffer());

  ONNX_NAMESPACE::TensorProto nhwc_conv_W_tensor_proto;

  nhwc_conv_W_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  nhwc_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName("nhwc_conv_W_arg"));
  nhwc_conv_W_tensor_proto.set_raw_data(reordered_filter, new_weights.info()->tensor_shape().total_size() * sizeof(float));

  for (size_t i = 0; i < 4; i++) {
    nhwc_conv_W_tensor_proto.add_dims(conv_W->dims()[i]);
  }

  new_weights.allocator()->free();

  graph_.AddInitializedTensor(nhwc_conv_W_tensor_proto);

  *nhwc_conv_W_arg = &graph_.GetOrCreateNodeArg(nhwc_conv_W_tensor_proto.name(), nullptr);
}

void NhwcTransformerImpl::InsertPermuteNchwNode(__attribute__ ((unused)) Node& node, __attribute__ ((unused)) const Node* childNode) {

  auto& output_defs = node.MutableOutputDefs();

  //ToDo multiple outputs
  std::string new_output_def_name = graph_.GenerateNodeArgName("output");
  auto* new_output_arg = &graph_.GetOrCreateNodeArg(new_output_def_name, output_defs[0]->TypeAsProto());

  Node& permute_node = graph_.AddNode(graph_.GenerateNodeName("PermuteNCHW"), 
                                     "ReorderOutput",
                                     "ReorderOutput",
                                     {new_output_arg},
                                     output_defs,
                                     nullptr,
                                     kMSNhwcDomain);
  permute_node.SetExecutionProviderType(node.GetExecutionProviderType());

  if (childNode) {
    graph_.RemoveEdge(node.Index(), childNode->Index(), 0, 0);
  }
  output_defs[0] = new_output_arg;

  //ToDo multiple outputs
  graph_.AddEdge(node.Index(), permute_node.Index(), 0, 0);
  if (childNode) {
    graph_.AddEdge(permute_node.Index(), childNode->Index(), 0, 0);
  }  
}

bool Requires_NHWC_Layout(const Node& node, __attribute__ ((unused)) bool display = true) {
//ToDo: check FuseConv
  bool requires = ((node.GetExecutionProviderType() == kAclExecutionProvider || node.GetExecutionProviderType() == kArmNNExecutionProvider) &&
    (node.OpType() == "Conv" ||
     node.OpType() == "FusedConv" ||
     node.OpType() == "MaxPool" ||
     node.OpType() == "AveragePool" ||
     node.OpType() == "GlobalMaxPool" ||
     node.OpType() == "GlobalAveragePool"));

  return requires;
}

bool SupportsBothLayouts(const Node& node, __attribute__ ((unused)) bool display = true) {
  bool requires = ((node.GetExecutionProviderType() == kAclExecutionProvider || node.GetExecutionProviderType() == kArmNNExecutionProvider) && !graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {7, 9})) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {6, 7}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Concat", {4, 11}) ||
          node.OpType() == "Clip" ||
          node.OpType() == "Elu" ||
          node.OpType() == "HardSigmoid" ||
          node.OpType() == "LeakyRelu" ||
          node.OpType() == "Selu" ||
          node.OpType() == "Sigmoid" ||
          node.OpType() == "Softplus" ||
          node.OpType() == "Softsign" ||
          node.OpType() == "Tanh" ||
          node.OpType() == "PRelu" ||
          node.OpType() == "RandomNormal" ||
          node.OpType() == "RandomUniform" ||
          node.OpType() == "RandomNormalLike" ||
          node.OpType() == "RandomUniformLike" ||
          node.OpType() == "Multinomial";

  return requires;
}

bool Requires_NCHW_Layout(const Node& node) {
  bool requires = !(Requires_NHWC_Layout(node, false) ||
                    SupportsBothLayouts(node, false) ||
                    graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {7, 9}) || 
                    graph_utils::IsSupportedOptypeVersionAndDomain(node, "ReorderInput", {1}) ||
                    graph_utils::IsSupportedOptypeVersionAndDomain(node, "BatchNormalization", {7, 9}));

  return requires;
}

void NhwcTransformerImpl::Transform(Node& node) {
  long unsigned int new_index = node.Index();

  bool requiresNHWC = Requires_NHWC_Layout(node);

  bool firstNode = true;
  if (requiresNHWC) {
    for (auto it = node.InputNodesBegin(); it != node.InputNodesEnd(); ++it) {
      firstNode = false;

      if (Requires_NCHW_Layout(*it))
         InsertPermuteNhwcNode(node, &(*it));
    }

    // Edges without nodes
    if (firstNode)
      InsertPermuteNhwcNode(node, NULL);
  }

  // Replace specific ops
  if (requiresNHWC)
    new_index = ReplaceNode(node);

  Node& newNode = *graph_.GetNode(new_index);

  bool lastNode = true;
  if (requiresNHWC) {
    for (auto it = newNode.OutputNodesBegin(); it != newNode.OutputNodesEnd(); ++it) {
      lastNode = false;
      // Skip unnnecessary permutations if the operation supports NHWC
      if (Requires_NCHW_Layout(*it))
        InsertPermuteNchwNode(newNode, &(*it));
    }

    if (lastNode)
      InsertPermuteNchwNode(newNode, NULL);
  }
}

void NhwcTransformerImpl::Finalize(__attribute__ ((unused)) bool& modified) {
}

Status NhwcTransformer::ApplyImpl(__attribute__ ((unused))Graph& graph, __attribute__ ((unused))bool& modified, __attribute__ ((unused))int graph_level, __attribute__ ((unused)) const logging::Logger& logger) const {

  NhwcTransformerImpl impl(graph);
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    Node* node = graph.GetNode(index);

    // check that node hasn't already been removed
    if (!node)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
    impl.Transform(*node);
  } 

  impl.Finalize(modified);

  modified = false;
  return Status::OK();
}

}  // namespace onnxruntime
#endif
