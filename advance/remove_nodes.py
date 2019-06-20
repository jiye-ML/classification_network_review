import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import graph_pb2
import re
def remove_training_nodes(input_graph):
  """Prunes out nodes that aren't needed for inference.
  There are nodes like Identity and CheckNumerics that are only useful
  during training, and can be removed in graphs that will be used for
  nothing but inference. Here we identify and remove them, returning an
  equivalent graph. To be specific, CheckNumerics nodes are always removed, and
  Identity nodes that aren't involved in control edges are spliced out so that
  their input and outputs are directly connected.
  Args:
    input_graph: Model to analyze and prune.
  Returns:
    A list of nodes with the unnecessary ones removed.
  """

  types_to_remove = {"CheckNumerics": True}
  

  input_nodes = input_graph.node
  names_to_remove = {}
  for node in input_nodes:
    print(node.name)
    if node.op in types_to_remove or node.name.find('Aux')>=0:
      names_to_remove[node.name] = True

  nodes_after_removal = []
  for node in input_nodes:
    if node.name in names_to_remove:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      if input_name in names_to_remove:
        continue
      new_node.input.append(full_input_name)
    nodes_after_removal.append(new_node)

  types_to_splice = {"Identity": True}
  names_to_splice = {}
  for node in nodes_after_removal:
    if node.op in types_to_splice:
      # We don't want to remove nodes that have control edge inputs, because
      # they might be involved in subtle dependency issues that removing them
      # will jeopardize.
      has_control_edge = False
      for input_name in node.input:
#        print('input_name [%s]' % input_name)
        if re.match(r"^\^", input_name):
#          print('control edge:[%s]' % input_name)
          has_control_edge = True
      if not has_control_edge:
        names_to_splice[node.name] = node.input[0]

  nodes_after_splicing = []
  for node in nodes_after_removal:
    if node.name in names_to_splice:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      while input_name in names_to_splice:
        full_input_name = names_to_splice[input_name]
        input_name = re.sub(r"^\^", "", full_input_name)
      new_node.input.append(full_input_name)
    nodes_after_splicing.append(new_node)

  output_graph = graph_pb2.GraphDef()
  output_graph.node.extend(nodes_after_splicing)
  return output_graph

