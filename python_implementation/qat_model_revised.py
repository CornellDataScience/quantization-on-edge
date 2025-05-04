import onnx
import numpy as np
from onnx import helper, numpy_helper

def prepare_qat_model(onnx_model_path, output_model_path, bit_size : int = 8,):
    # Load model
    model = onnx.load(onnx_model_path)
    graph = model.graph

    new_nodes = []

    nodes_to_add = []
    for i, node in enumerate(graph.node):
        # Insert QuantDequant nodes after ReLU outputs
        if node.op_type == 'Relu':
            orig_out = node.output[0]
            qdq_out  = f"{node.name}_qdq_out"
            qdq_node = helper.make_node(
                'QuantDequant',
                inputs=[orig_out],
                outputs=[qdq_out],
                domain='ai.onnx.contrib',
                name=f"{node.name}_QuantDequant"
            )

            input_index = graph.node[i + 1].input.index(orig_out)
            graph.node[i + 1].input[input_index] = qdq_out
            nodes_to_add.append(qdq_node)

            new_nodes.append(node)
            new_nodes.append(qdq_node)
        elif node.op_type == 'MatMul': # Insert QuantDequant nodes before MatMul weights
            w_in = node.input[1]
            qdq_w_out  = f"{w_in}_qdq_w"
            qdq_w_node = helper.make_node(
                'QuantDequant',
                inputs=[w_in],
                outputs=[qdq_w_out],
                domain='ai.onnx.contrib',
                name=f"{w_in}_QuantDequant"
            )
            # Swap QuantDequant output with the new weight input
            node.input[1] = qdq_w_out
            nodes_to_add.append(qdq_w_node)

            new_nodes.append(qdq_w_node)
            new_nodes.append(node)
        else:
            new_nodes.append(node)
    

    for node in list(graph.node):
        graph.node.remove(node)

    for node in new_nodes:
        graph.node.append(node)

    # Save qat model
    model_def = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13), helper.make_opsetid("ai.onnx.contrib", 1)])
    
    onnx.save(model_def, output_model_path)
    print(f"QAT model saved to {output_model_path}")

    print('** Quantized nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))

if __name__ == '__main__':
    onnx_model_path = "models/untrained_model.onnx"
    output_model_path = "models/qat_model.onnx"
    prepare_qat_model(onnx_model_path, output_model_path)