import onnx
import numpy as np
from onnx import helper, numpy_helper

def prepare_qat_model(onnx_model_path, output_model_path, bit_size : int = 8,):

    # Load model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Embed a constant tensor for bit_size
    bit_init = numpy_helper.from_array(
        np.array([bit_size], dtype=np.int8),
        name='__quant_bits'
    )
    graph.initializer.append(bit_init)

    nodes_to_add = []
    for node in list(graph.node):
        # Insert QuantDequant nodes after ReLU outputs
        if node.op_type == 'Relu':
            orig_out = node.output[0]
            qdq_out  = f"{node.name}_qdq_out"
            qdq_node = helper.make_node(
                'QuantDequant',
                inputs=[orig_out, '__quant_bits'],
                outputs=[qdq_out],
                domain='ai.onnx.contrib',
                name=f"{node.name}_QuantDequant"
            )
            # Reroute downstream inputs to the QuantDequant output
            for consumer in graph.node:
                for idx, inp in enumerate(consumer.input):
                    if inp == orig_out:
                        consumer.input[idx] = qdq_out
            nodes_to_add.append(qdq_node)

        # Insert QuantDequant nodes before MatMul weights
        if node.op_type == 'MatMul':
            w_in = node.input[1]
            qdq_w_out  = f"{w_in}_qdq_w"
            qdq_w_node = helper.make_node(
                'QuantDequant',
                inputs=[w_in, '__quant_bits'],
                outputs=[qdq_w_out],
                domain='ai.onnx.contrib',
                name=f"{w_in}_QuantDequant"
            )
            # Swap QuantDequant output with the new weight input
            node.input[1] = qdq_w_out
            nodes_to_add.append(qdq_w_node)
    
    # Append new nodes in topological order
    graph.node.extend(nodes_to_add)

    # Save qat model
    model_def = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    
    onnx.save(model_def, output_model_path)
    print(f"QAT model saved to {output_model_path}")

if __name__ == '__main__':
    onnx_model_path = "models/model.onnx"
    output_model_path = "models/qat_model.onnx"
    prepare_qat_model(onnx_model_path, output_model_path)