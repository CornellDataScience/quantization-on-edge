import onnx
import numpy as np
from onnx import helper, numpy_helper

def prepare_qat_model(onnx_model_path, output_model_path, bit_size : int = 8,):

    # Load model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Embed constant for range and zero-point
    bit_range = float((2 ** bit_size) - 1)
    denom_init = numpy_helper.from_array(
        np.array([bit_range], dtype=np.float32),
        name='__quant_range'
    )
    zp_init = numpy_helper.from_array(
        np.array([0], dtype=np.int8),
        name='__zero_point'
    )
    graph.initializer.extend([denom_init, zp_init])

    nodes_to_add = []
    for node in list(graph.node):
        # Insert QuantDequant nodes after ReLU outputs
        if node.op_type == 'Relu':
            act_out   = node.output[0]
            mn_name   = f"{node.name}_min"
            mx_name   = f"{node.name}_max"
            range_name= f"{node.name}_range"
            scale_name= f"{node.name}_scale"
            zp_name   = '__zero_point'
            qdq_out   = f"{node.name}_qdq_out"

            # Compute min, max, range, scale dynamically
            nodes_to_add += [
                helper.make_node('ReduceMin', [act_out], [mn_name], keepdims=0, name=f"{node.name}_ReduceMin"),
                helper.make_node('ReduceMax', [act_out], [mx_name], keepdims=0, name=f"{node.name}_ReduceMax"),
                helper.make_node('Sub', [mx_name, mn_name], [range_name],  name=f"{node.name}_Sub"),
                helper.make_node('Div', [range_name, '__quant_range'], [scale_name], name=f"{node.name}_Div"),
                helper.make_node(
                    'QuantDequant',
                    inputs=[act_out, scale_name, zp_name],
                    outputs=[qdq_out],
                    domain='qat',
                    name=f"{node.name}_QuantDequant"
                )
            ]

            # Reroute downstream inputs to the QuantDequant output
            for consumer in graph.node:
                for i, inp in enumerate(consumer.input):
                    if inp == act_out:
                        consumer.input[i] = qdq_out

        # Insert QuantDequant nodes before MatMul weights
        if node.op_type == 'MatMul':
            w_in      = node.input[1]
            mn_name   = f"{w_in}_min"
            mx_name   = f"{w_in}_max"
            range_name= f"{w_in}_range"
            scale_name= f"{w_in}_scale"
            zp_name   = '__zero_point'
            qdq_w_out = f"{w_in}_qdq_w"

            nodes_to_add += [
                helper.make_node('ReduceMin', [w_in], [mn_name], keepdims=0, name=f"{w_in}_ReduceMin"),
                helper.make_node('ReduceMax', [w_in], [mx_name], keepdims=0, name=f"{w_in}_ReduceMax"),
                helper.make_node('Sub', [mx_name, mn_name], [range_name],  name=f"{w_in}_Sub"),
                helper.make_node('Div', [range_name, '__quant_range'], [scale_name], name=f"{w_in}_Div"),
                helper.make_node(
                    'QuantDequant',
                    inputs=[w_in, scale_name, zp_name],
                    outputs=[qdq_w_out],
                    domain='qat',
                    name=f"{w_in}_QuantDequant"
                )
            ]
            # Swap QuantDequant output with the new weight input
            node.input[1] = qdq_w_out
    
    # Append new nodes in topological order
    graph.node.extend(nodes_to_add)

    # Save qat model
    model_def = helper.make_model(graph, opset_imports=[
            helper.make_opsetid('', 13), helper.make_opsetid('qat', 1)])
    
    onnx.save(model_def, output_model_path)
    print(f"QAT model saved to {output_model_path}")

if __name__ == '__main__':
    onnx_model_path = "models/model.onnx"
    output_model_path = "models/qat_model.onnx"
    prepare_qat_model(onnx_model_path, output_model_path)