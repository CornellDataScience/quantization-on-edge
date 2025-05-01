import onnx
import json
import numpy as np
from onnx import helper, numpy_helper

def prepare_qat_model(onnx_model_path, quantized_params_path, output_model_path):

    # Load model and quantized parameters
    model = onnx.load(onnx_model_path)
    graph = model.graph
    with open(quantized_params_path) as f:
        params = json.load(f)

    # Iterate through parameters, add scale and zero point
    new_initializers = []
    for tensor in graph.initializer:
        if tensor.name in params:
            if params[tensor.name]["to_quantize"]:

                quantized_weight = np.array(params[tensor.name]["data"], dtype=np.int8)
                scale = np.array([params[tensor.name]["scale"]], dtype=np.float32)
                zero_point = np.array([params[tensor.name]["zero_point"]], dtype=np.int8)

                quantized_weight_initializer = numpy_helper.from_array(quantized_weight, tensor.name)
                quant_scale = numpy_helper.from_array(scale, tensor.name + "_scale")
                quant_zero_point = numpy_helper.from_array(zero_point, tensor.name + "_zero_point")

                tensor.CopyFrom(quantized_weight_initializer)

                new_initializers.extend([quant_scale, quant_zero_point])

                print(f"Added {tensor.name} scale={scale[0]} and zero_point={zero_point[0]}")
    graph.initializer.extend(new_initializers)

    nodes_to_add = []
    for node in list(graph.node):

        # Insert QuantDequant nodes after ReLU outputs
        if node.op_type == 'Relu':
            orig_out = node.output[0]
            scale_name = node.name + '_scale'
            zp_name    = node.name + '_zero_point'
            qdq_out    = node.name + '_quantdequant_out'
            qdq_node = helper.make_node(
                'QuantDequant',
                inputs=[orig_out, scale_name, zp_name],
                outputs=[qdq_out],
                domain = "ai.onnx.contrib"
            )
            # Reroute downstream inputs to the QuantDequant output
            for consumer in graph.node:
                for idx, inp in enumerate(consumer.input):
                    if inp == orig_out:
                        consumer.input[idx] = qdq_out
            nodes_to_add.append(qdq_node)
    
        # Insert QuantDequant nodes before MatMul weights
        if node.op_type == 'MatMul':
            weight_in  = node.input[1]
            scale_name = weight_in + '_scale'
            zp_name    = weight_in + '_zero_point'
            qdq_w_out  = weight_in + '_quantdequant_w'
            qdq_w = helper.make_node(
                'QuantDequant',
                inputs=[weight_in, scale_name, zp_name],
                outputs=[qdq_w_out],
                domain = "ai.onnx.contrib"
            )
            # Swap in the QuantDequant output as the new weight input
            node.input[1] = qdq_w_out
            nodes_to_add.append(qdq_w)
    
    # Append new nodes in topological order
    graph.node.extend(nodes_to_add)

    # Save qat model
    model_def = helper.make_model(graph, opset_imports=[
            helper.make_opsetid('', 13), helper.make_opsetid('qat', 1)])
    
    onnx.save(model_def, output_model_path)
    print(f"QAT model saved to {output_model_path}")

if __name__ == '__main__':
    onnx_model_path = "models/model.onnx"
    quantized_params_path = "params/quantized_params.json"
    output_model_path = "models/qat_model.onnx"
    prepare_qat_model(onnx_model_path, quantized_params_path, output_model_path)