import onnx
import json
import numpy as np
from onnx import numpy_helper

def quantize_biases(onnx_model_path, input_path, output_path):
    '''
    Quantize model biases stored in a JSON file and save the quantized biases to another JSON file.

    Input
    -----
    onnx_model_path: ONNX model to extract biases from
    input_path : file path to the input JSON file containing quantized parameters
    output_path : file path where the quantized biases JSON file will be saved
    
    Output
    -----
    Writes the quantized biases to the specified output file. No value is returned.
    '''
    
    model = onnx.load(onnx_model_path)
    graph = model.graph

    with open(input_path, 'r') as f:
        params = json.load(f)

    bias_floats = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        bias_floats[init.name] = arr.astype(np.float32)

    quantized_biases = {}
    nodes = graph.node
    for i, node in enumerate(nodes):
        if node.op_type == "MatMul" and i+1 < len(nodes):
            next_node = nodes[i+1]
            if next_node.op_type == "Add":
                add_inputs = next_node.input
                matmul_out = node.output[0]
                b_name = add_inputs[0] if add_inputs[1] == matmul_out else add_inputs[1]

                W_name = node.input[1]

                s_W = float(params[W_name]["weight_scale"])
                s_x = float(params[W_name]["activation_scale"])
                s_b = s_x * s_W

                if b_name not in bias_floats:
                    raise KeyError(f"Bias initializer '{b_name}' not found in model.")
                b_raw = bias_floats[b_name]

                b_q = np.round(b_raw / s_b).astype(np.int32)

                quantized_biases[b_name] = {
                    "quantized": b_q.tolist(),
                    "scale": s_b,
                    "bit_width": 32
                }

    with open(output_path, 'w') as f:
        json.dump(quantized_biases, f, indent=2)

if __name__ == "__main__":
    input_json = "params/quantized_params.json"
    output_json = "biases/quantized_biases.json" 
    model_path = "models/quantized_model.onnx"
    quantize_biases(model_path, input_json, output_json)
    print(f"Quantized {input_json} -> {output_json} (32-bit)")