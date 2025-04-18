import onnx
import json
import numpy as np
from onnx import numpy_helper

def quantize_biases(onnx_model_path, input_path, input_path_2, output_path):
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
        activations = json.load(f)
    with open(input_path_2, 'r') as f:
        params = json.load(f)
    bias_floats = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        bias_floats[init.name] = arr.astype(np.float32)
    quantized_biases = {}
    nodes = graph.node
    for i, node in enumerate(nodes):
        # print(node.name)
        # print(node.input)
        print(node.op_type)
        if node.op_type != "MatMul":
            continue
        # print("check 1")
        # if i + 1 >= len(nodes):
        #     continue
        next_node = nodes[i + 1]
        # print("check 2")
        # if next_node.op_type != "MatMul":
        #     continue
        # print("check 3")
        if i < 1:
            continue
        prev_node = nodes[i - 1]
        # print("check 4")
        # if len(next_node.input) < 2 or len(node.input) < 2 or len(prev_node.input) < 2:
        #     continue
        # print("check 5")
        print(node.name)
        print(node.input)
        print(node.op_type)
        add_inputs = next_node.input
        b_name = add_inputs[1]
        W_name = node.input[1]
        print(W_name)
        print(node.name)
        W_name_prev = prev_node.output[0]
        s_W = float(params[W_name]["scale"])
        # s_x = float(activations[W_name_prev]["activation_scale"])
        s_x = float(activations[prev_node.name]["scale"])
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
    input_json = "activations/quantized_activations.json"
    input_json_2 = "params/quantized_params.json"
    output_json = "biases/quantized_biases.json"
    model_path = "models/prep_model.onnx"
    quantize_biases(model_path, input_json,input_json_2, output_json)
    print(f"Quantized {input_json} -> {output_json} (32-bit)")
    