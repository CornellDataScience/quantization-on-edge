import onnx
import json
import numpy as np
from onnx import numpy_helper
import sys

def quantize_biases(prep_model_path, quantized_activations_path, quantized_params_path, output_quantized_biases_path):
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

    model = onnx.load(prep_model_path)
    graph = model.graph
    with open(quantized_activations_path, 'r') as f:
        activations = json.load(f)

    with open(quantized_params_path, 'r') as f:
        params = json.load(f)

    bias_floats = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        bias_floats[init.name] = arr.astype(np.float32)

    quantized_biases = {}
    nodes = graph.node
    for i, node in enumerate(nodes):
        if node.op_type == "MatMul":
            prev_node = nodes[i - 1]
            add_node = nodes[i + 1]
        
            b_name = add_node.input[1]
            W_name = node.input[1]

            s_W = float(params[W_name]["scale"])
            s_x = float(activations[prev_node.name]["scale"])
            s_b = s_x * s_W # bias_scale = activation_scale * weight_scale

            if b_name not in bias_floats:
                raise KeyError(f"Bias initializer '{b_name}' not found in model.")
            
            b_raw = bias_floats[b_name]
            b_q = np.round(b_raw / s_b).astype(np.int32)
            quantized_biases[b_name] = {
                "data": b_q.tolist(),
                "scale": s_b
                # for bias, zero_point = 0 always
            }
            
    with open(output_quantized_biases_path, 'w') as f:
        json.dump(quantized_biases, f, indent=2)
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected quantization mode argument.")
    else:
        mode = sys.argv[1]

        prep_model_path, quantized_activations_path, quantized_params_path, output_quantized_biases_path = None, None, None, None

        if mode == "symmetric":
            prep_model_path = "models/prep_model.onnx"
            quantized_activations_path = "activations/quantized_activations.json"
            quantized_params_path = "params/quantized_params.json"
            output_quantized_biases_path = "biases/quantized_biases.json"

        elif mode == "asymmetric":
            prep_model_path = "models/prep_model_asymm.onnx"
            quantized_activations_path = "activations/quantized_activations_asymm.json"
            quantized_params_path = "params/quantized_params_asymm.json"
            output_quantized_biases_path = "biases/quantized_biases_asymm.json"
    
        elif mode == "logarithmic":
            prep_model_path = "models/prep_model_log.onnx"
            quantized_activations_path = "activations/quantized_activations_log.json"
            quantized_params_path = "params/quantized_params_log.json"
            output_quantized_biases_path = "biases/quantized_biases_log.json"
        
        elif mode == "convolution":
            prep_model_path = "models/prep_cnn_model.onnx"
            quantized_activations_path = "activations/quantized_activations_cnn.json"
            quantized_params_path = "params/quantized_params_cnn.json"
            output_quantized_biases_path = "biases/quantized_biases_cnn.json"
            
        quantize_biases(prep_model_path, quantized_activations_path, quantized_params_path, output_quantized_biases_path)
        print(f"Quantized {quantized_activations_path}, {quantized_params_path} -> {output_quantized_biases_path} (32-bit)")