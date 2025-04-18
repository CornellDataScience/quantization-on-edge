import json
import numpy as np
from linear_quantization import linear_quantize_data

def quantize_activations(input_path, output_path, bit_size):
    '''
    Quantize model activations stored in a JSON file and save the quantized activations to another JSON file.

    Input
    -----
    input_path : file path to the input JSON file containing unquantized activations
    output_path : file path where the quantized activations JSON file will be saved
    bit_size : number of bits used for quantization
    
    Output
    -----
    Writes the quantized activations to the specified output file. No value is returned.
    '''
    with open(input_path, 'r') as f:
        activations = json.load(f)

    quantized_activations = {}
    
    for activation_name, activation_value in activations.items():
        activation_array = np.array(activation_value, dtype=np.float32)
        _, S, Z = linear_quantize_data(activation_array, bit_size)
        
        quantized_activations[activation_name] = {
            "activation_scale": float(S),
            "activation_zero_point": int(Z),
            "bit_width": bit_size
        }

    with open(output_path, 'w') as f:
        json.dump(quantized_activations, f, indent=2)

if __name__ == "__main__":
    input_json = "activations/unquantized_activations.json"
    output_json = "activations/quantized_activations.json" 
    bit_size = 8
    quantize_activations(input_json, output_json, bit_size)
    print(f"Quantized {input_json} -> {output_json} ({bit_size}-bit)")


