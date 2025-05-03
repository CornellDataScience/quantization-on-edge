import json
import numpy as np
from linear_quantization import linear_quantize_data, linear_quantize_data_asymm
import sys

def quantize_activations(input_path, output_path, bit_size, is_symm):
    '''
    Quantize model activations stored in a JSON file and save the quantized activations to another JSON file.

    Input
    -----
    input_path : file path to the input JSON file containing unquantized activations
    output_path : file path where the quantized activations JSON file will be saved
    bit_size : number of bits used for quantization
    is_symm : True for symmetric quantization; False otherwise
    
    Output
    -----
    Writes the quantized activations to the specified output file. No value is returned.
    '''
    with open(input_path, 'r') as f:
        activations = json.load(f)

    quantized_activations = {}
    
    for activation_name, activation_value in activations.items():
        activation_array = np.array(activation_value, dtype=np.float32)
        S, Z = None, None
        if is_symm: # symmetric
            _, S, Z = linear_quantize_data(activation_array, bit_size)
        else: # asymmetric
            _, S, Z = linear_quantize_data_asymm(activation_array, bit_size)
        
        quantized_activations[activation_name] = {
            "scale": float(S),
            "zero_point": int(Z)
        }

    with open(output_path, 'w') as f:
        json.dump(quantized_activations, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected quantization mode argument.")
    else:
        mode = sys.argv[1]
        bit_size = 8

        if mode == "symmetric":
            input_json = "activations/prep_activations.json"
            output_json = "activations/quantized_activations.json" 
            quantize_activations(input_json, output_json, bit_size, is_symm=True)
            print(f"Quantized {input_json} -> {output_json} ({bit_size}-bit)")
        elif mode == "asymmetric":
            input_json = "activations/prep_activations_asymm.json"
            output_json = "activations/quantized_activations_asymm.json" 
            quantize_activations(input_json, output_json, bit_size, is_symm=False)
            print(f"Quantized {input_json} -> {output_json} ({bit_size}-bit)")
