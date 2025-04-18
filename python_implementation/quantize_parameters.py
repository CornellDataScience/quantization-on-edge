import json
import numpy as np
from linear_quantization import linear_quantize_data

def quantize_parameters(input_path, output_path, bit_size):
    '''
    Quantize model parameters stored in a JSON file and save the quantized parameters to another JSON file.

    Input
    -----
    input_path : file path to the input JSON file containing unquantized parameters
    output_path : file path where the quantized parameters JSON file will be saved
    bit_size : number of bits used for quantization
    
    Output
    -----
    Writes the quantized parameters to the specified output file. No value is returned.
    '''
    with open(input_path, 'r') as f:
        params = json.load(f)

    quantized_params = {}
    
    for param_name, param_value in params.items():
        param_array = np.array(param_value, dtype=np.float32)
        
        if "ReadVariableOp" in param_name and "MatMul" in param_name: # Only quantize weights
            Q, S, Z = linear_quantize_data(param_array, bit_size)
            
            quantized_params[param_name] = {
                "data": Q.tolist(),
                "scale": float(S),
                "zero_point": float(Z),
                "to_quantize": True
            }
        else:
            quantized_params[param_name] = {
                "data": param_array.tolist(),
                "to_quantize": False
            }

    with open(output_path, 'w') as f:
        json.dump(quantized_params, f, indent=2)

if __name__ == "__main__":
    input_json = "params/unquantized_params.json"
    output_json = "params/quantized_params.json" 
    bit_size = 8
    quantize_parameters(input_json, output_json, bit_size)
    print(f"Quantized {input_json} -> {output_json} ({bit_size}-bit)")