import json
import numpy as np
from linear_quantization import linear_quantize_data

def quantize_parameters(input_path, output_path, bit_size):
    with open(input_path, 'r') as f:
        params = json.load(f)

    quantized_params = {}
    
    for param_name, param_value in params.items():
        param_array = np.array(param_value, dtype=np.float32)
        Q, S, Z = linear_quantize_data(param_array, bit_size)
        
        quantized_params[param_name] = {
            "quantized": Q.tolist(),
            "scale": float(S),
            "zero_point": float(Z),
            "bit_width": bit_size
        }

    with open(output_path, 'w') as f:
        json.dump(quantized_params, f, indent=2)

if __name__ == "__main__":
    input_json = "unquantized_params.json"
    output_json = "quantized_params.json" 
    bit_size = 8                          
    quantize_parameters(input_json, output_json, bit_size)
    print(f"Quantized {input_json} -> {output_json} ({bit_size}-bit)")