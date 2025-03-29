import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import json
import numpy as np
from onnx import numpy_helper

def quantize(onnx_model_path, quantized_params_path, output_model_path):
    '''
    Quantize ONNX model and save quantized model

    Input
    -----
    onnx_model_path: file path of ONNX model to quantize
    quantized_params_path: file path of quantized parameter, scalar, and zero point values used to quantize
    output_model_path: file path to save quantized model to
    
    Output
    -----
    Saves quantized model to output_model_path 
    '''
    # Load unquantized model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Load JSON
    with open(quantized_params_path) as f:
        params = json.load(f)
    
    # Iterate through params, replace with quantized from JSON
    new_initializers = []
    for tensor in graph.initializer:
        if tensor.name in params:
            # Retrieve quantization params from JSON
            quantized_data = np.array(params[tensor.name]["quantized"], dtype=np.int8)
            scale = np.array([params[tensor.name]["scale"]], dtype=np.float32)
            zero_point = np.array([params[tensor.name]["zero_point"]], dtype=np.int8)

            # Convert to ONNX tensors
            quantized_initializer = numpy_helper.from_array(quantized_data, tensor.name)
            scale_initializer = numpy_helper.from_array(scale, f"{tensor.name}_scale")
            zero_point_initializer = numpy_helper.from_array(zero_point, f"{tensor.name}_zero_point")

            # Add quantized initializer and scale/zero-point tensors
            new_initializers.extend([quantized_initializer, scale_initializer, zero_point_initializer])

            print(f"Quantized {tensor.name} with scale={scale[0]} and zero_point={zero_point[0]}")

    # Remove original initializers and replace with new ones
    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers)

    # Save quantized ONNX model
    onnx.save(model, output_model_path)
    print(f"Quantized model saved to {output_model_path}")


if __name__ == "__main__":
    onnx_model_path = "../models/model.onnx"
    quantized_params_path = "../params/quantized_params.json"
    output_model_path = "../models/quantized_model.onnx"
    quantize(onnx_model_path, quantized_params_path, output_model_path)