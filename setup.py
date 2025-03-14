import tensorflow as tf
import tf2onnx
import onnx
import os
import json
from onnx import numpy_helper

def convert_tf_to_onnx(saved_model_path, output_path):
    model = tf.saved_model.load(saved_model_path)
    
    concrete_function = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    input_signature = [tf.TensorSpec([1] + list(input_shape), tf.float32, name=name)
                      for name, input_shape in concrete_function.structured_input_signature[1].items()]
    
    onnx_model, _ = tf2onnx.convert.from_saved_model(
        saved_model_path,
        input_signature,
        opset=13  
    )
    
    onnx.save(onnx_model, output_path)
    print(f"Model successfully converted and saved to: {output_path}")

def extract_unquantized_parameters(onnx_model, output_path):
    params = {}
    for tensor in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(tensor)
        params[tensor.name] = arr.tolist()

    with open (output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Parameters extracted and saved to: {output_path}")

if __name__ == "__main__":
    saved_model_path = "mnist_model.h5"
    onnx_model_path = "model.onnx"
    formatted_parameters_file = "unquantized_params.json"

    onnx_model = convert_tf_to_onnx(saved_model_path, onnx_model_path)
    extract_unquantized_parameters(onnx_model, formatted_parameters_file)
