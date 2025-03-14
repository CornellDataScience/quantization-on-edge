import tensorflow as tf
import tf2onnx
import onnx
import os
import json
from onnx import numpy_helper
import subprocess as subprocess


def convert_tf_to_onnx(saved_model_path, output_path):
    # Use the CLI command through subprocess
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_path,
        "--output", output_path,
        "--opset", "13"
    ]
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error:", result.stderr)
        raise RuntimeError(f"Failed to convert model: {result.stderr}")
    
    print(result.stdout)
    print(f"Model successfully converted and saved to: {output_path}")
    
    # Load the ONNX model for parameter extraction
    onnx_model = onnx.load(output_path)
    
    return onnx_model

# def convert_tf_to_onnx(saved_model_path, output_path):
#     model = tf.saved_model.load(saved_model_path)
    
#     concrete_function = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
#     # Fix the input signature creation
#     input_signature = []
#     for name, tensor_spec in concrete_function.structured_input_signature[1].items():
#         # Get shape from the TensorSpec and create a new TensorSpec
#         shape = tensor_spec.shape.as_list()
#         # Ensure batch dimension is 1
#         if shape[0] is None:
#             shape[0] = 1
#         input_signature.append(tf.TensorSpec(shape, tensor_spec.dtype, name=name))
    
#     print(input_signature)  # Debugging line
    
#     # Use from_function instead of from_saved_model
#     onnx_model, _ = tf2onnx.convert.from_function(
#         function=concrete_function,
#         input_signature=input_signature,
#         opset=13,
#         output_path=output_path
#     )
    

#     # No need to save separately as from_function with output_path does it
#     print(f"Model successfully converted and saved to: {output_path}")
    
#     return onnx_model


# def convert_tf_to_onnx(saved_model_dir_path, saved_model_path, output_path):
#     saved_model = tf.saved_model.load(saved_model_dir_path)
    
#     # concrete_function = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    

#     # print(input_signature)
                        
#     # onnx_model, _ = tf2onnx.convert.from_keras(
#     #     concrete_function,
#     #     input_signature,
#     #     opset=13
#     # )
    
#     model = tf.keras.models.load_model(saved_model_path)

#     concrete_function = saved_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

#     input_signature = []
#     for name, input_shape in concrete_function.structured_input_signature[1].items():
#         shape = input_shape.shape.as_list()


#         if shape[0] is None:
#             shape[0]= 1
#         input_signature.append(tf.TensorSpec(shape, input_shape.dtype, name=name))

#     # input_signature = [tf.TensorSpec([1] + list(input_shape), tf.float32, name=name)
#     #                   for name, input_shape in concrete_function.structured_input_signature[1].items()]

#     onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path="model.onnx")
    
#     onnx.save(onnx_model, output_path)
#     print(f"Model successfully converted and saved to: {output_path}")

def extract_unquantized_parameters(onnx_model, output_path):
    params = {}
    for tensor in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(tensor)
        params[tensor.name] = arr.tolist()

    with open (output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Parameters extracted and saved to: {output_path}")

if __name__ == "__main__":
    # saved_model_dir_path = "saved_model_dir"
    saved_model_path = "saved_model_dir"
    onnx_model_path = "model.onnx"
    formatted_parameters_file = "unquantized_params.json"

    onnx_model = convert_tf_to_onnx(saved_model_path, onnx_model_path)
    extract_unquantized_parameters(onnx_model, formatted_parameters_file)
