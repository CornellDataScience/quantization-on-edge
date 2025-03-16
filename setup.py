import tensorflow as tf
import tf2onnx
import onnx
import os
import json
from onnx import numpy_helper
import subprocess as subprocess


def convert_tf_to_onnx(saved_model_path, output_path):
    tf_model = tf.keras.models.load_model(saved_model_path)
    
    input_layer = tf_model.inputs[0]

    input_signature = [tf.TensorSpec(input_layer.shape, input_layer.dtype, name=input_layer.name)]
    # print(input_signature)

    tf_model.output_names = ["output"]

    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature, opset=13)

    onnx.save(onnx_model, output_path)
    print(f"ONNX model saved to: {output_path}")

    return onnx_model

def extract_unquantized_parameters(onnx_model, output_path):
    params = {}
    for tensor in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(tensor)
        params[tensor.name] = arr.tolist()

    with open (output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Parameters extracted and saved to: {output_path}")

if __name__ == "__main__":
    saved_model_path = "model.keras"
    onnx_model_path = "model.onnx"
    formatted_parameters_file = "unquantized_params.json"

    onnx_model = convert_tf_to_onnx(saved_model_path, onnx_model_path)
    extract_unquantized_parameters(onnx_model, formatted_parameters_file)
