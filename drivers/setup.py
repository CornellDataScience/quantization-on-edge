import tensorflow as tf
import tf2onnx
import onnx
import json
from onnx import numpy_helper

def convert_tf_to_onnx(saved_model_path, output_path):
    '''
    Convert saved TensorFlow model to ONNX format

    Input
    -----
    saved_model_path: file path of saved TensorFlow model in .keras format
    output_path: file path to save ONNX model to
    
    Output
    -----
    Returns ONNX model converted from TensorFlow 
    '''
    tf_model = tf.keras.models.load_model(saved_model_path)
    
    # Create input signature for ONNX model
    input_layer = tf_model.inputs[0]
    input_signature = [tf.TensorSpec(input_layer.shape, input_layer.dtype, name=input_layer.name)]

    # Create stub dict key to make model capatible with from_keras converation
    tf_model.output_names = ["output"]

    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature, opset=13)

    onnx.save(onnx_model, output_path)
    print(f"ONNX model saved to: {output_path}")

    return onnx_model

def extract_parameters(onnx_model, output_path):
    '''
    Extract model parameters and save in JSON

    Input
    -----
    onnx_model: ONNX model to extract parameters from
    output_path: JSON file path to save to
    
    Output
    -----
    Saves (unquantized or quantized) model parameters to output_path
    '''
    params = {}
    for tensor in onnx_model.graph.initializer:
        print(tensor.name)
        arr = numpy_helper.to_array(tensor)
        params[tensor.name] = arr.tolist()

    with open (output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Parameters extracted and saved to: {output_path}")

if __name__ == "__main__":
    saved_model_path = "models/model.keras"
    onnx_model_path = "models/quantized_model.onnx"
    formatted_parameters_file = "params/quantized_model_params.json"

    onnx_model = convert_tf_to_onnx(saved_model_path, onnx_model_path)
    # onnx_model = onnx.load(onnx_model_path)
    extract_parameters(onnx_model, formatted_parameters_file)
