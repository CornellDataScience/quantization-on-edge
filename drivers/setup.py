import tensorflow as tf
import tf2onnx
import onnx
import json
from onnx import numpy_helper
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
from validate import MnistCalibrationDataReader

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

def extract_activations(onnx_model, output_path):
    '''
    Extract model activations and save in JSON

    Input
    -----
    onnx_model: ONNX model to extract parameters from
    output_path: JSON file path to save to
    
    Output
    -----
    Saves (unquantized or quantized) model activations to output_path
    '''
    original_outputs = [x.name for x in onnx_model.graph.output]

    for node in onnx_model.graph.node:
        for output in node.output:
            if output not in original_outputs:
                new_output = helper.make_tensor_value_info(output, TensorProto.FLOAT, None)
                onnx_model.graph.output.append(new_output)
                original_outputs.append(output)

    session = ort.InferenceSession(onnx_model.SerializeToString())
    output_names = [x.name for x in session.get_outputs()]
    print("Model outputs (including intermediate activations):")
    for name in output_names:
        print(" -", name)

    activation_distributions = {name: [] for name in output_names}

    reader = MnistCalibrationDataReader(10)
    for _ in range(len(reader)):
        sample = reader.get_next()
        if sample is None:
            break
        ort_outs = session.run(output_names, sample)
        for name, act in zip(output_names, ort_outs):
            activation_distributions[name].append(act)

    aggregated_activations = {}
    for name, activations in activation_distributions.items():
        try:
            aggregated_activations[name] = np.concatenate(
                [a.flatten() for a in activations], axis=0
            ).astype(np.float32)
        except Exception as e:
            print(f"Error concatenating activations for {name}: {e}")

    serialized_activations = {name: acts.tolist() for name, acts in aggregated_activations.items()}

    with open(output_path, 'w') as f:
        json.dump(serialized_activations, f, indent=2)

    print(f"Activations extracted and saved to: {output_path}")

if __name__ == "__main__":
    saved_model_path = "models/model.keras"
    onnx_model_path = "models/model.onnx"
    formatted_parameters_file = "params/unquantized_params.json"
    formatted_activations_file = "activations/unquantized_activations.json"

    onnx_model = convert_tf_to_onnx(saved_model_path, onnx_model_path)
    # onnx_model = onnx.load(onnx_model_path)
    # extract_parameters(onnx_model, formatted_parameters_file)
    # extract_activations(onnx_model, formatted_activations_file)
