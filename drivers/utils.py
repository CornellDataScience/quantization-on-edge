import tensorflow as tf
import tensorflow_datasets as tfds
import tf2onnx
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader
import json
import numpy as np

class MnistCalibrationDataReader(CalibrationDataReader):
    '''
    Create custom CalibrationDataReader for MNIST dataset

    Input
    -----
    num: number of training examples to calibrate on
    
    Output
    -----
    Returns MnistCalibrationDataReader object
    '''
    def __init__(self, input_layer_name, num):
        super().__init__()

        self.input_layer_name = input_layer_name

        # Load dataset and extract `num` training examples
        dataset = tfds.load("mnist", shuffle_files=True)
        dataset_subset = dataset["train"].take(num)
        for item in dataset_subset:
            print(int(item["label"]))
        self.calibration_images = [np.array(item["image"]) for item in dataset_subset]

        self.current_item = 0

    def get_next(self) -> dict:
        '''
        Generate the input data dict in the input format to ONNX model
        '''
        if self.current_item == len(self.calibration_images):
            return None  # None signals that the calibration is finished

        image = self.calibration_images[self.current_item]
        image = np.reshape(image, (1, 28, 28))
        image = image.astype("float32")

        self.current_item += 1

        return {f"{self.input_layer_name}": image}

    def __len__(self) -> int:
        return len(self.calibration_images)

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
    onnx_model: ONNX model to extract activations from
    output_path: JSON file path to save to
    
    Output
    -----
    Saves (unquantized or quantized) model activations to output_path
    '''
    original_outputs = [x.name for x in onnx_model.graph.output]
    node_output_pairs = [("QuantizeLayer", "quantized_input")] # Create stub for model input

    for node in onnx_model.graph.node:
        for output in node.output:
            if output not in original_outputs:
                new_output = helper.make_tensor_value_info(output, TensorProto.FLOAT, None)
                onnx_model.graph.output.append(new_output)
                original_outputs.append(output)

            node_output_pairs.append((node.name, output))

    session = ort.InferenceSession(onnx_model.SerializeToString())

    output_names = [pair[1] for pair in node_output_pairs[1:]] # Do not include stub in inference session outputs
    print("Model node-output pairs (including intermediate activations):")
    for pair in node_output_pairs:
        print(" -", pair)

    activation_distributions = {pair: [] for pair in node_output_pairs}

    input_layer_name = onnx_model.graph.input[0].name

    reader = MnistCalibrationDataReader(input_layer_name, 10)
    for _ in range(len(reader)):
        sample = reader.get_next()
        if sample is None:
            break

        ort_outs = [sample[input_layer_name].flatten()] # Add model input to outputs
        ort_outs.extend(session.run(output_names, sample))
        
        for pair, act in zip(node_output_pairs, ort_outs):
            activation_distributions[pair].append(act)

    aggregated_activations = {}
    for pair, activations in activation_distributions.items():
        try:
            aggregated_activations[pair] = np.concatenate(
                [a.flatten() for a in activations], axis=0
            ).astype(np.float32)
        except Exception as e:
            print(f"Error concatenating activations for {pair}: {e}")

    serialized_activations = {pair[0]: acts.tolist() for pair, acts in aggregated_activations.items()} # Associate activations with nodes instead of initializers

    with open(output_path, 'w') as f:
        json.dump(serialized_activations, f, indent=2)

    print(f"Activations extracted and saved to: {output_path}")
