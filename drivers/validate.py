import onnx
import onnxruntime as ort
from onnxruntime_extensions import get_library_path
import numpy as np
import tensorflow_datasets as tfds
import time
import os
import custom_ops

def test(onnx_model, inference_session, dataset_name, num_samples):
    '''
    Run num_samples of inference on the ONNX model using the provided inference session

    Input
    -----
    onnx_model: ONNX model to run inference on
    inference_session: ONNX inference session to run inference on
    dataset_name: name of TensorFlow dataset to use for inference
    num_samples: num of data point to use for inference
        If num_samples is None, all test samples will be used
    
    Output
    -----
    Returns accuracy, num_samples of inference, average inference time per sample (ms)
    '''

    dataset = tfds.load(dataset_name, shuffle_files=True)

    num_samples = num_samples if num_samples else len(dataset["test"])
    test_set = dataset["test"].take(num_samples)

    num_correct = 0
    total_time = 0
    input_layer_name = onnx_model.graph.input[0].name
    for sample in test_set:
        label = int(sample["label"])
        image = np.array(sample["image"])
        image = np.reshape(image, (1, image.shape[0], image.shape[1])).astype(np.float32)

        input = {f"{input_layer_name}": image}

        start_time = time.perf_counter()
        pred = int(np.argmax(inference_session.run(["output"], input)))
        end_time = time.perf_counter()

        num_correct += label == pred
        total_time += (end_time - start_time)

    return num_correct / num_samples, num_samples, (total_time / num_samples * 1000)

def create_inference_session(onnx_model_path, hasCustom):
    '''
    Load ONNX model and create inference session

    Input
    -----
    onnx_model_path: path to ONNX model to run inference on
    hasCustom: boolean specifying whether or not the model contains custom operators
    
    Output
    -----
    Returns onnx_model, inference_session for corresponding
    '''

    onnx_model = onnx.load(onnx_model_path)

    if hasCustom:
        domain = "ai.onnx.contrib"
        version = 1
        new_opset = onnx.helper.make_opsetid(domain, version)
        onnx_model.opset_import.append(new_opset)

        so = ort.SessionOptions()
        so.log_severity_level = 3 # only errors (default 2)
        so.register_custom_ops_library(get_library_path())
        session = ort.InferenceSession(onnx_model.SerializeToString(), so)
    else:
        session = ort.InferenceSession(onnx_model.SerializeToString())
        
    return onnx_model, session

if __name__ == "__main__":
    dataset_name = "mnist"
    num_samples = None

    # Unquantized
    onnx_model_path = "models/model.onnx"

    model_size = os.path.getsize(onnx_model_path)
    model, session = create_inference_session(onnx_model_path, hasCustom=False)
    accuracy, num_samples, avg_time = test(model, session, dataset_name, num_samples)

    print("** BASELINE **")
    print(f"Unquantized model size: {model_size} bytes")
    print(f"Unquantized accuracy: {accuracy * 100:.2f}% on {num_samples} samples")
    print(f"Unquantized average time: {avg_time:.4f} ms")

    # Quantized (post-training static, symmetric linear)
    onnx_model_path = "models/quantized_model.onnx"

    model_size = os.path.getsize(onnx_model_path)
    model, session = create_inference_session(onnx_model_path, hasCustom=True)
    accuracy, num_samples, avg_time = test(model, session, dataset_name, num_samples)

    print("** SYMMETRIC **")
    print(f"Quantized model size: {model_size} bytes")
    print(f"Quantized accuracy: {accuracy * 100:.2f}% on {num_samples} samples")
    print(f"Quantized average time: {avg_time:.4f} ms")

    # Quantized (post-training static, asymmetric linear)
    onnx_model_path = "models/asymmetric_model.onnx"

    model_size = os.path.getsize(onnx_model_path)
    model, session = create_inference_session(onnx_model_path, hasCustom=True)
    accuracy, num_samples, avg_time = test(model, session, dataset_name, num_samples)

    print("** ASYMMETRIC **")
    print(f"Quantized model size: {model_size} bytes")
    print(f"Quantized accuracy: {accuracy * 100:.2f}% on {num_samples} samples")
    print(f"Quantized average time: {avg_time:.4f} ms")

    # Quantized (post-training static, asymmetric logarithmic)
    onnx_model_path = "models/logarithmic_model.onnx"

    model_size = os.path.getsize(onnx_model_path)
    model, session = create_inference_session(onnx_model_path, hasCustom=True)
    accuracy, num_samples, avg_time = test(model, session, dataset_name, num_samples)

    print("** LOGARITHMIC **")
    print(f"Quantized model size: {model_size} bytes")
    print(f"Quantized accuracy: {accuracy * 100:.2f}% on {num_samples} samples")
    print(f"Quantized average time: {avg_time:.4f} ms")