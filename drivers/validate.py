import onnx
import onnxruntime as ort
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ..python_implementation import linear_quantization as lq

# Create and register custom ONNX operators
@onnx_op(op_type="SymmMatMulAddReLUFusion",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_int8])
def SymmMatMulAddReLUFusion(x, W, b, s_x, s_W, s_R):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    M = (s_x * s_W) / s_R
    return np.array(M * (np.maximum(np.matmul(x, W) + b, 0)), dtype=np.int8)

@onnx_op(op_type="SymmMatMulAddFusion",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_int8])
def SymmMatMulAddFusion(x, W, b, s_x, s_W, s_b):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    M = (s_x * s_W) / s_b
    return np.array(M * (np.matmul(x, W) + b), dtype=np.int8)

@onnx_op(op_type="Quantize",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8],
         outputs=[PyCustomOpDef.dt_int8])
def Quantize(x, s_x, Z):
    bit_size = 8
    return np.array(np.clip(np.round(x / s_x + Z), -2**(bit_size-1), 2**(bit_size-1) - 1), dtype=np.int8)

@onnx_op(op_type="Dequantize",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8],
         outputs=[PyCustomOpDef.dt_float])
def Dequantize(x, s_x, Z):
    return np.array(s_x * (x - Z), dtype=np.float32)

@onnx_op(op_type = "QuantDequant",
         inputs =[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8],
         outputs=[PyCustomOpDef.dt_float])
def QuantDequant(data, bit_size):
    _, S, Z = lq.linear_quantize_data(data, bit_size)
    return lq.linear_dequantize_data(data, S, Z)

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
    Returns accuracy, num_samples of inference
    '''

    dataset = tfds.load(dataset_name, shuffle_files=True)

    num_samples = num_samples if num_samples else len(dataset["test"])
    test_set = dataset["test"].take(num_samples)

    num_correct = 0
    input_layer_name = onnx_model.graph.input[0].name
    for sample in test_set:
        label = int(sample["label"])
        image = np.array(sample["image"])
        image = np.reshape(image, (1, image.shape[0], image.shape[1])).astype(np.float32)

        input = {f"{input_layer_name}": image}
        pred = int(np.argmax(inference_session.run(["output"], input)))

        num_correct += label == pred

    return num_correct / num_samples, num_samples


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
        so.register_custom_ops_library(get_library_path())
        session = ort.InferenceSession(onnx_model.SerializeToString(), so)
    else:
        session = ort.InferenceSession(onnx_model.SerializeToString())

    return onnx_model, session

if __name__ == "__main__":
    onnx_model_path = "models/quantize_model.onnx"

    hasCustom = True
    model, session = create_inference_session(onnx_model_path, hasCustom)
    
    dataset_name = "mnist"
    num_samples = None
    accuracy, num_samples = test(model, session, dataset_name, num_samples)

    print(f"Unquantized Accuracy: {accuracy}% on {num_samples} samples")