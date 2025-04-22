import onnx
import onnxruntime as ort
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path
import numpy as np
from onnx.checker import check_model
from utils import MnistCalibrationDataReader, extract_activations

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

if __name__ == "__main__":
    onnx_model_path = "models/quantized_model.onnx"

    onnx_model = onnx.load(onnx_model_path)
    domain = "ai.onnx.contrib"
    version = 1 # try 2 or 3, I had some issues with the versioning
    new_opset = onnx.helper.make_opsetid(domain, version)
    onnx_model.opset_import.append(new_opset)

    print('** Original nodes **')
    for node in onnx_model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))

    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    session = ort.InferenceSession(onnx_model.SerializeToString(), so)

    # extract_activations(onnx_model, "activations/full_quant_activations.json")

    input_layer_name = onnx_model.graph.input[0].name
    output_names = [x.name for x in onnx_model.graph.output]

    print("label")
    reader = MnistCalibrationDataReader(input_layer_name, 10)
    
    print()
    print("predicted")

    for _ in range(len(reader)):
        sample = reader.get_next()
        if sample is None:
            break

        res = session.run(output_names, sample)
        print(np.argmax(res))
        # print(res)

    # check_model(onnx_model)