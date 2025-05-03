from onnxruntime_extensions import onnx_op, PyCustomOpDef
import numpy as np

# Create and register custom ONNX operators
@onnx_op(op_type="SymmMatMulAddReLUFusion",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int32, 
                 PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_int8])
def SymmMatMulAddReLUFusion(x, W, b, s_x, s_W, s_R):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    M = (s_x * s_W) / s_R
    bit_size = 8
    return np.clip(M * np.maximum(np.matmul(x, W) + b, 0), -2**(bit_size-1), 2**(bit_size-1) - 1).astype(np.int8)
    # return np.array(M * (np.maximum(np.matmul(x, W) + b, 0)), dtype=np.int8)

@onnx_op(op_type="SymmMatMulAddFusion",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int32, 
                 PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_int8])
def SymmMatMulAddFusion(x, W, b, s_x, s_W, s_b):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    M = (s_x * s_W) / s_b # for symmetric, s_b = s_R = s_x * s_W
    bit_size = 8
    return np.clip(M * (np.matmul(x, W) + b), -2**(bit_size-1), 2**(bit_size-1) - 1).astype(np.int8)
    # return np.array(M * (np.matmul(x, W) + b), dtype=np.int8)
  
@onnx_op(op_type="AsymmMatMulAddReLUFusion",
         inputs=[PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_int32, 
                 PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float,  
                 PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8],
         outputs=[PyCustomOpDef.dt_uint8])
def AsymmMatMulAddReLUFusion(x, W, b, s_x, s_W, s_R, z_x, z_W, z_R):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    acc = np.matmul(x - z_x, W - z_W) + b # Matmul, Add
    acc = np.maximum(acc, 0) # ReLU

    # Rescale into uint8 output
    M = (s_x * s_W) / s_R
    bit_size = 8
    return np.clip((M * acc) + z_R, 0, 2**bit_size - 1).astype(np.uint8)

@onnx_op(op_type="AsymmMatMulAddFusion",
         inputs=[PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_int32, 
                 PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, 
                 PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8],
         outputs=[PyCustomOpDef.dt_uint8])
def AsymmMatMulAddFusion(x, W, b, s_x, s_W, s_R, z_x, z_W, z_R):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    acc = np.matmul(x - z_x, W - z_W) + b # Matmul, Add (no ReLU)

    # Rescale into uint8 output
    M = (s_x * s_W) / s_R
    bit_size = 8
    return np.clip((M * acc) + z_R, 0, 2**bit_size - 1).astype(np.uint8)
  
@onnx_op(op_type="SymmQuantize",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8],
         outputs=[PyCustomOpDef.dt_int8])
def SymmQuantize(x, s_x, Z):
    bit_size = 8
    return np.array(np.clip(np.round(x / s_x + Z), -2**(bit_size-1), 2**(bit_size-1) - 1), dtype=np.int8)

@onnx_op(op_type="SymmDequantize",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8],
         outputs=[PyCustomOpDef.dt_float])
def SymmDequantize(x, s_x, Z):
    return np.array(s_x * (x - Z), dtype=np.float32)

@onnx_op(op_type="AsymmQuantize",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8],
         outputs=[PyCustomOpDef.dt_uint8])
def AsymmQuantize(x, s_x, Z):
    bit_size = 8
    return np.array(np.clip(np.round(x / s_x + Z), 0, 2**bit_size - 1), dtype=np.uint8)

@onnx_op(op_type="AsymmDequantize",
         inputs=[PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8],
         outputs=[PyCustomOpDef.dt_float])
def AsymmDequantize(x, s_x, Z):
    return np.array(s_x * (x - Z), dtype=np.float32)

print("Custom operators registered successfully.")