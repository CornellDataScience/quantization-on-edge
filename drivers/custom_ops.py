from onnxruntime_extensions import onnx_op, PyCustomOpDef
import numpy as np

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
  
@onnx_op(op_type="AsymmMatMulAddReLUFusion",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float,  PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8],
         outputs=[PyCustomOpDef.dt_int8])
def AsymmMatMulAddReLUFusion(x, W, b, s_x, s_W, s_R, z_x, z_W):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    
    M = (s_x * s_W) / s_R
    return np.array(M * (np.maximum(np.matmul(x - z_x, W - z_W) + b, 0)), dtype=np.int8)

@onnx_op(op_type="AsymmMatMulAddFusion",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8],
         outputs=[PyCustomOpDef.dt_int8])
def AsymmMatMulAddFusion(x, W, b, s_x, s_W, s_b, z_x, z_W):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    
    M = (s_x * s_W) / s_b
    return np.array(M * (np.matmul(x - z_x, W - z_W) + b), dtype=np.int8)
  
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