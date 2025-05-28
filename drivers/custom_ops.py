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

@onnx_op(op_type="DynSymmMatMulAddReLUFusion",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_float, 
                 PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_float])
def DynSymmMatMulAddReLUFusion(x, W, b, s_W):
    bit_size = 8
    x = x.copy().astype(np.float32)
    W = W.copy().astype(np.int32)
    b = b.copy().astype(np.float32)

    # Calculate quantized input activation
    r_min, r_max = np.min(x), np.max(x)
    
    s_x = (r_max - r_min)/(2**bit_size - 1) if r_min != r_max else 1

    x = np.clip(np.round(x / s_x), -2**(bit_size-1), 2**(bit_size-1) - 1).astype(np.int32)

    # Calculate quantized bias
    s_b = s_x * s_W

    b = np.round(b / s_b).astype(np.int32)

    acc = np.matmul(x, W) + b # Matmul, Add
    acc = np.maximum(acc, 0) # ReLU

    # Dequantization scalar
    M = s_x * s_W

    return (M * acc).astype(np.float32) # Return dequantized activation

@onnx_op(op_type="DynSymmMatMulAddFusion",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_float, 
                 PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_float])
def DynSymmMatMulAddFusion(x, W, b, s_W):
    bit_size = 8
    x = x.copy().astype(np.float32)
    W = W.copy().astype(np.int32)
    b = b.copy().astype(np.float32)

    # Calculate quantized input activation
    r_min, r_max = np.min(x), np.max(x)
    
    s_x = (r_max - r_min)/(2**bit_size - 1) if r_min != r_max else 1

    x = np.clip(np.round(x / s_x), -2**(bit_size-1), 2**(bit_size-1) - 1).astype(np.int32)

    # Calculate quantized bias
    s_b = s_x * s_W

    b = np.round(b / s_b).astype(np.int32)

    acc = np.matmul(x, W) + b # Matmul, Add

    # Dequantization scalar
    M = s_x * s_W

    return (M * acc).astype(np.float32)
  
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
                 PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8],
         outputs=[PyCustomOpDef.dt_uint8])
def AsymmMatMulAddFusion(x, W, b, s_x, s_W, s_b, z_x, z_W):
    x = x.copy().astype(np.int32)
    W = W.copy().astype(np.int32)
    acc = np.matmul(x - z_x, W - z_W) + b # Matmul, Add (no ReLU)

    # Rescale into uint8 output
    M = (s_x * s_W) / s_b
    bit_size = 8
    return np.clip((M * acc), 0, 2**bit_size - 1).astype(np.uint8)

@onnx_op(op_type="DynAsymmMatMulAddReLUFusion",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_float, 
                 PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8],
         outputs=[PyCustomOpDef.dt_float])
def DynAsymmMatMulAddReLUFusion(x, W, b, s_W, z_W):
    bit_size = 8
    x = x.copy().astype(np.float32)
    W = W.copy().astype(np.int32)
    b = b.copy().astype(np.float32)

    # Calculate quantized input activation
    r_min, r_max = np.min(x), np.max(x)
    
    s_x = (r_max - r_min)/(2**bit_size - 1) if r_min != r_max else 1

    qmin, qmax = 0, 2**bit_size - 1
    z_x = np.clip(np.round(-r_min / s_x), qmin, qmax)

    x = np.clip(np.round(x / s_x + z_x), qmin, qmax).astype(np.int32)

    # Calculate quantized bias
    s_b = s_x * s_W

    b = np.round(b / s_b).astype(np.int32)

    acc = np.matmul(x - z_x, W - z_W) + b # Matmul, Add
    acc = np.maximum(acc, 0) # ReLU

    # Dequantization scalar
    M = s_x * s_W

    return (M * acc).astype(np.float32) # Return dequantized activation

@onnx_op(op_type="DynAsymmMatMulAddFusion",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_float, 
                 PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8],
         outputs=[PyCustomOpDef.dt_float])
def DynAsymmMatMulAddFusion(x, W, b, s_W, z_W):
    bit_size = 8
    x = x.copy().astype(np.float32)
    W = W.copy().astype(np.int32)
    b = b.copy().astype(np.float32)

    # Calculate quantized input activation
    r_min, r_max = np.min(x), np.max(x)
    
    s_x = (r_max - r_min)/(2**bit_size - 1) if r_min != r_max else 1

    qmin, qmax = 0, 2**bit_size - 1
    z_x = np.clip(np.round(-r_min / s_x), qmin, qmax)

    x = np.clip(np.round(x / s_x + z_x), qmin, qmax).astype(np.int32)

    # Calculate quantized bias
    s_b = s_x * s_W

    b = np.round(b / s_b).astype(np.int32)

    acc = np.matmul(x - z_x, W - z_W) + b # Matmul, Add

    # Dequantization scalar
    M = s_x * s_W

    return (M * acc).astype(np.float32) # Return dequantized activation
  
@onnx_op(op_type="SymmQuantize",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_int8])
def SymmQuantize(x, s_x):
    bit_size = 8
    return np.array(np.clip(np.round(x / s_x), -2**(bit_size-1), 2**(bit_size-1) - 1), dtype=np.int8)

@onnx_op(op_type="SymmDequantize",
         inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_float])
def SymmDequantize(x, s_x):
    return np.array(s_x * x, dtype=np.float32)

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