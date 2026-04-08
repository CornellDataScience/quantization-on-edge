from onnxruntime_extensions import onnx_op, PyCustomOpDef
# from scipy.signal import convolve2d
import numpy as np
import torch.nn.functional as F
import torch

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

@onnx_op(op_type="ConvBNReLUFusion",
        inputs=[PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int8, PyCustomOpDef.dt_int32, 
                PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
        outputs=[PyCustomOpDef.dt_int8],
        attrs={"strides": PyCustomOpDef.dt_int64, "auto_pad": PyCustomOpDef.dt_int64}) # MaxPool input=float
def ConvBNReLUFusion(x, W, b, s_x, s_W, s_R, **kwargs):
    print(x.shape)
    Cin, N, H, W_in = x.shape # N = batch size
    Cout, _, kH, kW = W.shape
    Hout = H - kH + 1
    Wout = W_in - kW + 1

    print("="*20)
    print(Cin, N, H, W_in)
    print(Cout, kH, kW)
    print("="*20)

    bit_width = 8

    strides = kwargs["strides"]
    auto_pad = kwargs["auto_pad"]

    print(f"auto_pad: {auto_pad}")
    print(f"strides: {strides}")

    mode = "valid"
    if auto_pad:
        if strides == 1:
            mode = "same"
        else:
            mode = "valid"
            # Padding (assume SAME_UPPER)
            h_padding = np.ceil(H / strides) - H
            h_top = np.floor(h_padding / 2)
            h_bottom = np.ceil(h_padding / 2)

            w_padding = np.ceil(W_in / strides) - H
            w_left = np.floor(w_padding / 2)
            w_right = np.ceil(w_padding / 2)
            
            pad_width = (
                (0, 0), # no padding for N axis
                (0, 0), # no padding for Cin axis
                (int(h_top), int(h_bottom)),
                (int(w_left), int(w_right))
            )
            x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

    # Convolution
    x = np.moveaxis(x, 0, 1)

    # print(x.shape)
    Y = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)
    # for n in range(N):
    #     for cout in range(Cout):
            # acc = np.zeros((Hout, Wout), dtype=np.int32)
            # for cin in range(Cin):
            #     acc += F.conv2d(
            #         torch.tensor(x[cin, n, :, :]),            
            #         torch.tensor(W[cout, cin]),         
            #         padding=mode,
            #         stride=strides
            #     )

            # acc = F.conv2d(
            #         torch.tensor(x),            
            #         torch.tensor(W[cout]),         
            #         padding=mode,
            #         stride=strides
            #     )
            # acc = np.moveaxis(np.array(acc), 0, 1)

            # Y[cout, n] = acc + b[cout]

    Y = np.array(F.conv2d(torch.tensor(x),
                 torch.tensor(W),
                 torch.tensor(b),
                 padding=mode,
                 stride=strides
                 ))
    
    Y = np.moveaxis(Y, 0, 1)

    M = s_x * s_W / s_R

    # Relu
    result = np.clip(M * np.maximum(Y, 0), -2**(bit_width-1), 2**(bit_width-1)-1).astype(np.int8)
    
    print(result.shape)
    return result

# TODO: concat op?? some error


print("Custom operators registered successfully.")