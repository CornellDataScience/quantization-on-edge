import numpy as np

def compute_scale_and_zero_point_values(r_min, r_max, bit_size):
    '''
    Computes the scale (S) and zero-point (Z) values required for symmetric linear quantization.

    Input
    -----
    r_min : minimum value in the data range.
    r_max : maximum value in the data range.
    bit_size : number of bits to represent quantized values.
    
    Output
    -----
    Returns a tuple (S,Z)
    '''
    if r_min == r_max:
        return 1.0, 0
    
    S = (r_max - r_min) / (2**bit_size - 1)
    Z = 0
    return S, Z

def compute_scale_and_zero_point_values_asymm(r_min, r_max, bit_size):
    '''
    Computes the scale (S) and zero-point (Z) values required for asymmetric linear quantization.

    Input
    -----
    r_min : minimum value in the data range.
    r_max : maximum value in the data range.
    bit_size : number of bits to represent quantized values.
    
    Output
    -----
    Returns a tuple (S,Z)
    '''
    if r_min == r_max:
        return 1.0, 0
    
    qmin, qmax = 0, 2**bit_size - 1
    S = (r_max - r_min) / (qmax - qmin)
    Z = np.clip(np.round(-r_min / S), qmin, qmax)
    return S, Z

def linear_quantize_data(data, bit_size):
    '''
    Quantizes a floating-point numpy array or list to a fixed-point representation using symmetric linear quantization.

    Input
    -----
    data : The input data containing floating-point values to be quantized.
    bit_size : The number of bits used for quantization.

    Output
    -----
    Returns a tuple (Q,S,Z)
    '''
    data = np.asarray(data, dtype=np.float32)
    rmin, rmax = np.min(data), np.max(data)

    S, Z = compute_scale_and_zero_point_values(rmin, rmax, bit_size)

    qmin, qmax = -2**(bit_size-1), 2**(bit_size-1) - 1 # signed int8
    Q = np.clip(np.round(data / S + Z), qmin, qmax)
    
    return Q.astype(np.int8), S, Z

def linear_quantize_data_asymm(data, bit_size):
    '''
    Quantizes a floating-point numpy array or list to a fixed-point representation using asymmetric linear quantization.

    Input
    -----
    data : The input data containing floating-point values to be quantized.
    bit_size : The number of bits used for quantization.

    Output
    -----
    Returns a tuple (Q,S,Z)
    '''
    data = np.asarray(data, dtype=np.float32)
    rmin, rmax = np.min(data), np.max(data)

    S, Z = compute_scale_and_zero_point_values_asymm(rmin, rmax, bit_size)

    qmin, qmax = 0, 2**bit_size - 1 # unsigned int8
    Q = np.clip(np.round(data / S + Z), qmin, qmax)
    
    return Q.astype(np.uint8), S, Z

def linear_dequantize_data(data, S, Z):
    '''
    Dequantize quantized data back to the original floating-point representation using the scale and zero-point.

    Input
    -----
    data : The quantized data to be dequantized.
    S : The scale factor used during quantization.
    Z : The zero-point (offset) used during quantization.

    Output
    -----
    Returns the dequantized data as a floating-point numpy array.
    '''
    data = np.asarray(data)
    return (data - Z) * S
