import numpy as np

def logarithmic_quantize_data(data, bit_size, base=2):
    '''
    Logarithmically quantizes the input data.

    Input
    -----
    data : numpy array of float values to quantize (must be > 0)
    bit_size : number of bits for quantization
    base : base of the logarithm (default: 2)

    Output
    ------
    Returns a tuple (Q, S, Z) where:
        Q : quantized values
        S : scale
        Z : zero-point
    '''
    data = np.asarray(data, dtype=np.float32)
    
    if np.any(data <= 0):
        raise ValueError("Logarithmic quantization requires all data > 0.")
    
    # computes log base 'base' for the given data point
    log_data = np.log(data) / np.log(base)
    
    # get min and max of data
    rmin, rmax = np.min(log_data), np.max(log_data)
    
    # S is how much one quantization step covers in the log space
    S = (rmax - rmin) / (2**bit_size - 1)
    Z = 0  # symmetric
    
    Q = np.clip(np.round((log_data - rmin) / S), 0, 2**bit_size - 1)
    return Q.astype(np.uint8), S, rmin

def logarithmic_dequantize_data(Q, S, rmin, base=2):
    '''
    Dequantizes data from logarithmic quantized values.

    Input
    -----
    Q : quantized data
    S : scale
    rmin : minimum log value used in quantization
    base : base of the logarithm

    Output
    ------
    Returns the dequantized data as a float array
    '''
    log_data = Q * S + rmin
    return base ** log_data
