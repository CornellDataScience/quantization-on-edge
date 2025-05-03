import numpy as np

shift = 0

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
    Returns a tuple (Q, S, rmin) where:
        Q : quantized values
        S : scale (quantization step in log space)
        rmin : minimum log value (used in dequantization)
    '''
    data = np.asarray(data, dtype=np.float32)
    
    if np.any(data <= 0):
        # raise ValueError("Logarithmic quantization requires all data > 0.")
        shift = abs(np.min(data)) + 1e-6
        data = data + shift
    
    # Convert to logarithmic scale (base `base`)
    log_data = np.log(data) / np.log(base)
    
    # Find range in log domain
    rmin, rmax = np.min(log_data), np.max(log_data)
    
    # Calculate scale
    S = (rmax - rmin) / (2**bit_size - 1)
    
    # Quantize
    Q = np.clip(np.round((log_data - rmin) / S), 0, 2**bit_size - 1)
    
    return Q.astype(np.uint8), S, rmin

def logarithmic_dequantize_data(Q, S, rmin, base=2):
    '''
    Dequantizes data from logarithmic quantized values.

    Input
    -----
    Q : quantized data
    S : scale (quantization step in log space)
    rmin : minimum log value used in quantization
    base : base of the logarithm

    Output
    ------
    Returns the dequantized data as a float array
    '''
    log_data = Q * S + rmin
    return base ** log_data - shift
