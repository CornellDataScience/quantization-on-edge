import numpy as np

def compute_scale_and_zero_point_values(r_min, r_max, bit_size):
    '''
    (description)

    Input
    -----
    r_min: 
    r_max: 
    bit_size: 
    
    Output
    -----
    Returns 
    '''
    if r_min == r_max:
        return 1.0, 0
    
    S = (r_max - r_min)/(2**bit_size - 1)
    # Z = np.round(-r_min/S)
    Z = 0 # implement symmetric quantization for simplicity
    return S, Z

def linear_quantize_data(data, bit_size):
    '''
    (description)

    Input
    -----
    data: 
    bit_size: 
    
    Output
    -----
    Returns 
    '''
    data = np.asarray(data, dtype=np.float32)
    rmin, rmax = np.min(data), np.max(data)

    S, Z = compute_scale_and_zero_point_values(rmin, rmax, bit_size)

    Q = np.clip(np.round(data / S + Z), -2**(bit_size-1), 2**(bit_size-1) - 1) # valid range from -2^(b-1) to 2^(b-1)-1
    return Q.astype(np.int8), S, Z

def linear_dequantize_data(data, S, Z):
    '''
    (description)

    Input
    -----
    data:
    S:
    Z:
    
    Output
    -----
    Returns
    '''
    data = np.asarray(data)
    return (data - Z) * S

if __name__ == "__main__":
    data = np.array([-0.30, -0.22, -0.24, 0, 0.35])
    bit_size = 8
    
    Q, S, Z = linear_quantize_data(data, bit_size)
    
    dQ = linear_dequantize_data(Q, S, Z)
    
    print(f"Original data:\n{data}")
    print(f"\nQuantized ({bit_size}-bit):\n{Q}")
    print(f"\nDequantized:\n{dQ}")
    print(f"\nScale (S): {S}, Zero-Point (Z): {Z}")
    print(f"Quantization error (MSE): {np.mean((data - dQ)**2)}")