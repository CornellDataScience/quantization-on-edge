import numpy as np

def compute_scale_and_zero_point_values(r_min, r_max, bit_size):
    if r_min == r_max:
        return 1.0, 0
    
    S = (r_max - r_min)/(2**bit_size -1)
    Z = np.round(-r_min/S)
    return S, Z

def linear_quantize_data(data, bit_size):
    data = np.asarray(data, dtype=np.float32)
    rmin, rmax = np.min(data), np.max(data)

    S, Z = compute_scale_and_zero_point_values(rmin, rmax, bit_size)

    Q = np.clip(np.round(data / S + Z),0, 2**bit_size -1)
    return Q.astype(np.int32), S, Z

def linear_dequantize_data(data, S, Z):
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