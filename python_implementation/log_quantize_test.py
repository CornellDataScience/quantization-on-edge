import numpy as np
from logarithmic_quantization import logarithmic_quantize_data, logarithmic_dequantize_data

def test_logarithmic_quantization():
    # Generate test data: positive values only
    np.random.seed(42)
    original_data = np.random.uniform(0.1, 100.0, size=1000).astype(np.float32)

    # Test with different bit sizes
    for bit_size in [2, 4, 8]:
        print(f"\n=== Testing bit size: {bit_size} ===")

        # Quantize
        try:
            Q, S, rmin = logarithmic_quantize_data(original_data, bit_size)
        except Exception as e:
            print(f"Error during quantization: {e}")
            continue

        # Dequantize
        reconstructed_data = logarithmic_dequantize_data(Q, S, rmin)

        # Compute errors
        mse = np.mean((original_data - reconstructed_data) ** 2)
        max_error = np.max(np.abs(original_data - reconstructed_data))
        relative_error = np.mean(np.abs(original_data - reconstructed_data) / original_data)

        print(f"Mean Squared Error     : {mse:.6f}")
        print(f"Max Absolute Error     : {max_error:.6f}")
        print(f"Mean Relative Error    : {relative_error:.4%}")

        # Optional: check if reconstruction is within tolerance
        if relative_error < 0.1:
            print("✅ Quantization looks good for this bit size!")
        else:
            print("⚠️  High error – consider increasing bit size.")

if __name__ == "__main__":
    test_logarithmic_quantization()
