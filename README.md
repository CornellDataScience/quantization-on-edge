# Quantization on Edge

This project aims to implement and evaluate quantization techniques on machine learning models deployed to Jetson Nano edge devices.

### To Run

Install dependencies: `pip install -r requirements.txt`

Create quantized model from scratch: `make`

Validate quantized parameters: `make validate`

Clear `models/` and `params/` directories: `make clean`

### Quantization Flow

1. Extract unquantized parameters: `make setup`
2. Calculate quantized parameters: `make quantize_params`
3. Quantize model: `make quantize_model`