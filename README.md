# Quantization on Edge

This project aims to implement and evaluate quantization techniques on machine learning models deployed to Jetson Nano edge devices.

### To Run

Install dependencies: `pip install -r requirements.txt`

Create quantized model from scratch: `make`

### Quantize Symmetric from Scratch

1. Extract unquantized parameters: `make setup`
2. Calculate quantized parameters: `make quantize_params`
3. Create calibration prep model: `make prep_model`
4. Calculate quantized activations: `make quantize_activations`
5. Calculate quantized biases: `make quantize_biases`
3. Statically quantize model: `make quantize_model_symm`
3. Dynamically quantize model: `make quantize_model_dyn_symm`

### Quantize Asymmetric from Scratch

1. Extract unquantized parameters: `make setup`
2. Calculate quantized parameters: `make quantize_params_asymm`
3. Create calibration prep model: `make prep_model_asymm`
4. Calculate quantized activations: `make quantize_activations_asymm`
5. Calculate quantized biases: `make quantize_biases_asymm`
3. Statically quantize model: `make quantize_model_asymm`
3. Dynamically quantize model: `make quantize_model_dyn_asymm`

### Miscellaneous

Validate quantized parameters: `make validate`

Clear `activations/`, `biases/`, `params/`, and `models/` directories: `make clean`