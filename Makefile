.PHONY: all clean setup quantize_params prep_model quantize_model validate

all: setup quantize_params quantize_model

clean:
	rm -f models/*.onnx
	rm -f models/*.keras
	rm -f params/*.json

# Create dependencies if missing
models/model.keras:
	python3 drivers/model.py

params/unquantized_params.json:
	python3 drivers/setup.py
	
params/quantized_params.json:
	python3 python_implementation/quantize_parameters.py

# Functional commands
setup: models/model.keras
	python3 drivers/setup.py

quantize_params: params/unquantized_params.json
	python3 python_implementation/quantize_parameters.py

prep_model: params/quantized_params.json
	python3 python_implementation/quantize_model.py prep

quantize_model: activations/quantized_activations.json biases/quantized_biases.json
	python3 python_implementation/quantize_model.py full

validate: models/model.onnx
	python3 drivers/validate.py