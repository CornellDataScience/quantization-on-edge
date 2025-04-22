.PHONY: all clean setup quantize_params prep_model quantize_activations quantize_biases quantize_model validate reference_model

all: setup quantize_params prep_model quantize_activations quantize_biases quantize_model

clean:
	rm -f models/*.onnx
	rm -f models/*.keras
	rm -f params/*.json
	rm -f activations/*.json
	rm -f biases/*.json

# Create dependencies if missing
models/model.keras:
	python3 drivers/model.py

params/unquantized_params.json:
	python3 drivers/setup.py
	
params/quantized_params.json:
	python3 python_implementation/quantize_parameters.py

activations/prep_activations.json:
	python3 drivers/activations.py

# Functional commands
setup: models/model.keras
	python3 drivers/setup.py

quantize_params: params/unquantized_params.json
	python3 python_implementation/quantize_parameters.py

prep_model: params/quantized_params.json
	python3 python_implementation/quantize_model.py prep

quantize_activations: activations/prep_activations.json
	python3 python_implementation/quantize_activations.py

quantize_biases: activations/prep_activations.json
	python3 python_implementation/quantize_biases.py

quantize_model: activations/quantized_activations.json biases/quantized_biases.json
	python3 python_implementation/quantize_model.py full

validate: models/quantized_model.onnx
	python3 drivers/validate.py

reference_model: models/onnx_model.onnx
	python3 drivers/reference.py