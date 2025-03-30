all: setup quantize_params quantize_model

clean:
	rm -f models/*.onnx
	rm -f models/*.keras
	rm -f params/*.json

.PHONY: clean all setup quantize_params quantize_model

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

quantize_model: params/quantized_params.json
	python3 python_implementation/quantize_model.py

validate: models/model.onnx
	python3 drivers/validate.py