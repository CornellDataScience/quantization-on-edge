.PHONY: all clean setup quantize_params prep_model quantize_activations quantize_biases quantize_symm quantize_asymm validate reference_model

all: setup \
	quantize_params prep_model quantize_activations quantize_biases quantize_model_symm \
	quantize_params_asymm prep_model_asymm quantize_activations_asymm quantize_biases_asymm quantize_model_asymm \
	quantize_params_log prep_model_log quantize_activations_log quantize_biases_log quantize_model_log

clean:
	rm -f models/*.onnx
	rm -f models/*.keras
	rm -f params/*.json
	rm -f activations/*.json
	rm -f biases/*.json

## Create dependencies if missing
models/model.keras:
	python3 drivers/model.py

params/unquantized_params.json:
	python3 drivers/setup.py
	
params/quantized_params.json:
	python3 python_implementation/quantize_parameters.py symmetric

params/quantized_params_asymm.json:
	python3 python_implementation/quantize_parameters.py asymmetric

params/quantized_params_log.json:
	python3 python_implementation/quantize_parameters.py logarithmic

activations/prep_activations.json:
	python3 drivers/activations.py symmetric

activations/prep_activations_asymm.json:
	python3 drivers/activations.py asymmetric

activations/prep_activations_log.json:
	python3 drivers/activations.py logarithmic

activations/quantized_activations.json:
	python3 python_implementation/quantize_activations.py symmetric

activations/quantized_activations_asymm.json:
	python3 python_implementation/quantize_activations.py asymmetric

activations/quantized_activations_log.json:
	python3 python_implementation/quantize_activations.py logarithmic

biases/quantized_biases.json:
	python3 python_implementation/quantize_biases.py symmetric

biases/quantized_biases_asymm.json:
	python3 python_implementation/quantize_biases.py asymmetric

biases/quantized_biases_log.json:
	python3 python_implementation/quantize_biases.py logarithmic

## Functional commands
setup: models/model.keras
	python3 drivers/setup.py

# For symmetric:
quantize_params: params/unquantized_params.json
	python3 python_implementation/quantize_parameters.py symmetric

prep_model: params/quantized_params.json
	python3 python_implementation/quantize_model.py prep

quantize_activations: activations/prep_activations.json
	python3 python_implementation/quantize_activations.py symmetric

quantize_biases: activations/prep_activations.json
	python3 python_implementation/quantize_biases.py symmetric

quantize_model_symm: activations/quantized_activations.json biases/quantized_biases.json
	python3 python_implementation/quantize_model.py symmetric

# For asymmetric:
quantize_params_asymm: params/unquantized_params.json
	python3 python_implementation/quantize_parameters.py asymmetric

prep_model_asymm: params/quantized_params_asymm.json
	python3 python_implementation/quantize_model.py prep_asymm

quantize_activations_asymm: activations/prep_activations_asymm.json
	python3 python_implementation/quantize_activations.py asymmetric

quantize_biases_asymm: activations/prep_activations_asymm.json
	python3 python_implementation/quantize_biases.py asymmetric

quantize_model_asymm: activations/quantized_activations_asymm.json biases/quantized_biases_asymm.json
	python3 python_implementation/quantize_model.py asymmetric

# For logarithmic:
quantize_params_log: params/unquantized_params.json
	python3 python_implementation/quantize_parameters.py logarithmic

prep_model_log: params/quantized_params_log.json
	python3 python_implementation/quantize_model.py prep_log

quantize_activations_log: activations/prep_activations_log.json
	python3 python_implementation/quantize_activations.py logarithmic

quantize_biases_log: activations/prep_activations_log.json
	python3 python_implementation/quantize_biases.py logarithmic

quantize_model_log: activations/quantized_activations_log.json biases/quantized_biases_log.json
	python3 python_implementation/quantize_model.py logarithmic

# For evaluation:
validate: models/quantized_model.onnx models/asymmetric_model.onnx
	python3 drivers/validate.py

reference_model: models/onnx_model.onnx
	python3 drivers/reference.py