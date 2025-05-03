import onnx
from utils import extract_activations
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected quantization mode argument.")
    else:
        mode = sys.argv[1]

        onnx_model_path, formatted_activations_file = None, None

        if mode == "symmetric":
            onnx_model_path = "models/prep_model.onnx"
            formatted_activations_file = "activations/prep_activations.json"
        elif mode == "asymmetric":
            onnx_model_path = "models/prep_model_asymm.onnx"
            formatted_activations_file = "activations/prep_activations_asymm.json"

        onnx_model = onnx.load(onnx_model_path)
        extract_activations(onnx_model, formatted_activations_file)