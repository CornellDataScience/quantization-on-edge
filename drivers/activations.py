import onnx
from utils import extract_activations

if __name__ == "__main__":
    onnx_model_path = "models/quantized_model.onnx"
    formatted_activations_file = "activations/quantized_activations.json"

    onnx_model = onnx.load(onnx_model_path)
    extract_activations(onnx_model, formatted_activations_file)

    