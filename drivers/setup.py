from utils import convert_tf_to_onnx, extract_parameters

if __name__ == "__main__":
    saved_model_path = "models/model.keras"
    onnx_model_path = "models/model.onnx"
    formatted_parameters_file = "params/unquantized_params.json"

    onnx_model = convert_tf_to_onnx(saved_model_path, onnx_model_path)
    extract_parameters(onnx_model, formatted_parameters_file)
