import onnx2tf

def convert_onnx_to_tf(onnx_path):
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_keras_v3=True, output_folder_path="models/converted_model/")

if __name__ == "__main__":
    onnx_path = "models/model.onnx"

    convert_onnx_to_tf(onnx_path)