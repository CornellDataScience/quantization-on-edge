import onnx
from onnxruntime.quantization import quant_pre_process, quantize_static
from utils import MnistCalibrationDataReader, extract_parameters

if __name__ == "__main__":
    onnx_model_path = "models/model.onnx"
    model_prep_path = "models/validation_prep_model.onnx"
    validation_model_path = "models/validation_model.onnx"
    validation_params_path = "params/validation_quantized_params.json"

    model = onnx.load(onnx_model_path)
    input_layer_name = model.graph.input[0].name

    # create CalibrationDataReader with dataset of size 10
    calibration_data_reader = MnistCalibrationDataReader(input_layer_name, 10)
    
    # prepare model for quantization
    quant_pre_process(onnx_model_path, model_prep_path)
    
    # quantize_static saves model to file, no return value
    quantize_static(model_prep_path, validation_model_path, calibration_data_reader)

    # load quantized_model
    quantized_model = onnx.load(validation_model_path)
    
    # print(quantized_model)

    # print('** nodes **')
    # for node in quantized_model.graph.node:
    #     print("name=%r type=%r input=%r output=%r" % (
    #         node.name, node.op_type, node.input, node.output))

    # parse parameters from quantized validation model into JSON file
    extract_parameters(quantized_model, validation_params_path)
    