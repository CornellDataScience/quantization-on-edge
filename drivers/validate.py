import onnx
from onnxruntime.quantization import quantize_static, quant_pre_process, CalibrationDataReader
from setup import extract_parameters
import tensorflow_datasets as tfds
import numpy as np

class MnistCalibrationDataReader(CalibrationDataReader):
    '''
    Create custom CalibrationDataReader for MNIST dataset

    Input
    -----
    num: number of training examples to calibrate on
    
    Output
    -----
    Returns MnistCalibrationDataReader object
    '''
    def __init__(self, num):
        super().__init__()

        # Load dataset and extract `num` training examples
        dataset = tfds.load("mnist", shuffle_files=True)
        dataset_subset = dataset["train"].take(num)
        self.calibration_images = [np.array(item["image"]) for item in dataset_subset]

        self.current_item = 0

    def get_next(self) -> dict:
        '''
        Generate the input data dict in the input format to ONNX model
        '''
        if self.current_item == len(self.calibration_images):
            return None  # None signals that the calibration is finished

        image = self.calibration_images[self.current_item]
        image = np.reshape(image, (1, 28, 28))
        image = image.astype("float32")

        self.current_item += 1

        return {"input_layer_1": image}

    def __len__(self) -> int:
        return len(self.calibration_images)

if __name__ == "__main__":
    onnx_model_path = "models/model.onnx"
    model_prep_path = "models/model_prep.onnx"
    validation_model_path = "models/validation_model.onnx"
    validation_params_path = "params/validation_quantized_params.json"

    # create CalibrationDataReader with dataset of size 10
    calibration_data_reader = MnistCalibrationDataReader(10)
    
    # prepare model for quantization
    quant_pre_process(onnx_model_path, model_prep_path)
    
    # quantize_static saves model to file, no return value
    quantize_static(model_prep_path, validation_model_path, calibration_data_reader)

    # load quantized_model
    quantized_model = onnx.load(validation_model_path)

    # parse parameters from quantized validation model into JSON file
    extract_parameters(quantized_model, validation_params_path)
    