import onnx2tf
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

def convert_onnx_to_tf(onnx_path, output_path):
    '''
    Convert saved ONNX model to TensorFlow formats

    Input
    -----
    onnx_path: file path of saved model in .onnx format
    output_path: folder path of saved model in .keras and .tflite formats
    
    Output
    -----
    Saves model in TensorFlow formats to output_path
    '''
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_keras_v3=True, output_folder_path=output_path)


def perform_inference(tflite_path):
    '''
    Perform inference on MNIST dataset using saved TFLite model and compute accuracy of predictions

    Input
    -----
    tflite_path: file path of saved model in .tflite format
    
    Output
    -----
    Outputs accuracy of inference using TFLite model at tflite_path
    '''
    # Load mnist dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Normalize image (uint8 -> float32)
    ds_test = ds_test.map(lambda image, label: (tf.cast(image, tf.float32) / 255., label), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.shuffle(buffer_size=1000)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Load saved tflite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Perform inference and compute accuracy of predictions
    correct = 0
    total = 0

    for images, labels in ds_test:
        for i in range(images.shape[0]):
            input_data = images[i].numpy().reshape(input_details[0]['shape']).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted_label = np.argmax(output_data)

            if predicted_label == labels[i].numpy():
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    onnx_path = "models/model.onnx"
    output_folder_path = "models/converted_model/"
    tflite_path = "models/converted_model/model_float32.tflite"

    convert_onnx_to_tf(onnx_path, output_folder_path)
    perform_inference(tflite_path)