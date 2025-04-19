# from utils import convert_onnx_to_tf
# import tensorflow as tf
# import numpy as np
# from onnx_tf.backend import prepare
# import onnx


# def convert_onnx_to_tf(onnx_path, tf_output_path):
#     '''
#     Convert ONNX model to TensorFlow SavedModel format

#     Input
#     -----
#     onnx_path: file path of ONNX model
#     tf_output_path: directory to save the TensorFlow SavedModel
    
#     Output
#     -----
#     Returns the TensorFlow model object
#     '''
#     # Load ONNX model
#     onnx_model = onnx.load(onnx_path)

#     # Convert ONNX to TensorFlow
#     tf_rep = prepare(onnx_model)
#     tf_rep.export_graph(tf_output_path)

#     print(f"TensorFlow model saved to: {tf_output_path}")
#     return tf.keras.models.load_model(tf_output_path)

# def convert_tf_to_tflite(saved_model_dir, tflite_output_path):
#     converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#     tflite_model = converter.convert()
#     with open(tflite_output_path, "wb") as f:
#         f.write(tflite_model)
#     print(f"TFLite model saved to: {tflite_output_path}")

# model = convert_onnx_to_tf("models/model.onnx", "converted_tf_model")
# convert_tf_to_tflite("converted_tf_model", "converted_model.tflite")

# interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_sample = x_test[0] 
# y_true = y_test[0]

# x_input = x_sample.astype(np.float32) / 255.0  
# x_input = np.expand_dims(x_input, axis=0)  
# if len(input_details[0]['shape']) == 4:
#     x_input = np.expand_dims(x_input, axis=-1)  
    

# interpreter.set_tensor(input_details[0]['index'], x_input)

# interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])
# predicted_label = np.argmax(output_data)

# print(f"Predicted label: {predicted_label}")
# print(f"True label: {y_true}")



import tensorflow as tf
import numpy as np
import onnx
from onnx_tf.frontend import convert


def convert_onnx_to_tf(onnx_path, tf_output_path):
    """
    Convert ONNX model to TensorFlow SavedModel format

    Input
    -----
    onnx_path: file path of ONNX model
    tf_output_path: directory to save the TensorFlow SavedModel

    Output
    -----
    Returns the TensorFlow model object
    """
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert ONNX to TensorFlow (returns a tf.Module)
    tf_model = convert(onnx_model)

    # Save as TensorFlow SavedModel
    tf.saved_model.save(tf_model, tf_output_path)

    print(f"TensorFlow model saved to: {tf_output_path}")
    return tf_model


def convert_tf_to_tflite(saved_model_dir, tflite_output_path):
    """
    Convert TensorFlow SavedModel to TFLite format and save it

    Input
    -----
    saved_model_dir: path to the SavedModel directory
    tflite_output_path: path to output .tflite file
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_output_path}")


# Step 1: Convert ONNX → TensorFlow SavedModel
convert_onnx_to_tf("models/model.onnx", "converted_tf_model")

# Step 2: Convert TensorFlow → TFLite
convert_tf_to_tflite("converted_tf_model", "converted_model.tflite")

# Step 3: Load and run inference on TFLite model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Step 4: Load sample from MNIST dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_sample = x_test[0]
y_true = y_test[0]

# Step 5: Preprocess input
x_input = x_sample.astype(np.float32) / 255.0
x_input = np.expand_dims(x_input, axis=0)
if len(input_details[0]['shape']) == 4:
    x_input = np.expand_dims(x_input, axis=-1)

# Step 6: Inference
interpreter.set_tensor(input_details[0]['index'], x_input)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)

print(f"Predicted label: {predicted_label}")
print(f"True label: {y_true}")
