import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load mnist dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Normalize images (uint8 -> float32)
def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.shuffle(buffer_size=1000)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Load saved tflite model
interpreter = tf.lite.Interpreter(model_path="models/converted_model/model_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Perform inference and display predictions
for images, labels in ds_test.take(1):
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    axes = axes.flatten()

    for i in range(20):
        input_data = images[i].numpy().reshape(input_details[0]['shape'])

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_label = np.argmax(output_data)

        axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
        axes[i].set_title(f"Pred: {predicted_label}\nTrue: {labels[i].numpy()}")
        axes[i].axis("off")

    # plt.show()
    plt.savefig("predictions.png")