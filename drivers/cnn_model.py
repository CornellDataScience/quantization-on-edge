from tensorflow.keras.applications import InceptionV3

model = InceptionV3(weights='imagenet')

# print(model.summary())

model.save("models/cnn_model.keras")