from tensorflow.keras.applications import InceptionV3

model = InceptionV3(weights='imagenet', include_top=False)

# print(model.summary())

model.save("models/cnn_model.keras")