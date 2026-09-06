import tensorflow as tf
import numpy as np

MODEL_PATH = "best_model.keras"
IMG_SIZE = (224, 224)

class_names = ["COVID", "NORMAL", "PNEUMONIA"]

model = tf.keras.models.load_model(MODEL_PATH)

image_path = input("Enter image path: ")

img = tf.keras.utils.load_img(
    image_path,
    target_size=IMG_SIZE
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array, verbose=0)

predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100

print()
print("Prediction:", predicted_class)
print("Confidence:", f"{confidence:.2f}%")