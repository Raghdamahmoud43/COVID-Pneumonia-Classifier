import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR = "split_dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)

# Load the best trained model
model = tf.keras.models.load_model("best_model_v2.keras")

# Predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))