import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import tensorflow as tf
import keras
import random

from tensorflow.keras.models import load_model
#my_loaded_model = tf.keras.models.load_model('my_trained_model.h5', custom_objects={'KerasLayer':hub.KerasLayer , 'AdamWeightDecay': optimizer})
model = keras.models.load_model('my_trained_model.h5', compile=False)
new_image_path = '5_left.jpg'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (224, 224))
new_image = new_image / 255.0

# Perform inference
predictions = model.predict(np.expand_dims(new_image, axis=0))
print(predictions)

# Get the class with the highest probability
predicted_class = np.argmax(predictions)
print(predicted_class)
diagnosis = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
result = diagnosis[predicted_class]
print(result)
print(predicted_class)
# Now 'predicted_class' contains the predicted class label for the new image
#print(f"Actual: {label}")
#print(f"Predicted class: {pred_label}")