import warnings
warnings.filterwarnings('ignore')
import numpy
import tensorflow as tf
classifierLoad = tf.keras.models.load_model('my_trained_model.h5', compile=False)

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('39_left.jpg',target_size = (224,224))
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
print(test_image)
result = classifierLoad.predict(test_image)

print(result)
print(type(result))
result = np.round(result).astype(int)
print(result[0][0])
if result[0][0] == 1:
    label = "Normal"
elif result[0][1] == 1:
    label = "Cataract"

elif result[0][2] == 1:
    label = "Diabetes"
elif result[0][3] == 1:
    label = "Glaucoma"
elif result[0][4] == 1:
    label = "Hypertension"
elif result[0][5] == 1:
    label = "Myopia"
elif result[0][6] == 1:
    label = "Age Issues"
else:
    label = "Other"
print(label)