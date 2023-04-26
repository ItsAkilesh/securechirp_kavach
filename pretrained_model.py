import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def pre_process(img):
    img = tf.keras.preprocessing.image.load_img(img, target_size=(160, 160))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


nude_path = r'D:\Anki\Python Projects\Workspace\Kavach23\nude128_resized.jpg'
non_nude_path = r'D:\Anki\Python Projects\Workspace\Kavach23\Dataset\Non-nudes\0a60fc5f3f_resized.jpg'

img = pre_process(nude  _path)

model = tf.keras.models.load_model("FeatureExtraction_model1.h5")
print(int(tf.round(tf.nn.sigmoid(model.predict(img)))))
