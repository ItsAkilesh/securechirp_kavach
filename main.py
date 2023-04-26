import os
from flask import Flask, request
import tensorflow as tf
import numpy as np
import requests
from PIL import Image

app = Flask(__name__)

def pre_process(img):
    img = tf.keras.preprocessing.image.load_img(img, target_size=(160, 160))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Load the pre-trained model
model = tf.keras.models.load_model("FeatureExtraction_model1.h5")

# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image URL from the POST request body
    img_url = request.json['img_path']

    # Download and save the image locally
    img_data = requests.get(img_url).content
    with open('img.jpg', 'wb') as handler:
        handler.write(img_data)

    # Preprocess the image
    img = pre_process('img.jpg')

    # Make predictions
    prediction = int(tf.round(tf.nn.sigmoid(model.predict(img))))

    # Delete the locally saved image
    os.remove('img.jpg')

    # Return the prediction as the API response
    return {'prediction': prediction}

@app.route('/mailer', methods=['POST'])
def mailer():
    data = request.json
    links = data.get('links', [])
    websites = data.get('websites', [])
    result = mails.mailer(links, websites)
    return jsonify(result)

if __name__ == '__main__':
    app.run()
