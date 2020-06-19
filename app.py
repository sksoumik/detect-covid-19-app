from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'ml_models/covid19.model'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()  # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads',
                                 secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1)  # Simple argmax
        # pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
        result = str(pred_class)  # Convert to string
        if result == '[1]':
            return "Negative (Patient Normal)"
        else:
            return "Positive (high possibility: covid-19 affected)"
    return None


if __name__ == '__main__':
    app.run(debug=True)
