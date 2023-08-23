import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
import keras
from flask import Flask, request, jsonify
from utils import *

app = Flask(__name__)

# Load the pre-trained Xception model
model = load_model('model/20230822_model_stanford_breed_dogs.h5',
                   custom_objects={"f1_m": f1_m})

print(model.output_names)

# Load encoder classes from the saved file
encoder_classes = np.load('model/encoder_classes.npy', allow_pickle=True)
class_labels = ['American_Staffordshire_terrier', 'Bouvier_des_Flandres', 'Cardigan',
                'Doberman', 'EntleBucher', 'French_bulldog' 'Leonberg', 'Newfoundland',
                'Old_English_sheepdog', 'Pembroke', 'Saluki', 'Tibetan_terrier', 'collie',
                'malamute', 'papillon']

print(len(encoder_classes))
print(len(class_labels))


@app.route('/', methods=['Get'])
def index():
    return 'welcome to standford breed dogs classification'


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"})

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No image provided"})

    img_array = preprocess_image(image)
    img_array = img_array.reshape((-1, 299, 299, 3))
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    predicted_label_index = np.argmax(prediction)
    # Get the label using the index
    predicted_label = class_labels[predicted_label_index]
    confidence = float(prediction[0][predicted_label_index])

    return jsonify({"prediction": predicted_label})


if __name__ == '__main__':
    app.run(debug=True)
