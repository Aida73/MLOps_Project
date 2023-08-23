from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import render_template, redirect
from flask import request, url_for, render_template, redirect
import io
import tensorflow as tf
import os
import ssl
import logging
from keras.models import load_model
from utils import *

#ignore AVX AVX2 warning 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

UPLOAD_FOLDER = os.path.join(app.root_path ,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ssl._create_default_https_context = ssl._create_unverified_context



def load_models():
	
	global model
	model = load_model('20230822_model_stanford_breed_dogs.h5',
                   custom_objects={"f1_m": f1_m})
        
encoder_classes = np.load('encoder_classes.npy', allow_pickle=True)
class_labels = ['American_Staffordshire_terrier', 'Bouvier_des_Flandres', 'Cardigan',
                'Doberman', 'EntleBucher', 'French_bulldog' 'Leonberg', 'Newfoundland',
                'Old_English_sheepdog', 'Pembroke', 'Saluki', 'Tibetan_terrier', 'collie',
                'malamute', 'papillon']
        



def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image



@app.route("/accueil")
def accueil():
    return render_template("accueil.html")




@app.route("/", methods=["POST","GET"])
def predict():
    # Initialize the data dictionary that will be returned to the view
    data = {"success": False}
    title = "Charger une image"
    name = "default.png"

    logging.info("Received request")

    # Ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        image1 = flask.request.files.get("image")
        if image1:
            print("Image uploaded successfully")

            logging.info(f"Received image: {image1.filename}")
            
            # Save the image to the upload folder, for display on the webpage.
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image1.filename)
            image1.save(image_path)
            
            # Read the image in PIL format
            with open(image_path, 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))
            
            # Preprocess the image and prepare it for classification
            processed_image = prepare_image(image, target=(299, 299))
          
            # Classify the input image and initialize the list
            # of predictions to return to the client
             # Predict
            prediction = model.predict(processed_image)
            predicted_label_index = np.argmax(prediction)
            # Get the label using the index
            predicted_label = class_labels[predicted_label_index]
            confidence = float(prediction[0][predicted_label_index])

              # Prepare the prediction result for display
            prediction_result = {"label": predicted_label, "probability": confidence}

            # Update the data["predictions"] list with the prediction result
            data["predictions"] = [prediction_result]
            
            # Indicate that the request was a success
            data["success"] = "Uploaded"
            title = "predict"
            
            return render_template('index.html', data=data, title=title, name=image1.filename)

    # Return the data dictionary as a JSON response
    return render_template('index.html', data=data, title=title, name=name)

# If this is the main thread of execution, first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started.(60sec)"))
    load_models()

    logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    app.run(debug=True, port=5019)  # Change the port number


