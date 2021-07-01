"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request, flash, redirect, url_for
from FlaskTest import app
import urllib.request
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2

print(os.getcwd())

# os.chdir(r"./FlaskTest")
# print(os.getcwd())

import tensorflow as tf
from tensorflow import keras
#from keras.preprocessing import img_to_array

def load_model():
    loaded_model = tf.keras.models.load_model(r'./assets/assets')

    return loaded_model

UPLOAD_FOLDER = './images_Flask/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'jfif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Renders the home page."""
    #return "hello world"
    return render_template(
        'index2.html'
    )

@app.route('/', methods = ['POST'])
def submit_and_predict():
    #print(os.getcwd())

    if 'imagefile' not in request.files:
        flash("no file was uploaded")
        return redirect(request.url)
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    image_path = './images_Flask/' + imagefile.filename
    if allowed_file(imagefile.filename):
        imagefile.save(image_path)

        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        print(resized_image.shape)
        labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        resized_image = resized_image.reshape(-1, 32, 32, 3)

        model_to_predict = CV()
        pred2 = model_to_predict.predict(resized_image)
        # print(pred2)
        Y_pred_classes1 = np.argmax(pred2, axis=1) 
        # print(labels[Y_pred_classes1[0]])
        class_image = labels[Y_pred_classes1[0]]

        return render_template("index2.html", prediction=class_image)
    else:
         flash('Allowed image types are - png, jpg, jpeg, gif. jfif')
         return redirect(request.url)

    

#@app.route('/display')
#def display_image(filename):
#    #print('display_image filename: ' + filename)
#    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def CV():
   CV_model = load_model()
   # print(CV_model.summary())
   return CV_model

#@app.route('/contact')
#def contact():
#    """Renders the contact page."""
#    return render_template(
#        'contact.html',
#        title='Contact',
#        year=datetime.now().year,
#        message='Your contact page.'
#    )

#@app.route('/about')
#def about():
#    """Renders the about page."""
#if 'file' not in request.files:
#        flash('No file part')
#        return redirect(request.url)
#    file = request.files['file']
#    if file.filename == '':
#        flash('No image selected for uploading')
#        return redirect(request.url)
#    if file and allowed_file(file.filename):
#        filename = secure_filename(file.filename)
#        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#        #print('upload_image filename: ' + filename)
#        flash('Image successfully uploaded and displayed below')
#        return render_template('index.html', filename=filename)
#    else:
#        flash('Allowed image types are - png, jpg, jpeg, gif')
#        return redirect(request.url)
