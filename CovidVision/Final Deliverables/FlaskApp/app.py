import re
import numpy as np 
import os
from flask import Flask, app, request, render_template
from tensorflow. keras import models 
from tensorflow. keras.models import load_model 
import cv2
from tensorflow. keras.preprocessing import image 
from tensorflow.python.ops.gen_array_ops import concat 
from tensorflow.keras.applications.inception_v3 import preprocess_input 
import requests
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
#Loading the model

model=load_model ("model1.h5",compile=False)

app = Flask(__name__, template_folder = 'templates')

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('about.html')

@app.route('/precautions')
def prec():
    return render_template('precautions.html')

@app.route('/test')
def home():
    return render_template('test.html')

@app.route('/vaccine')
def vaccine():
    return render_template('vaccinations.html')

@app. route('/result',methods=["GET", "POST"])
def res():
    print(list(request.files))
    if request.method== "POST":
        f=request.files['image']
        basepath=UPLOAD_FOLDER
        filepath=basepath+f.filename #from anywhere in the system we can #print("upload folder is", filepath)
        f.save(filepath)
#         img=image.load_img(filepath, target_size=(299, 299))
#         x=image.img_to_array(img)#img to array
#         x=np.expand_dims (x,axis=0)#used for adding one more dimension #print(x)
#         x=Image.open(f.stream)
#         img_data=preprocess_input(x)
#         prediction=np.argmax(model.predict(img_data), axis=1)
#         #prediction-model.predict (x) #instead of predict_classes(x) we can use predict (X)
#         #print("prediction is "prediction)
#         index=['COVID', 'Lung_Capacity','Normal', 'Viral Pneumonia']
#         #result = str(index/output[011)
#         result=str(index[ prediction[0]])
#         print(result)
        img = cv2.imread(filepath) 
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = cv2.resize(img,(128,128))
        img=np.array([img])
        result=['Normal','Covid'][int(model.predict(img)[0])]
        if result=='Normal':
            res=1
        else:
            res=0
        return render_template('result.html',result=result,res=res)
    
app.run(port="4000",debug=True, use_reloader=False)