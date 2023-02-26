from flask import Flask, render_template, request
import cv2
import numpy as np

from tensorflow.keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
import keras
import keras.backend as K

import nibabel as nib
from PIL import Image
import imutils

import os

from helper import create_detection_model, preprocess_img, standardize, dice_coef, dice_coef_loss, seg_process_files

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/detection',methods = ['POST','GET'])
def detection():
    
    img = request.files['image']
    img.save('static/detect_img.jpg')

    img_arr = cv2.imread('static/detect_img.jpg')

    print(img_arr.shape)

    img_arr = preprocess_img(img_arr)
    img_arr = np.array([img_arr])

    model = create_detection_model()

    model.load_weights('static/models/checkpoint.h5')

    y_hat = model.predict(img_arr)

    if y_hat >= 0.5 : 
        y_hat = 1
    else: 
        y_hat = 0

    print(y_hat) 
    return render_template('detection.html', y_hat = y_hat, img_src = "static/detect_img.jpg")


@app.route('/segmentation', methods = ['POST','GET'])
def segmentation():

    input_to_model = np.zeros((1,240,240,5))
    age = np.zeros((1,1))

    age[0,0] = request.form.get('age')
    file1 = request.files['image1']
    file2 = request.files['image2']
    file3 = request.files['image3']
    file4 = request.files['image4']
    file5 = request.files['image5']

    path = 'static/seg_images'

    try: 
        os.mkdir(path) 
    except OSError as error: 
        pass 

    file1.save('static/seg_images/flair_img.nii')
    file2.save('static/seg_images/t1_img.nii')
    file3.save('static/seg_images/t1ce_img.nii')
    file4.save('static/seg_images/t2_img.nii')
    file5.save('static/seg_images/seg.nii')
    data = seg_process_files(path)

    modalities = os.listdir(path)
    modalities.sort()
    w=0

    for j in range(len(modalities)-1):
        image_path = path + '/' + modalities[j]
        if not(image_path.find('seg.nii') == -1):
            image_path = path + '/' + modalities[j]
            img = nib.load(image_path)
            img_data = img.get_data()
            img_data = np.asarray(img_data)

    img_data[img_data==4] = 3
    input_to_model[0,:,:,:4] = data[:,:,75,:]
    input_to_model[0,:,:,4] = img_data[:,:,75]

    reshaped_data=data[56:184,80:208,13:141,:]
    reshaped_data=reshaped_data.reshape(1,128,128,128,4)

    seg_model = load_model('static/models/3d_model.h5', custom_objects = {'dice_coef_loss' :  dice_coef_loss , 'dice_coef' : dice_coef})
    surv_model = load_model('static/models/surv_pred.h5')

    y_hat = seg_model.predict(x=reshaped_data)
    y_hat = np.argmax(y_hat,axis=-1)


    print(y_hat.shape)
    y_hat_survival = surv_model.predict(x=[input_to_model,age])

    y_hat_survival *= 1767
    y_hat_survival = int(y_hat_survival)
    print(y_hat_survival)

    # print(y_hat.shape)
    # y_hat = y_hat.astype(int)
    # pred_img = Image.fromarray(y_hat)
    # pred_img.save('seg_predicted_img.jpg')

    return render_template('segmentation.html',y_hat_survival = y_hat_survival)

if __name__ == "__main__" :
    app.run()