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

import os
import nibabel as nib

IMG_SIZE = (224,224)

def create_detection_model():
    
    vgg = VGG19(
        #weights=vgg16_weight_path,
        weights="imagenet",include_top=False, input_shape=IMG_SIZE +  (3,)
    )
    vgg19 = Sequential()
    vgg19.add(vgg)
    vgg19.add(layers.Dropout(0.3))
    vgg19.add(layers.Flatten())
    vgg19.add(layers.Dropout(0.5))
    vgg19.add(layers.Dense(1, activation='sigmoid'))

    vgg19.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=['accuracy']
    )
    
    return vgg19

def preprocess_img(img):
    """
    Resize and apply VGG-19 preprocessing
    """
    img = cv2.resize(
        img,
        dsize=IMG_SIZE,
        interpolation=cv2.INTER_CUBIC
    )
    preprocess_input(img)
    return img


def standardize(image):

    standardized_image = np.zeros(image.shape)
    for z in range(image.shape[2]):
        image_slice = image[:,:,z]
        centered = image_slice - np.mean(image_slice)
        if(np.std(centered)!=0):
            centered = centered/np.std(centered) 

        standardized_image[:, :, z] = centered

    return standardized_image

def dice_coef(y_true, y_pred, epsilon=0.00001):
    axis = (0,1,2,3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true*y_true, axis=axis) + K.sum(y_pred*y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator)/(dice_denominator))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def seg_process_files(path):
    modalities = os.listdir(path)
    modalities.sort()

    data = np.zeros((240,240,155,4))
    w=0

    for j in range(len(modalities)-1):
        image_path = path + '/' + modalities[j]
        if (image_path.find('seg.nii') == -1):
            img = nib.load(image_path)
            img_data = img.get_data()
            img_data = np.asarray(img_data)
            img_data = standardize(img_data)

            data[:,:,:,w] = img_data
            w+=1

    return data