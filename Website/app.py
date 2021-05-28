from PIL.Image import Image
from flask import Flask, render_template, request, send_from_directory, send_file
#import cv2
import numpy as np





'''from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import easyocr

IMG_SIZE = (250, 500)
NUM_CLASSES = 7
BATCH_SIZE = 6
NUM_EPOCH = 15
FREEZE_LAYERS = 16
LEARNING_RATE = 0.0002
DROP_OUT = .2
class_dictionary = {'10': 0, '100': 1, '20': 2, '200': 3, '2000': 4, '50': 5, '500': 6}
vals = list(class_dictionary.values())
keys = list(class_dictionary.keys())

model = Xception(include_top = False,
              weights = 'imagenet',
              input_tensor = None,
              input_shape = (250, 500, 3))

top_layer = model.output
x = GlobalAveragePooling2D()(top_layer)
op = Dense(NUM_CLASSES, activation = 'softmax', name = 'softmax')(x)

model_final = Model(inputs = model.input, outputs = op)
for layer in model_final.layers[:FREEZE_LAYERS]:
  layer.trainable = False

for layer in model_final.layers[FREEZE_LAYERS:]:
  layer.trainable = True

model_final.compile(optimizer = Adam(lr = LEARNING_RATE),
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

model_final.load_weights('static/Xception_model.h5')
reader = easyocr.Reader(['en'])'''

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/prediction', methods=["POST"])
def prediction():
    
    img = request.files['img']
    img.save("img.jpg")
    image = cv2.imread("img.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    image = np.reshape(image, (1,224,224,3))
    pred = resnet.predict(image)
    pred = np.argmax(pred)
    pred = labels[pred]
    
    return render_template("prediction.html", data=pred)


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img_arr = cv2.resize(img_arr, (500,250))
    test_image = np.expand_dims(img_arr, axis=0)
    test_image = preprocess_input(test_image)
    prediction = model_final.predict(test_image)
    idx = np.argmax(prediction, axis=1)
    confidence = prediction[0, idx] * 100
    digit = keys[vals.index(idx)]

    preds = np.array([digit,confidence])
    COUNT += 1
    return render_template('predict.html', data=preds)

@app.route('/docread')
def docread(path = 'static/{}.jpg'.format(COUNT)):

    global reader

    output = reader.readtext(path)
    paragraph = ''
    i = 0
    while i < len(output):
        paragraph += output[i][1] + " "
        i += 1

    return {"Text": paragraph}

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))



if __name__ == '__main__':
    app.run(debug=True)