from flask import Flask, render_template, request
import random
import numpy as np
from torchvision import transforms, datasets, models
import torch
from utils_model import *
from PIL import Image
from utils_model import get_pretrained_model, predict, load_model, display_prediction_top_k, imshow_tensor, predict_top_k
import matplotlib.pyplot as plt
import os

np.random.seed(2021)
random.seed(2021)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello():
    message = 'Welcome to the flask app of Rakuten test code ! <br/> ' \
              'You can try to classify one picture by two options : <br/>' \
              '1) POST request containing the image can be sent. <br/>' \
              'Info : The server has been tested with client.py. Do not hesitate to check it <br/>' \
              '2) Sending path of the image and image name directly to the URL like this example : ' \
              'localhost:8080/CNN/get_img?PATH='+os.getcwd()+'/'+'data/images/test/LABEL_TO_REPLACE' \
              '/IMAGE_NAME.jpg<br/>' \
              'For example :  LABEL_TO_REPLACE = 2885 &  IMAGE_NAME = image_1232639305_product_3667226148'
    return message


@app.route("/classification_model", methods=['GET', 'POST'])
def get_img_pred():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        file.save(file.filename)
        print(file.name)
        print('load trained model')
        class_names = datasets.ImageFolder(root='data/images/test/').classes
        model = load_model('mobilenet_v2-transfer.pt', len(class_names))
        print('model loaded')
        image_path = file.filename
        model_p, real = predict(image_path, model, class_names)
        plt.switch_backend('Agg')
        display_prediction_top_k(image_path, model, class_names, 5)
        print('Top {} predictions plot exported in graphs folder'.format(5))
        print('Predicted label : {}'.format(model_p))
        print('Real label : {}'.format(real))
        return {'PATH': image_path, 'prediction': model_p, 'real': real}


@app.route("/CNN/get_img")
def get_img_pred_V2():
    print('getting image...')
    image_path = request.args.get('PATH')
    print('load trained model...')
    class_names = datasets.ImageFolder(root='data/images/test/').classes
    model = load_model('mobilenet_v2-transfer.pt', len(class_names))
    print('model loaded')
    model_p, real = predict(image_path, model, class_names)
    plt.switch_backend('Agg')
    display_prediction_top_k(image_path, model, class_names, 5)
    print('Top {} predictions plot exported in graphs folder'.format(5))
    print('Predicted label : {}'.format(model_p))
    print('Real label : {}'.format(real))
    return {'PATH': image_path, 'prediction': model_p, 'real': real}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
