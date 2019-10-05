# coding: utf_8
import keras

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"     #specify a GPU if you need
images_path = './testimages/*'                 #input images which you want to inference 
output_dir  = './inference/'                   #directory which inferenced images will 
score_threshold = 0.3                          #decide threshold
model_path = './snapshots/resnet152_pascal.h5' #path to the trained h5 file

#os.makedirs(output_dir)

keras.backend.tensorflow_backend.set_session(get_session())

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet152')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

print(model.summary())

labels_to_names = {
    11: 'Malignant'}

import glob
from PIL import Image, ImageDraw, ImageFont

for imgpath in glob.glob(images_path):
    print(imgpath)
    image = np.asarray(Image.open(imgpath).convert('RGB'))

    # copy to draw on
    draw = image.copy()
    #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < score_threshold:
            break
        
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        print((label, score, b))

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    
    imgfile = os.path.basename(imgpath)
    Image.fromarray(draw).save(os.path.join(output_dir, imgfile))




