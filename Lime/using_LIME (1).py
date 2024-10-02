# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:38:30 2023

@author: Sukumar
"""
import os, sys
import linecache
import urllib.request
import numpy as np
import argparse
import time
import matplotlib
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import skimage.io

from lime import lime_image
from tensorflow.keras.applications.inception_v3 import preprocess_input as prepro_inp
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

from keras.applications import inception_v3
#try:
#    from .model import InceptionV3  # Use for django run
#except ImportError:
#    from model import InceptionV3  # Use for terminal run


# from .common.data import Dataset

#%%
def get_imagenet_to_label():
    imagenet_code_to_label = {}
    with open("D:/Sukumar/Research_Work/Codes/ImageNet/imagenet_to_label.txt") as f:
            lines = f.readlines()

    for line in lines:
        temp = line.replace('{', '').replace('}', '').split(':')
        imagenet_code_to_label[int(temp[0])] = temp[1].replace('\'', '').strip()
    return imagenet_code_to_label

def get_model(test_image_path):
    my_model = inception_v3.InceptionV3()
    img = image.load_img(test_image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = prepro_inp(x)
    x = np.vstack([x])

    return x[0], my_model


def predict(test_image_path):
    my_model = inception_v3.InceptionV3()
    # my_model.model.summary()
    my_model.summary()
    # my_model = inc_net.InceptionV3()

    # data = Dataset()
    # data.load_data_tfrecord()
    # print(data.train_data)
    # my_model.train(data, epochs=2)

    img = image.load_img(test_image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = prepro_inp(x)
    x = np.vstack([x])

    prediction = my_model.predict(x)

    # for i in my_model.decode_predict(prediction):
    #     print(i)
    return x, decode_predictions(prediction), my_model

#%%
def explain(image, my_model, prediction_rank=1, show_img=True):
    start = time.time()
    explainer = lime_image.LimeImageExplainer(verbose=True)
    explanation = explainer.explain_instance(image, my_model.predict, top_labels=10, hide_color=0, num_samples=1000)
    # print(explanation)

    decoder = get_imagenet_to_label()
    print(explanation.top_labels[prediction_rank], decoder[explanation.top_labels[prediction_rank]])

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[prediction_rank], positive_only=False,
                                                num_features=5,
                                                hide_rest=False)  # num_features is top super pixel that gives positive value

    print("Explanation time", time.time() - start)
    if show_img:
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.show()
    masked_image = mark_boundaries(temp / 2 + 0.5, mask)
    return masked_image

#%%
# Referencing lime: https://github.com/marcotcr/lime
print("1")
parser = argparse.ArgumentParser()
print("2")
parser.add_argument("-i", "--image", default=None)
args = parser.parse_args()
print("3")
test_image_path = "D:/Sukumar/Research_Work/Codes/ImageNet/Cats/cat_5.jpg"
print("4")
print("Using file from {}".format(test_image_path))
print("5")
images, prediction, my_model = predict(test_image_path)
print("Decoding")
print("6")
print(prediction)
print("7")
masked_image = explain(images[0], my_model, show_img=True)
print("8")
skimage.io.imshow(masked_image)