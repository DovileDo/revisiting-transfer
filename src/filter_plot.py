from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import ResNet50
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import argparse
import os, sys

sys.path.append("./svcca/")
import cca_core
import pwcca
import csv
from io_fun.data_import import compute_class_mode
from sklearn.utils import shuffle
import random
from PIL import ImageFile

parser = argparse.ArgumentParser()
# parser.add_argument('--stimulus', type=str, help='dataset for activations')
parser.add_argument("--m", type=str, help="model after finetuning")
parser.add_argument("--fold", type=str, help="fold")
args = parser.parse_args()

# load the model
def load_resnet(weights):
    if weights != "RadImageNet" and weights != "ImageNet" and weights != "random":
        model_dir = "models/" + weights + ".h5"
        model = load_model(model_dir)
        model = model.layers[1]
        print("loaded", model_dir)
    else:
        if weights == "random":
            model = ResNet50(
                weights=None,
                input_shape=(image_height, image_width, 3),
                include_top=False,
                pooling="avg",
            )
            print("loaded random")
        elif weights == "ImageNet":
            model = ResNet50(
                weights="imagenet",
                input_shape=(image_height, image_width, 3),
                include_top=False,
                pooling="avg",
            )
            print("loaded ImageNet")
        elif weights == "RadImageNet":
            model_dir = "models/RadImageNet-ResNet50_notop.h5"
            model = ResNet50(
                weights=model_dir,
                input_shape=(image_height, image_width, 3),
                include_top=False,
                pooling="avg",
            )
            print("loaded RadImageNet")

    return model


image_height, image_width = 224, 224
model = load_resnet(args.m)
# retrieve weights from the second hidden layer

layer = model.layers[2]
print(model.layers[2].get_weights())

filters, biases = model.layers[2].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 36, 0

fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
ax = [fig.add_subplot(6, 6, i + 1) for i in range(36)]

for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    # specify subplot and turn of axis
    a = ax[ix]
    a.set_xticks([])
    a.set_yticks([])
    # plot filter channel in grayscale
    a.imshow(f[:, :, 0], cmap="gray")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.spines["left"].set_visible(False)
    ix += 1
# show the figure
plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.savefig("results/filters/RadImageNet_conv1.pdf", bbox_inches="tight")
