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
parser.add_argument("--stimulus", type=str, help="dataset for activations")
parser.add_argument("--m1", type=str, help="model after finetuning")
parser.add_argument("--m2", type=str, help="model initialization")
parser.add_argument("--fold", type=str, help="fold")
args = parser.parse_args()

p = 10000
d = 64
val_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = pd.read_csv("data/" + args.stimulus + ".csv")

if (
    args.stimulus.split("/")[0] == "pcam-small"
    or args.stimulus.split("/")[0] == "pcam-middle"
):
    image_height, image_width = 96, 96
elif (
    args.stimulus.split("/")[0] == "breast"
    or args.stimulus.split("/")[0] == "mammograms"
    or args.stimulus.split("/")[0] == "thyroid"
):
    image_height, image_width = 224, 224
else:
    image_height, image_width = 112, 112

mean_CCA = [args.m1, args.m2, args.fold, args.stimulus.split("/")[0]]


def load_resnet(weights):
    if weights != "RadImageNet" and weights != "ImageNet" and weights != "random":
        model_dir = "models/" + weights + "-" + args.fold + ".h5"
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


model1 = load_resnet(args.m1)
model2 = load_resnet(args.m2)

class_mode = compute_class_mode(args.stimulus.split("/")[0])

if args.stimulus.split("/")[0] == "mammograms":
    ImageFile.LOAD_TRUNCATED_IMAGES = True

for layer1, layer2 in zip(model1.layers, model2.layers):
    if (
        layer1.name == "conv1_relu"
        or layer1.name == "conv2_block3_out"
        or layer1.name == "conv3_block4_out"
        or layer1.name == "conv4_block6_out"
        or layer1.name == "conv5_block3_out"
    ):
        func1 = K.function([model1.layers[1].input], [layer1.output])
        func2 = K.function([model2.layers[1].input], [layer2.output])
        h, w = layer1.output_shape[1], layer1.output_shape[2]
        test_data = shuffle(test_data, random_state=random.randint(0, 100)).reset_index(
            drop=True
        )
        layer_corr = []
        for i in range(5):
            validation_generator = val_data_generator.flow_from_dataframe(
                dataframe=test_data,
                directory="../../../data"
                if args.stimulus.split("/")[0] == "RadImageNet"
                else None,
                x_col="path",
                y_col="class",
                target_size=(image_height, image_width),
                batch_size=round(p / (h * w)),
                shuffle=True,
                class_mode=class_mode,
            )

            acts1 = func1([validation_generator[0][0]])[0]
            acts2 = func2([validation_generator[0][0]])[0]
            num_datapoints, h, w, channels = acts1.shape
            f_acts1 = acts1.reshape((num_datapoints * h * w, channels))
            f_acts2 = acts2.reshape((num_datapoints * h * w, channels))
            idx = np.random.choice(np.arange(len(f_acts1.T)), d, replace=False)
            s_acts1 = f_acts1.T[idx, :]
            s_acts2 = f_acts2.T[idx, :]
            f_results = cca_core.get_cca_similarity(
                s_acts1, s_acts2, epsilon=1e-12, verbose=False
            )
            layer_corr.append(np.mean(f_results["cca_coef1"]))
        cca_mean = np.mean(layer_corr)
        mean_CCA.append(cca_mean)
        print("cca_mean", mean_CCA)
with open(r"results/CCA_" + args.stimulus.split("/")[0] + ".csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(mean_CCA)
