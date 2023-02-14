from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from io_fun.data_import import compute_class_mode
from sklearn.metrics import roc_auc_score
import csv
from PIL import ImageFile

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, help="target dataset")
args = parser.parse_args()
val_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

if args.target == "mammograms":
    ImageFile.LOAD_TRUNCATED_IMAGES = True

class_mode = class_mode = compute_class_mode(args.target)

if args.target == "pcam-small" or args.target == "pcam-middle":
    image_height, image_width = 96, 96
elif args.target == "breast":
    image_height, image_width = 224, 224
else:
    image_height, image_width = 112, 112

for f in range(1, 6):
    test_data = pd.read_csv("data/" + args.target + "/val_fold" + str(f) + ".csv")
    validation_generator = val_data_generator.flow_from_dataframe(
        dataframe=test_data,
        x_col="path",
        y_col="class",
        target_size=(image_height, image_width),
        batch_size=len(test_data),
        shuffle=False,
        class_mode=class_mode,
    )
    model_dir = "models/" + args.target + "-random-freezeFalse-fold" + str(f) + ".h5"
    ResNet = load_model(model_dir)
    ResNetpp = ResNet.predict(validation_generator)
    # AUC
    if class_mode == "binary":
        AUC = roc_auc_score(validation_generator.classes, ResNetpp)
    else:
        AUC = roc_auc_score(
            validation_generator.classes,
            ResNetpp,
            multi_class="ovr",
            average="weighted",
        )
    ResNetAUC = [args.target, "random", "freezeFalse", "fold" + str(f), AUC]
    with open(r"results/AUC.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ResNetAUC)
