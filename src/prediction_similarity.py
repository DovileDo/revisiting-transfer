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

ImNets = ["ImageNet-freezeFalse", "ImageNet-freezeTrue"]
RadNets = ["RadImageNet-freezeFalse", "RadImageNet-freezeTrue"]

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

for i, r in zip(ImNets, RadNets):
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
        realProbs = [
            args.target,
            "ImageNet",
            "RadImageNet",
            i.split("-")[1],
            "fold" + str(f),
            "real",
        ]
        randProbs = [
            args.target,
            "ImageNet",
            "RadImageNet",
            i.split("-")[1],
            "fold" + str(f),
            "rand",
        ]
        model_dir = "models/" + args.target + "-" + i + "-fold" + str(f) + ".h5"
        ImNet = load_model(model_dir)
        model_dir = "models/" + args.target + "-" + r + "-fold" + str(f) + ".h5"
        RadNet = load_model(model_dir)
        ImNetpp = ImNet.predict(validation_generator)
        RadNetpp = RadNet.predict(validation_generator)
        if class_mode == "binary":
            ImNetPreds = np.array(ImNetpp) > 0.5
            RadNetPreds = np.array(RadNetpp) > 0.5
            ImNetPred_error = validation_generator.classes != ImNetPreds.astype(int)
            RadNetPred_error = validation_generator.classes != RadNetPreds.astype(int)
        else:
            ImNetPreds = np.argmax(ImNetpp, axis=1)
            RadNetPreds = np.argmax(RadNetpp, axis=1)
            ImNetPred_error = validation_generator.classes != ImNetPreds
            RadNetPred_error = validation_generator.classes != RadNetPreds
        # similarity
        ImRadSim = ImNetPred_error.astype(int) == RadNetPred_error.astype(int)
        P = np.mean(ImRadSim.astype(int))
        # independent similarity
        ImNetScore, ImNetAcc = ImNet.evaluate(validation_generator)
        RadNetScore, RadNetAcc = RadNet.evaluate(validation_generator)
        print("ImNet acc", ImNetAcc)
        print("RadNet acc", RadNetAcc)
        randP = ImNetAcc * ImNetAcc + (1 - ImNetAcc) * (1 - ImNetAcc)
        print(randP)
        realProbs.append(P)
        randProbs.append(randP)
        # AUC
        if class_mode == "binary":
            IAUC = roc_auc_score(validation_generator.classes, ImNetpp)
            RAUC = roc_auc_score(validation_generator.classes, RadNetpp)
        else:
            IAUC = roc_auc_score(
                validation_generator.classes,
                ImNetpp,
                multi_class="ovr",
                average="weighted",
            )
            RAUC = roc_auc_score(
                validation_generator.classes,
                RadNetpp,
                multi_class="ovr",
                average="weighted",
            )
        ImNetAUC = [
            args.target,
            i.split("-")[0],
            i.split("-")[1],
            "fold" + str(f),
            IAUC,
        ]
        RadNetAUC = [
            args.target,
            r.split("-")[0],
            r.split("-")[1],
            "fold" + str(f),
            RAUC,
        ]
        with open(r"results/AUC.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ImNetAUC)
            writer.writerow(RadNetAUC)
        with open(r"results/similarity.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(realProbs)
            writer.writerow(randProbs)
