import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from .data_paths import get_path
from sklearn import preprocessing
import cv2
from numpy.random import seed
import tensorflow as tf

# set seeds for reproducibility
seed(1)
tf.random.set_seed(2)


def import_ISIC(img_dir, label_dir):
    """
    :param img_dir: directory where images are stored
    :param label_dir: directory where labels are stored
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .jpg
    images = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")
    ]

    # import labels and set image id as index column
    labels = pd.read_csv(label_dir)
    labels = labels.set_index("image")

    tables = []  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=["path"])  # add img path to dataframe
        img_id = img_path[-16:-4]  # get image id from image path
        extracted_label = labels.loc[img_id]  # extract label from label csv
        if extracted_label[0] == 1:
            extracted_label = "MEL"
        elif extracted_label[1] == 1:
            extracted_label = "NV"
        elif extracted_label[2] == 1:
            extracted_label = "BCC"
        elif extracted_label[3] == 1:
            extracted_label = "AKIEC"
        elif extracted_label[4] == 1:
            extracted_label = "BKL"
        elif extracted_label[5] == 1:
            extracted_label = "DF"
        elif extracted_label[6] == 1:
            extracted_label = "VASC"
        entry["class"] = extracted_label  # add label in dataframe in column 'class'

        tables.append(entry)  # combine entry with other entries for dataframe

    train_labels = pd.concat(
        tables, ignore_index=True
    )  # create dataframe from list of tables and reset index
    print(
        train_labels["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return train_labels


def import_chest(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # set paths where training and test data can be found
    train_images = os.path.join(data_dir, "train")
    val_images = os.path.join(data_dir, "val")
    test_images = os.path.join(data_dir, "test")
    types = list(os.listdir(train_images))  # get unique labels (i.e. folders)

    # initiliaze empty lists that will store entries for train and test dataframes
    dataframe_entries = []

    for type_set in types:
        if type_set == ".DS_Store":
            continue
        else:
            for image_dir in [train_images, val_images, test_images]:
                sub_folder = os.path.join(image_dir, type_set)  # set path to images
                # get all files in folder ending with .jpg
                image = [
                    os.path.join(sub_folder, f)
                    for f in os.listdir(sub_folder)
                    if f.endswith(".jpeg")
                ]
                entry = pd.DataFrame(
                    image, columns=["path"]
                )  # add image in dataframe column 'path'
                entry["class"] = type_set  # add label in dataframe in column 'class'
                dataframe_entries.append(
                    entry
                )  # combine entry with other entries for dataframe

    dataframe = pd.concat(
        dataframe_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    file_name = dataframe["path"].str.split("/", expand=True)
    file_name[7] = file_name[7].map(
        lambda x: x.lstrip("IMNORMAL2").lstrip("-IM").replace("_", "-")
    )
    pid = file_name[7].str.split("-", expand=True)
    dataframe["pid"] = pid[0]
    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def import_textures_dtd(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # set paths where training and test data can be found
    types = list(os.listdir(data_dir))  # get all different labels

    # initiliaze empty lists that will store entries for train and test dataframes
    dataframe_entries = []

    for type_set in types:
        if type_set == ".DS_Store":
            continue
        else:
            sub_folder = os.path.join(data_dir, type_set)  # set path to images
            # get all files in folder ending with .jpg
            image = [
                os.path.join(sub_folder, f)
                for f in os.listdir(sub_folder)
                if f.endswith(".jpg")
            ]
            entry = pd.DataFrame(
                image, columns=["path"]
            )  # add image in dataframe column 'path'
            entry["class"] = type_set  # add label in dataframe in column 'class'
            dataframe_entries.append(
                entry
            )  # combine entry with other entries for dataframe

    dataframe = pd.concat(
        dataframe_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def import_PCAM(data_dir, target_data):
    """
    The .h5 files provided on https://github.com/basveeling/pcam have first been converted to numpy arrays in
    pcam_converter.py and saved locally as PNG-images. This function loads the png paths and labels in a dataframe.
    This was a workaround since using HDF5Matrix() from keras.utils gave errors when running Sacred.
    :param data_dir: directory where all data is stored (images and labels)
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .png
    images = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")
    ]

    dataframe_entries = (
        []
    )  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=["path"])  # add img path to dataframe
        if img_path[-5:-4] == "1":
            label = "yes"
        elif img_path[-5:-4] == "0":
            label = "no"
        entry["class"] = label  # add label in dataframe in column 'class'
        dataframe_entries.append(
            entry
        )  # combine entry with other entries for dataframe

    dataframe = pd.concat(
        dataframe_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index

    # get pcam-middle subset
    if target_data == "pcam-middle":
        subset = dataframe.sample(n=100000, replace=False, random_state=2)
        print("Subset PCam-middle created", len(subset))
        print(
            subset["class"].value_counts()
        )  # get information on distribution of labels in dataframe

        return subset

    # get pcam-small subset
    elif target_data == "pcam-small":
        subset = dataframe.sample(n=10000, replace=False, random_state=22)
        print("Subset PCam-small created", len(subset))
        print(
            subset["class"].value_counts()
        )  # get information on distribution of labels in dataframe

    else:
        subset = dataframe

    return subset


def import_KimiaPath(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .tif
    images = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".tif")
    ]

    dataframe_entries = (
        []
    )  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=["path"])  # add img path to dataframe
        # get label from image path, labels are of form LetterInteger (example A2 or A22)
        label = img_path[-7:-5]
        # some labels include double integer so need some preprocessing of the label: either remove / from label or
        # remove integer (i.e. last character of extracted label)
        if "/" in label:
            label = label.replace("/", "")
        else:
            label = label[:-1]
        entry["class"] = label  # append label to dataframe
        dataframe_entries.append(
            entry
        )  # combine entry with other entries for dataframe
    dataframe = pd.concat(
        dataframe_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def import_thyroid(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .tif

    # images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    img_entries = []  # initiliaze empty list that will store entries for dataframe
    label_entries = []
    for e, file_path in enumerate(files):
        if file_path.endswith(".jpg") == True:
            splited = file_path.split("/")[5]
            pid = splited.split("_")[0]
            entry = pd.DataFrame(
                [[pid, file_path]], columns=["pid", "path"]
            )  # add img path to dataframe
            img_entries.append(entry)
        elif file_path.endswith(".xml"):
            meta_df = pd.read_xml(file_path)
            if "tirads" in meta_df.columns:
                entry = pd.DataFrame(
                    [[str(int(meta_df.number[0])), str(meta_df.tirads[7])]],
                    columns=["pid", "class"],
                )
            else:
                entry = pd.DataFrame(
                    [[str(int(meta_df.number[0])), "1.0"]], columns=["pid", "class"]
                )

            label_entries.append(entry)
    imgs = pd.concat(
        img_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    labels = pd.concat(
        label_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    labels["class"] = labels["class"].map(lambda x: int(x.rstrip("abc").rstrip(".0")))
    labels["class"] = np.where(labels["class"] >= 4, "malignant", "benign")
    dataframe = pd.merge(imgs, labels, on="pid")
    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def import_breast(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .tif
    images = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")
    ]

    dataframe_entries = (
        []
    )  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=["path"])  # add img path to dataframe
        # get label from image path
        label = "".join(img_path)
        if "malignant" in str(label):
            entry["class"] = "malignant"  # append label to dataframe
        else:
            entry["class"] = "benign"

        dataframe_entries.append(
            entry
        )  # combine entry with other entries for dataframe
    dataframe = pd.concat(
        dataframe_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def import_knee(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .png
    acl_images = os.path.join(data_dir, "images/acl")
    meniscus_images = os.path.join(data_dir, "images/meniscus")
    # initiliaze empty lists that will store entries for train and test dataframes
    dataframe_entries = []
    for image_dir in [acl_images, meniscus_images]:
        image = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ]
        entry = pd.DataFrame(
            image, columns=["path"]
        )  # add image in dataframe column 'path'
        dataframe_entries.append(
            entry
        )  # combine entry with other entries for dataframe
    dataframe = pd.concat(
        dataframe_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    # get labels
    img_df = dataframe["path"].str.split("/", expand=True)
    img_df["pid"] = img_df[7].str.split("_", expand=True)[0]
    dataframe["pid"] = img_df["pid"]
    train_acl = pd.read_csv(data_dir + "/train-acl.csv", header=None, dtype={0: object})
    acl_df = img_df[img_df[6] == "acl"]
    meniscus_df = img_df[img_df[6] == "meniscus"]
    val_acl = pd.read_csv(data_dir + "/valid-acl.csv", header=None, dtype={0: object})
    train_meniscus = pd.read_csv(
        data_dir + "/train-meniscus.csv", header=None, dtype={0: object}
    )
    val_meniscus = pd.read_csv(
        data_dir + "/valid-abnormal.csv", header=None, dtype={0: object}
    )
    full_acl = pd.concat([train_acl, val_acl])
    full_acl.rename(columns={0: "pid"}, inplace=True)
    full_meniscus = pd.concat([train_meniscus, val_meniscus])
    full_meniscus.rename(columns={0: "pid"}, inplace=True)
    acl_df = acl_df.merge(full_acl, on="pid")
    acl_df["class"] = np.where(acl_df["1_y"] == 0, "normal", "acl")
    meniscus_df = meniscus_df.merge(full_meniscus, on="pid")
    meniscus_df["class"] = np.where(meniscus_df["1_y"] == 0, "normal", "meniscus")
    dataframe = pd.concat([acl_df, meniscus_df], ignore_index=True)
    dataframe["path"] = dataframe[[0, "1_x", 2, 3, 4, 5, 6, 7]].agg("/".join, axis=1)
    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def import_mammograms(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    calc_train = pd.read_csv(data_dir + "/calc_case_description_train_set.csv")
    calc_test = pd.read_csv(data_dir + "/calc_case_description_test_set.csv")
    mass_train = pd.read_csv(data_dir + "/mass_case_description_train_set.csv")
    mass_test = pd.read_csv(data_dir + "/mass_case_description_test_set.csv")
    full = pd.concat([calc_train, calc_test, mass_train, mass_test])
    full.loc[full["pathology"] == "BENIGN_WITHOUT_CALLBACK", "pathology"] = "BENIGN"
    full["image file path"] = (
        "../../../data/mammograms/images/"
        + full["image file path"].str.split("/", expand=True)[0]
        + ".png"
    )
    print(full.columns)
    full.rename(
        columns={"patient_id": "pid", "pathology": "class", "image file path": "path"},
        inplace=True,
    )
    dataframe = full[["pid", "path", "class"]]

    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def import_imagenet(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .jpeg
    images = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".JPEG")
    ]
    dataframe_entries = (
        []
    )  # initiliaze empty list that will store entries for dataframe
    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=["path"])  # add img path to dataframe
        dataframe_entries.append(
            entry
        )  # combine entry with other entries for dataframe
    dataframe = pd.concat(
        dataframe_entries, ignore_index=True
    )  # create dataframe from list of tables and reset index
    labels = pd.read_csv(
        "/media/dasya/3b6c80d6-7b4b-4135-ada5-ca8ad1d83e9b/Dovile/Data/ImageNet/ILSVRC2012_validation_ground_truth.txt",
        header=None,
    )
    dataframe["class"] = labels[0]
    print(
        dataframe["class"].value_counts()
    )  # get information on distribution of labels in dataframe

    return dataframe


def collect_data(home, target_data):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param target_data: dataset used as target dataset
    :return: training and test dataframes
    """
    if target_data == "isic":
        img_dir, label_dir = get_path(home, target_data)
        dataframe = import_ISIC(img_dir, label_dir)
    else:
        data_dir = get_path(home, target_data)

        if (target_data == "pcam-middle") | (target_data == "pcam-small"):
            dataframe = import_PCAM(data_dir, target_data)
        elif target_data == "chest":
            dataframe = import_chest(data_dir)
        elif target_data == "kimia":
            dataframe = import_KimiaPath(data_dir)
        elif target_data == "breast":
            dataframe = import_breast(data_dir)
        elif target_data == "thyroid":
            dataframe = import_thyroid(data_dir)
        elif target_data == "knee":
            dataframe = import_knee(data_dir)
        elif target_data == "mammograms":
            dataframe = import_mammograms(data_dir)
        elif target_data == "imagenet":
            dataframe = import_imagenet(data_dir)

    return dataframe


def compute_class_mode(target_data):
    """
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :return: computes class mode depending on which source dataset is used in case of pretraining and using flow_from_dataframe or which target dataset is used in case of TF
    """
    if (target_data == "isic") | (target_data == "kimia") | (target_data == "knee"):
        class_mode = "categorical"
    else:
        class_mode = "binary"

    return class_mode
