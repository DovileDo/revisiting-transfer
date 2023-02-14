import numpy as np
from keras.utils import HDF5Matrix
import imageio

# set data paths
train_img_path = "../../data/PCam_raw/camelyonpatch_level_2_split_train_x.h5"
train_label_path = "../../data/PCam_raw/camelyonpatch_level_2_split_train_y.h5"
val_img_path = "../../data/PCam_raw/camelyonpatch_level_2_split_valid_x.h5"
val_label_path = "../../data/PCam_raw/camelyonpatch_level_2_split_valid_y.h5"
test_img_path = "../../data/PCam_raw/camelyonpatch_level_2_split_test_x.h5"
test_label_path = "../../data/PCam_raw/camelyonpatch_level_2_split_test_y.h5"

home = "../../data"  # set home path that is as data storage location

# load data into hdf5 type and convert the data values to numpy arrays
x_train = np.asarray(HDF5Matrix(train_img_path, "x").data)
print(x_train.shape)
y_train = np.asarray(HDF5Matrix(train_label_path, "y").data)
print(y_train.shape)
x_val = np.asarray(HDF5Matrix(val_img_path, "x").data)
print(x_val.shape)
y_val = np.asarray(HDF5Matrix(val_label_path, "y").data)
print(y_val.shape)
x_test = np.asarray(HDF5Matrix(test_img_path, "x").data)
print(x_test.shape)
y_test = np.asarray(HDF5Matrix(test_label_path, "y").data)
print(y_test.shape)


def unzip_labels(y):
    """
    :param y: array with every individual label in one list
    :return: array with all labels in one list
    """
    temp = []  # initialize list temp that will store all labels
    y = np.array(y)
    for i in range(0, len(y)):
        temp.append(int(y[i]))  # get label, transform into integer and append to temp
    y = np.array(temp)  # convert list temp to array

    return y


# save images as png locally


def save_png(x, y):
    for index in range(len(y)):
        imageio.imwrite(
            f"{home}/PCam/png_images/{index}_label={y[index]}.png", x[index]
        )


# remove the unnecessary lists around the labels with the unzip_labels() function
y_train = unzip_labels(y_train)
y_val = unzip_labels(y_val)
y_test = unzip_labels(y_test)

save_png(x_train, y_train)
save_png(x_val, y_val)
save_png(x_test, y_test)
"""
# concat all data to prepare for nfolds-cross-validation
x_all = np.concatenate((x_train, x_val, x_test), axis=0)
y_all = np.concatenate((y_train, y_val, y_test), axis=0)


# save images as png locally
home = '../../data'   # set home path that is as data storage location
for index in range(len(y_all)):
    imageio.imwrite(f'{home}/PCam/png_images/{index}_label={y_all[index]}.png',
                    x_all[index])
"""
