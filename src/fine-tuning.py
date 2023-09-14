#!/usr/bin/env python
# coding: utf-8


import argparse
parser = argparse.ArgumentParser()


parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, help='choose RadImageNet or ImageNet')
parser.add_argument('--target', type=str, help='choose isic, chest, pcam-middle, thyroid, breast')
parser.add_argument('--k', type=int, help='fold num', default=1)
parser.add_argument('--freeze', type=bool, help='if True freeze base model and then fine-tune', default=False)
parser.add_argument('--batch_size', type=int, help='batch size', default=256)
parser.add_argument('--image_height', type=int, help='image height')
parser.add_argument('--image_width', type=int, help='image width')
parser.add_argument('--epoch', type=int, help='number of epochs', default=30)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
args = parser.parse_args()



import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras import regularizers, activations
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from io_fun.data_import import compute_class_mode
from PIL import ImageFile
from sklearn.preprocessing import OneHotEncoder

#MLDGPU stuff
import sys
sys.path.append("..")
# from mldgpu import MultilevelDNNGPUBenchmark
import mldgpu 
mlflow = None


### Import pre-trained weights from ImageNet or RadImageNet
database = args.base
if not database in ['RadImageNet', 'ImageNet', 'random']:
    raise Exception('Pre-trained database not exists. Please choose ImageNet or RadImageNet')

target = args.target
if not target in ['isic', 'chest', 'kimia', 'pcam-small', 'pcam', 'pcam-middle', 'rad_thyroid', 'thyroid', 'breast', 'knee', 'mammograms']:
    raise Exception('Target dataset not selected. Please choose isic, chest, pcam-middle/small, thyroid or breast')

if args.image_height is None:
    raise Exception('Image height not specified')

if args.image_width is None:
    raise Exception('Image width not specified')


### Set up training image size, batch size and number of epochs and home
image_height = args.image_height
image_width = args.image_width

# Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                         #rescale=1./255 if database == 'RadImageNet' else None,
                                          rotation_range=10,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          shear_range=0.1,
                                          zoom_range=0.1,
                                          fill_mode='nearest',
                                          horizontal_flip=False if target == 'chest' else True,
                                          )

data_generator = ImageDataGenerator(#rescale=1/255 if database == 'RadImageNet' else None,
                                    preprocessing_function=preprocess_input)
if args.target == 'mammograms':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
# Load data
df_train = pd.read_csv("data/" + target + "/train_fold" + str(args.k) + ".csv")
df_val = pd.read_csv("data/" + target + "/val_fold" + str(args.k) + ".csv")
df_test = pd.read_csv("data/" + target + "/test_fold" + str(args.k) + ".csv")

class_mode = compute_class_mode(target)

train_generator = train_data_generator.flow_from_dataframe(dataframe=df_train,
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            seed = 2,
                                                            #class_mode=class_mode
                                                            class_mode='categorical')


validation_generator = data_generator.flow_from_dataframe(dataframe=df_val,
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            seed = 2,
                                                            #class_mode=class_mode
                                                            class_mode='categorical')

test_generator = data_generator.flow_from_dataframe(dataframe=df_test,
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=len(df_test),
                                                            shuffle=False,
                                                            seed = 2,
                                                            #class_mode=class_mode
                                                            class_mdoe='categorical')

if database == 'random':
    model = ResNet50(weights=None, input_shape=(image_height, image_width, 3), pooling='avg', classes=2 if class_mode == 'binary' else df_train['class'].nunique(), classifier_activation='softmax')# if class_mode == 'binary' else 'softmax')

else:
    if database == 'RadImageNet':
        model_dir = "models/RadImageNet-ResNet50_notop.h5"
        base_model = ResNet50(weights=model_dir, input_shape=(image_height, image_width, 3), include_top=False,pooling='avg')
    elif database == 'ImageNet':
        base_model = ResNet50(weights='imagenet', input_shape=(image_height, image_width, 3), include_top=False,pooling='avg')
    inputs = keras.Input(shape=(image_height, image_width, 3))
    if args.freeze:
        base_model.trainable = False
        y = base_model(inputs, training=False)

    y = base_model(inputs)
    y = Dropout(0.5)(y)
    if class_mode == 'binary':
        predictions = Dense(2, activation='softmax')(y)
        #predictions = Dense(1, activation='sigmoid')(y)
    else:
        predictions = Dense(df_train['class'].nunique(), activation='softmax')(y)
    model = Model(inputs=inputs, outputs=predictions)

# Open a strategy scope.
# with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    # model = get_compiled_model()

filepath= "models/" + target + "-" + database + "-freeze" + str(args.freeze) + "-fold" + str(args.k) + ".h5"  
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', patience=30)
f_es = EarlyStopping(monitor='val_acc', patience=10)

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
        def _log_gradients(self, epoch):
            writer = self._writers['train']

            with writer.as_default(), tf.GradientTape() as g:
                # here we use test data to calculate the gradients
                y_true = tf.convert_to_tensor(validation_generator[0][1], dtype=tf.float32)
                y_pred = self.model(validation_generator[0][0])  # forward-propagation

                loss = self.model.loss(y_true=y_true, y_pred=y_pred)  # calculate loss
                gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                for weights, grads in zip(self.model.trainable_weights, gradients):
                    tf.summary.histogram(
                        weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

            writer.flush()

        def on_epoch_end(self, epoch, logs=None):
            # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
            # but we do need to run the original on_epoch_end, so here we use the super function.
            super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)
            if self.histogram_freq and epoch % self.histogram_freq == 0:
                self._log_gradients(epoch)

#Etensorboard = ExtendedTensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_graph=False, write_images=True)

train_steps =  len(train_generator.labels)/ args.batch_size
val_steps = len(validation_generator.labels) / args.batch_size

with mldgpu.MultiLevelDNNGPUBenchmark() as run:
    mlflow = run
    
    # Log parameters to mlflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    for _ in range(1):#len(train_generator.filenames)):
        image, label = train_generator.next()
        RGBimage = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
        plt.imshow(RGBimage)
        plt.title('Label: {}'.format(label[0]))
        plt.axis('off')
        plt.savefig('train_example.svg')#, bbox_inches='tight', pad_inches=0)
        mlflow.log_artifact('train_example.svg')
    
    ml_metrics = {}

    if args.freeze:
        if class_mode == 'binary':
            loss = BinaryCrossentropy()
        else:
            loss = CategoricalCrossentropy()
        adam = Adam()
        model.compile(optimizer=adam, loss=loss, metrics=['acc'])
        model.fit(train_generator,
                epochs=args.epoch,
                steps_per_epoch=train_steps,
                validation_data=validation_generator,
                validation_steps=val_steps,
                callbacks=[f_es],
                )

        base_model.trainable = True
    
        train_loss, train_acc = model.evaluate(train_generator)
        val_loss, val_acc = model.evaluate(validation_generator)
        ml_metrics['Fr train_loss'] = train_loss
        ml_metrics['Fr train_acc'] = train_acc
        ml_metrics['Fr val_loss'] = val_loss
        ml_metrics['Fr val_acc'] = val_acc

    mlflow.autolog()
    if class_mode == 'binary':
        loss = BinaryCrossentropy()
    else:
        loss = CategoricalCrossentropy()
    adam = Adam(learning_rate=args.lr)
    model.compile(optimizer=adam, loss=loss, metrics=['acc'])
    history = model.fit(train_generator,
                                 epochs=args.epoch,
                                 steps_per_epoch=train_steps,
                                 validation_data=validation_generator,
                                 validation_steps=val_steps,
                                 callbacks=[es,checkpoint])
                                 #callbacks=[Etensorboard,es,checkpoint])
    
    FT_model = load_model(filepath)
    predictions = FT_model.predict(test_generator) # get predictions
    onehot = OneHotEncoder()
    #y = onehot.fit_transform(np.array(test_generator.classes).reshape(-1,1)).toarray()
    y = test_generator.classes
    yonehot = onehot.fit_transform(np.array(y).reshape(-1,1)).toarray()
    m = keras.metrics.AUC(name='auc')
    m.update_state(yonehot, predictions)
    auc = m.result().numpy()
    #if class_mode == 'binary':
    #    auc = roc_auc_score(test_generator.classes, predictions)
    #else:
    #    auc = roc_auc_score(test_generator.classes, predictions, multi_class='ovr', average='weighted')
    ml_metrics['AUC'] = auc
    mlflow.log_metrics(ml_metrics, 0)
    mlflow.log_artifact("models/" + target + "-" + database + "-freeze" + str(args.freeze) + "-fold" + str(args.k) + ".h5")
