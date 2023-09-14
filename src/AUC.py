from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse 
from io_fun.data_import import compute_class_mode
from sklearn.metrics import roc_auc_score
import csv
from PIL import ImageFile
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, help='target dataset')
parser.add_argument('--target', type=str, help='target dataset')
parser.add_argument('--freeze', type=str, help='True, False')

args = parser.parse_args()
val_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

if args.target == 'mammograms':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
class_mode = class_mode = compute_class_mode(args.target)

if args.target == 'pcam-small' or args.target == 'pcam-middle':
	image_height, image_width = 96, 96
elif args.target == 'breast' or args.target == 'thyroid' or args.target == 'rad_thyroid':
	image_height, image_width = 256, 256
elif args.target == 'mammograms':
	image_height, image_width = 224, 224
else:
	image_height, image_width = 112, 112
                                                              
for f in range(1, 6):
	test_data = pd.read_csv("data/" + args.target + "/test_fold"+ str(f) +".csv")
	validation_generator = val_data_generator.flow_from_dataframe(dataframe=test_data,
                                                      x_col = 'path',
                                                      y_col ='class',
                                                      target_size=(image_height, image_width),
                                                      batch_size=len(test_data),
                                                      shuffle=False,
                                                      class_mode='categorical')
	model_dir = "models/" + args.target + "-" +args.base +"-freeze" + args.freeze + "-fold" + str(f) + ".h5"
	model = load_model(model_dir, compile=False)
	model.compile()
	predictions = model.predict(validation_generator) # get predictions
	onehot = OneHotEncoder()
	y = validation_generator.classes
	yonehot = onehot.fit_transform(np.array(y).reshape(-1,1)).toarray()
	m = keras.metrics.AUC(name='auc')
	m.update_state(yonehot, predictions)
	AUC = m.result().numpy()
	NetAUC = [args.target, args.base, "freeze" + args.freeze , "fold" + str(f), AUC]
	with open(r'results/AUC.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(NetAUC)

