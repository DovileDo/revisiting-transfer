import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from io_fun.data_import import collect_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='make of the dataset to be split into 5folds')
args = parser.parse_args()

home = "../../../data"
dataframe = collect_data(home, args.dataset)
if args.dataset == 'chest' or args.dataset == 'thyroid' or args.dataset == 'mammograms' or args.dataset == 'knee':
    gkf = GroupKFold(n_splits=5)#, shuffle=True, random_state=2)
    test_gkf = GroupKFold(n_splits=4)
    fold_no = 1  # initialize fold counter
    for train_index, temp_index in gkf.split(dataframe, groups=dataframe['pid']):
        train_data = dataframe.iloc[train_index]  # create training dataframe with indices from fold split
        print('train:', len(train_data))
        temp_data = dataframe.iloc[temp_index]
        for test_index, val_index in  test_gkf.split(temp_data, groups=temp_data['pid']):
            valid_data = temp_data.iloc[val_index]
            test_data = temp_data.iloc[test_index]
        train_data.to_csv("data/" + args.dataset + "/train_fold" + str(fold_no) + ".csv")
        print('val:', len(valid_data))
        valid_data.to_csv("data/" + args.dataset + "/val_fold" + str(fold_no) + ".csv")
        print('test:', len(test_data))
        test_data.to_csv("data/" + args.dataset + "/test_fold" + str(fold_no) + ".csv")
        fold_no += 1
else:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    test_skf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_no = 1  # initialize fold counter

    for train_index, temp_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):
        print(f'Starting fold {fold_no}')
        train_data = dataframe.iloc[train_index]  # create training dataframe with indices from fold split
        temp_data = dataframe.iloc[temp_index]
        for test_index, val_index in  test_skf.split(np.zeros(len(temp_data)), y=temp_data[['class']]):
            valid_data = temp_data.iloc[val_index]
            test_data = temp_data.iloc[test_index]
        print('train:', len(train_data))
        train_data.to_csv("data/" + args.dataset + "/train_fold" + str(fold_no) + ".csv")
        print('val:', len(valid_data))
        valid_data.to_csv("data/" + args.dataset + "/val_fold" + str(fold_no) + ".csv")
        print('test:', len(test_data))
        test_data.to_csv("data/" + args.dataset + "/test_fold" + str(fold_no) + ".csv")
        fold_no += 1
