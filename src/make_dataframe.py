import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from io_fun.data_import import collect_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, help="make of the dataset to be split into 5folds"
)
args = parser.parse_args()

home = "../../../data"
dataframe = collect_data(home, args.dataset)
if (
    args.dataset == "chest"
    or args.dataset == "thyroid"
    or args.dataset == "mammograms"
    or args.dataset == "knee"
):
    gkf = GroupKFold(n_splits=5)
    fold_no = 1  # initialize fold counter

    for train_index, val_index in gkf.split(dataframe, groups=dataframe["pid"]):
        train_data = dataframe.iloc[
            train_index
        ]  # create training dataframe with indices from fold split
        print("train:", len(train_data))
        valid_data = dataframe.iloc[val_index]
        train_data.to_csv(
            "data/" + args.dataset + "/train_fold" + str(fold_no) + ".csv"
        )
        print("val:", len(valid_data))
        valid_data.to_csv("data/" + args.dataset + "/val_fold" + str(fold_no) + ".csv")
        fold_no += 1
else:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    fold_no = 1  # initialize fold counter

    for train_index, val_index in skf.split(
        np.zeros(len(dataframe)), y=dataframe[["class"]]
    ):
        print(f"Starting fold {fold_no}")
        train_data = dataframe.iloc[
            train_index
        ]  # create training dataframe with indices from fold split
        valid_data = dataframe.iloc[val_index]
        train_data.to_csv(
            "data/" + args.dataset + "/train_fold" + str(fold_no) + ".csv"
        )
        valid_data.to_csv("data/" + args.dataset + "/val_fold" + str(fold_no) + ".csv")
        fold_no += 1
