import torch
import pandas as pd
import numpy as np

DEVICE = "cuda"
EPOCHS = 1000

def run_training(fold):
    df = pd.read_csv("../input/train_features.csv")
    df = df.drop(["cp_time", "cp_dose", "cp_type"], axis = 1)

    targets_df = pd.read_csv("../input/train_targets_folds.csv")

    feature_columns = df.drop("sig_id", axis = 1).columns
    targets_columns = targets_df.drop("sig_id", axis = 1).columns

    df = df.merge(targets_df, on = "sig_id", how = "left")
    print(df)

if __name__ == '__main__':
    run_training(fold=0)
