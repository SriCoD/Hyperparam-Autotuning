import torch
import pandas as pd
import numpy as np

import utils


DEVICE = "cuda"
DEVICE = "cpu"
EPOCHS = 1000


def run_training(fold):
    df = pd.read_csv("../input/train_features.csv")
    df = df.drop(["cp_time", "cp_dose", "cp_type"], axis=1)

    targets_df = pd.read_csv("../input/train_targets_folds.csv")

    feature_columns = df.drop("sig_id", axis=1).columns
    target_columns = targets_df.drop("sig_id", axis=1).columns

    df = df.merge(targets_df, on="sig_id", how="left")
    # print(df)

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[feature_columns].to_numpy()
    ytrain = train_df[target_columns].to_numpy()

    xvalid = valid_df[feature_columns].to_numpy()
    yvalid = valid_df[target_columns].to_numpy()

    train_dataset = utils.MOADataset(features=xtrain, targets=ytrain)
    valid_dataset = utils.MOADataset(features=xvalid, targets=yvalid)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, num_workers=8, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1024, num_workers=8
    )


if __name__ == "__main__":
    run_training(fold=0)
