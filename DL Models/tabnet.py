import torch
import pandas as pd
import dask.dataframe as dd

from dotenv import dotenv_values
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetRegressor

config = dotenv_values('../.env')
kfold = StratifiedKFold(n_splits=2, shuffle=True)
labels = pd.read_csv(config["TRAIN_LABELS_PATH"])["target"].to_numpy().reshape(-1,1)
df = pd.read_csv(config["WRANGLED_DATA"] + "scaled_train/train-0.csv.part").to_numpy()

num = 0
for train_idx, test_idx in kfold.split(df, labels):
    print(f"Split: {num}")
    reg = TabNetRegressor(device_name="cuda")
    reg.fit(
        X_train=df[train_idx],
        y_train=labels[train_idx],
        eval_set=[(df[train_idx], labels[train_idx]),
                  (df[test_idx], labels[test_idx])],
        eval_metric=['rmse'],
        max_epochs=1,
        batch_size=1024,
        virtual_batch_size=128
    )
    
