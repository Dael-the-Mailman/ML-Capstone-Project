import pandas as pd
import numpy as np
import xgboost as xgb

from dotenv import dotenv_values

print("Loading Data")
config = dotenv_values('../../.env')
train = pd.read_parquet(config["ENGINEERED_DATA"] + "viml_train_V1.parquet")

FEATURES = train.columns[:-1]
dtrain = xgb.DMatrix(train[FEATURES], label=train["target"])

param = {
    "objective": "reg:logistic",
    "booster": "dart",
    "tree_method": "gpu_hist",
    "lambda": 4.869823775427252e-08,
    "alpha": 6.742151163308733e-07,
    "subsample": 0.8207279814897129,
    "colsample_bytree": 0.521626543449627,
    "max_depth": 9,
    "min_child_weight": 7,
    "eta": 0.11489920584715124,
    "gamma": 0.000582014139829056,
    "grow_policy": "depthwise",
    "sample_type": "weighted",
    "normalize_type": "forest",
    "rate_drop": 0.00012420331313312861,
    "skip_drop": 3.5851806140342926e-07
}

print("Train Model")
bst = xgb.train(param, dtrain)

del dtrain, train

print("Load Test Data and Predict")
test = pd.read_parquet(config["ENGINEERED_DATA"] + "viml_test_V1.parquet")
dtest = xgb.DMatrix(test[FEATURES])
preds = bst.predict(dtest)

del test

print("Saving")
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = preds

submit.to_csv(config["SUBMISSION_FOLDER"] + f'viml_xgb.csv', index=None)