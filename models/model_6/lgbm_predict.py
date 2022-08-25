import pandas as pd
import numpy as np
import lightgbm as lgb

from dotenv import dotenv_values

print("Loading Data")
config = dotenv_values('../../.env')
train = pd.read_parquet(config["ENGINEERED_DATA"] + "viml_train_V1.parquet")

FEATURES = train.columns[:-1]
dtrain = lgb.Dataset(train[FEATURES], label=train["target"])

param = {
    "objective": "regression",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "lambda_l1": 9.129659341987261e-05,
    "lambda_l2": 1.094414795223843e-05,
    "num_leaves": 49,
    "feature_fraction": 0.8022866255496439,
    "bagging_fraction": 0.811831515876014,
    "bagging_freq": 2,
    "min_child_samples": 10,
}

print("Train Model")
gbm = lgb.train(param, dtrain)

del dtrain, train

print("Load Test Data and Predict")
test = pd.read_parquet(config["ENGINEERED_DATA"] + "viml_test_V1.parquet")
preds = gbm.predict(test[FEATURES])

del test

print("Saving")
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = preds

submit.to_csv(config["SUBMISSION_FOLDER"] + f'viml_lgbm.csv', index=None)