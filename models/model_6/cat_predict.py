import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from dotenv import dotenv_values

print("Loading Data")
config = dotenv_values('../../.env')
train = pd.read_parquet(config["ENGINEERED_DATA"] + "viml_train_V1.parquet")

FEATURES = train.columns[:-1]

param = {
    "objective": "RMSE",
    "colsample_bylevel": 0.09885422567866216,
    "depth": 6,
    "boosting_type": "Plain",
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.609945866330414,
    "used_ram_limit": "3gb",
    "verbose": False,
}

print("Train Model")
model = CatBoostRegressor(**param)
model.fit(train[FEATURES], train["target"])

del train

print("Load Test Data and Predict")
test = pd.read_parquet(config["ENGINEERED_DATA"] + "viml_test_V1.parquet")
preds = model.predict(test[FEATURES])

del test

print("Saving")
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = preds

submit.to_csv(config["SUBMISSION_FOLDER"] + f'viml_cat.csv', index=None)