import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

import autokeras as ak

house_dataset = fetch_california_housing()
df = pd.DataFrame(
    np.concatenate(
        (house_dataset.data, house_dataset.target.reshape(-1,1)), axis=1
    ),
    columns=house_dataset.feature_names + ["Price"],
)

train_size = int(df.shape[0] * 0.9)
df[:train_size].to_csv("train.csv", index=False)
df[train_size:].to_csv("eval.csv", index=False)
train_file_path = "train.csv"
test_file_path = "eval.csv"

reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=3
)

reg.fit(train_file_path, "Price", epochs=10)

predicted_y = reg.predict(test_file_path)
print(reg.evaluate(test_file_path, "Price"))