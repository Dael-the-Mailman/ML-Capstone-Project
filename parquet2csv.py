import pandas as pd
from dotenv import dotenv_values

config = dotenv_values(".env")

FILENAME = "extended_train_fe"
df = pd.read_parquet(config["ENGINEERED_DATA"] + f"{FILENAME}.parquet")
df.to_csv(config["ENGINEERED_DATA"] + f"{FILENAME}.csv", index=False)