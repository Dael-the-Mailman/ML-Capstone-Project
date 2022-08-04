import pandas as pd
from dotenv import dotenv_values

config = dotenv_values('.env')
df = pd.read_csv(config["ENGINEERED_DATA"] + "test_fe.csv")
print(df.head())
df = df.drop("Unnamed: 0", axis=1)
print(df.head())
df.to_csv(config["ENGINEERED_DATA"] + "test_fe.csv",index=False)