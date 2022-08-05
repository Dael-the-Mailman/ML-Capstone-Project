import pandas as pd
from dotenv import dotenv_values

config = dotenv_values('.env')
df = pd.read_csv(config["SUBMISSION_FOLDER"] + "flaml_regression_fe_submission.csv")
print(df.head())
df = df.drop("Unnamed: 0", axis=1)
print(df.head())
df.to_csv(config["SUBMISSION_FOLDER"] + "flaml_regression_fe_submission.csv",index=False)