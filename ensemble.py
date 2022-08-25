'''
Derived from https://www.kaggle.com/code/finlay/amex-rank-ensemble
'''
import numpy as np 
import pandas as pd 
import os

from dotenv import dotenv_values
from scipy.stats import rankdata

DESC = "V1"

# Read configuration file
config = dotenv_values('.env')


# Gather all submissions to ensemble
# paths = [config["SUBMISSION_FOLDER"] + x for x in os.listdir(config["OTHER_SUBMISSIONS"])]
paths = [config["SUBMISSION_FOLDER"] + "viml_cat.csv",
         config["SUBMISSION_FOLDER"] + "viml_xgb.csv",
         config["SUBMISSION_FOLDER"] + "viml_lgbm.csv"]
print("Read and Load DataFrames")
dfs = [pd.read_csv(x) for x in paths]
dfs = [x.sort_values(by='customer_ID') for x in dfs]

# Average ensemble
print("Average Ensemble")
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = 0

for df in dfs:
    submit['prediction'] += df['prediction']
    
submit['prediction'] /= len(paths)

submit.to_csv(config["SUBMISSION_FOLDER"] + f'viml_mean_{DESC}.csv', index=None)

# Rank ensemble
print("Rank Ensemble")
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = 0

for df in dfs:
    submit['prediction'] += rankdata(df['prediction'])/df.shape[0]
    
submit['prediction'] /= len(paths)

submit.to_csv(config["SUBMISSION_FOLDER"] + f'viml_rank_{DESC}.csv', index=None)