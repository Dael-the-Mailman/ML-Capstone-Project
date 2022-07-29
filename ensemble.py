'''
Derived from https://www.kaggle.com/code/finlay/amex-rank-ensemble
'''
import numpy as np 
import pandas as pd 

from dotenv import dotenv_values
from scipy.stats import rankdata

DESC = "lowered_ensemble_size"

# Read configuration file
config = dotenv_values('.env')


# Gather all submissions to ensemble
paths = [config["OTHER_SUBMISSIONS"] + "lgbm_fgen_submission.csv",
         config["OTHER_SUBMISSIONS"] + "best_of_both_worlds_submission.csv"]
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

submit.to_csv(config["SUBMISSION_FOLDER"] + f'mean_submission_{DESC}.csv', index=None)

# Rank ensemble
print("Rank Ensemble")
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = 0

for df in dfs:
    submit['prediction'] += rankdata(df['prediction'])/df.shape[0]
    
submit['prediction'] /= len(paths)

submit.to_csv(config["SUBMISSION_FOLDER"] + f'rank_submission_{DESC}.csv', index=None)