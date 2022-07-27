'''
Derived from https://www.kaggle.com/code/finlay/amex-rank-ensemble
'''
import numpy as np 
import pandas as pd 

from dotenv import dotenv_values
from scipy.stats import rankdata

DESC = ""

# Read configuration file
config = dotenv_values('.env')

# Gather all submissions to ensemble
paths = []
dfs = [pd.read_csv(x) for x in paths]
dfs = [x.sort_values(by='customer_ID') for x in dfs]

# Average ensemble
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = 0

for df in dfs:
    submit['prediction'] += df['prediction']
    
submit['prediction'] /= 4

submit.to_csv(config["SUBMISSION_FOLDER"] + f'mean_submission_{DESC}.csv', index=None)

# Rank ensemble
submit = pd.read_csv(config["SAMPLE_PATH"])
submit['prediction'] = 0

for df in dfs:
    submit['prediction'] += rankdata(df['prediction'])/df.shape[0]
    
submit['prediction'] /= 4

submit.to_csv(config["SUBMISSION_FOLDER"] + f'rank_submission_{DESC}.csv', index=None)