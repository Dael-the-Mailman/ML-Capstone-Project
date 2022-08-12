import pandas as pd
import numpy as np
import pickle
import json
from flaml import AutoML
from dotenv import dotenv_values
from scipy.stats import rankdata

if __name__ == '__main__':
    config = dotenv_values('../.env')
    # print("Load and predict test data")
    # test = pd.read_parquet(config["ENGINEERED_DATA"] + "test_fe.parquet")
    # test = test.drop("customer_ID", axis=1)

    # M1_PATH = config["FLAML_MODELS"] + f'extended_regression_fe_FOLD_1_0.1733.pkl'
    # M2_PATH = config["FLAML_MODELS"] + f'extended_regression_fe_FOLD_2_0.1959.pkl'

    # with open(M1_PATH, 'rb') as f:
    #     model1 = pickle.load(f)

    # with open(M2_PATH, 'rb') as f:
    #     model2 = pickle.load(f)

    # sub1 = model1.predict(test)
    # sub2 = model2.predict(test)
    # labels = pd.read_csv(config["SAMPLE_PATH"])

    # # Save First Model Submission
    # labels["prediction"] = sub1
    # print("Saving Model 1 Submission")
    # labels.to_csv(config["SUBMISSION_FOLDER"] + "flaml_extended_model_1_submission.csv", index=False)

    # # Save Second Model Submission
    # labels["prediction"] = sub2
    # print("Saving Model 2 Submission")
    # labels.to_csv(config["SUBMISSION_FOLDER"] + "flaml_extended_model_2_submission.csv", index=False)

    # del model1, model2, labels, sub1, sub2

    # Ensembles
    dfs = [pd.read_csv(config["SUBMISSION_FOLDER"] + "flaml_extended_model_1_submission.csv"),
           pd.read_csv(config["SUBMISSION_FOLDER"] + "flaml_extended_model_2_submission.csv")]
    dfs = [x.sort_values(by='customer_ID') for x in dfs]

    submit = pd.read_csv(config["SAMPLE_PATH"])
    submit['prediction'] = 0

    for df in dfs:
        df['prediction'] = np.clip(df['prediction'], 0, 1)

    # Save Equal Weighted Mean Ensemble
    print("Saving Mean Ensemble")
    for df in dfs:
        submit['prediction'] += df['prediction']
        
    submit['prediction'] /= 2
    submit.to_csv(config["SUBMISSION_FOLDER"] + "mean_extended_fold_ensemble.csv", index=False)

    # Save Rank Ensemble
    print("Saving Rank Ensemble")
    for df in dfs:
        submit['prediction'] += rankdata(df['prediction'])/df.shape[0]
        
    submit['prediction'] /= 2
    submit.to_csv(config["SUBMISSION_FOLDER"] + "rank_extended_fold_ensemble.csv", index=False)

    # Save Weighted Mean 
    print("Saving Weighted Ensemble")
    weights = [0.67, 0.33]
    for df, weight in zip(dfs, weights):
        submit['prediction'] += (df['prediction'] * weight)
    submit.to_csv(config["SUBMISSION_FOLDER"] + "weighted_extended_fold_ensemble.csv", index=False)
    
