import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from autoviml.Auto_ViML import Auto_ViML
from dotenv import dotenv_values

if __name__ == '__main__':
    config = dotenv_values('../../.env')
    train = pd.read_parquet(config["ENGINEERED_DATA"] + "train_fe.parquet")
    test = pd.read_parquet(config["ENGINEERED_DATA"] + "test_fe.parquet")

    target = 'target'
    sample_submission = pd.read_csv(config["SAMPLE_PATH"])
    scoring_parameter = 'roc_auc'
    VERSION = 1

    model, features, trainm, testm = Auto_ViML(
        train,
        target,
        test,
        sample_submission,
        hyper_param="RS",
        feature_reduction=True,
        scoring_parameter=scoring_parameter,
        KMeans_Featurizer=True,
        Boosting_Flag='catboost',
        Binning_Flag=True,
        Add_Poly=3,
        Stacking_Flag=True,
        Imbalanced_Flag=True,
        verbose=0,
    )

    model_path = config["FLAML_MODELS"] + f'viml_V{VERSION}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    
    trainm.to_csv(config["ENGINEERED_DATA"] + f"viml_train_V{VERSION}.csv", index=False)
    testm.to_csv(config["ENGINEERED_DATA"] + f"viml_test_V{VERSION}.csv", index=False)