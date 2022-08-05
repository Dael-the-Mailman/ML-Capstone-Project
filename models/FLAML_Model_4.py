import pandas as pd
import numpy as np
import pickle
import json
from flaml import AutoML
from dotenv import dotenv_values

def amex_metric_mod(y_true, y_pred):
    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

if __name__ == "__main__":
    config = dotenv_values('../.env')
    
    # Part 1: Using train_fe and test_fe
    print("Loading Train FE Data")
    train = pd.read_parquet(config["ENGINEERED_DATA"] + "train_fe.parquet")
    FEATURES = train.columns[1:-1]
    X_train = train[FEATURES]
    y_train = train["target"]

    # Part 1A: Regression Model
    print("Training Regression Model")
    automl = AutoML()
    automl.fit(X_train, y_train, task="regression", time_budget=3600)

    print('Best Regression ML learner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print('Best loss on validation data: {0:.4g}'.format(automl.best_loss))
    print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
    
    print("Dumping hyperparameter json")
    with open(f"{automl.best_estimator}_regression_fe.json", "w") as fp:
        model = {automl.best_estimator : automl.best_config}
        json.dump(model, fp)

    print("Saving Model")
    reg_path = config["FLAML_MODELS"] + f'automl_regression_fe_{(automl.best_loss):.4g}.pkl'
    with open(reg_path, 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    # Part 1B: Extended Regression Model
    print("Loading Extended Train FE Data")
    train = pd.read_parquet(config["ENGINEERED_DATA"] + "train_fe.parquet")
    FEATURES = train.columns[1:-1]
    X_train = train[FEATURES]
    y_train = train["target"]

    print("Training Extended Regression Model")
    automl = AutoML()
    automl.fit(X_train, y_train, task="regression", time_budget=3600)

    print('Best Extended Regression ML learner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print('Best loss on validation data: {0:.4g}'.format(automl.best_loss))
    print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
    print("Dumping hyperparameter json")
    with open(f"{automl.best_estimator}_extended_regression_fe.json", "w") as fp:
        model = {automl.best_estimator : automl.best_config}
        json.dump(model, fp)

    print("Saving Model")
    ext_reg_path = config["FLAML_MODELS"] + f'automl_extended_regression_fe_{(1-automl.best_loss):.4g}.pkl'
    with open(ext_reg_path, 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    # Part 1C: Load and predict
    print("Load and predict test data")
    del train, X_train, y_train
    test = pd.read_parquet(config["ENGINEERED_DATA"] + "test_fe.parquet")
    test = test.drop("customer_ID", axis=1)
    for path in [ext_reg_path, reg_path]:
        with open(path, 'rb') as f:
            automl = pickle.load(f)
        sub = automl.predict(test)
        labels = pd.read_csv(config["SAMPLE_PATH"])
        labels["prediction"] = sub
        if path == ext_reg_path:
            print("Saving Extended Regression Submission")
            labels.to_csv(config["SUBMISSION_FOLDER"] + "flaml_extended_regression_fe_submission.csv", index=False)
        else:
            print("Saving Regression Submission")
            labels.to_csv(config["SUBMISSION_FOLDER"] + "flaml_regression_fe_submission.csv", index=False)
    del labels, sub