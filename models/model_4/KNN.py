import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

from dotenv import dotenv_values
from sklearn.model_selection import KFold
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

SEED = 42
FOLDS = 5
NAN_VALUE = -127

def read_file(path='', usecols=None):
    if usecols is not None: df = pd.read_parquet(path, columns=usecols)
    else: df = pd.read_parquet(path)

    df.S_2 = pd.to_datetime(df.S_2)
    df = df.fillna(NAN_VALUE)

    print('Shape of Data:', df.shape)

    return df

def process_and_feature_engineer(df):
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2','target']]
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    num_features = [col for col in all_cols if col not in cat_features]
    
    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
    
    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    df = pd.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    print('shape after engineering', df.shape )
    
    return df

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

def get_rows(customers, test, NUM_PARTS = 4, verbose = ''):
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        print(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        print(f'There will be {chunk} customers in each part (except the last part).')
        print('Below are number of rows in each part:')
    rows = []

    for k in range(NUM_PARTS):
        if k==NUM_PARTS-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = test.loc[test.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows,chunk

if __name__ == '__main__':
    config = dotenv_values('../../.env')
    train = pd.read_parquet(config["INT_TRAIN_PARQUET"])

    # Process training data
    train = process_and_feature_engineer(train)

    # Load Target Data
    targets = pd.read_csv(config["TRAIN_LABELS_PATH"])
    targets = targets.set_index('customer_ID')
    train = train.merge(targets, left_index=True, right_index=True, how='left')
    del targets

    # Obtain feature columns and reset index
    train = train.sort_index().reset_index()
    FEATURES = train.columns[1:-1]

    oof = []
    TRAIN_SUBSAMPLE = 1.0
    gc.collect()
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold,(train_idx, valid_idx) in enumerate(skf.split(
                train, train.target )):
        print('#'*25)
        print('### Fold',fold+1)
        print('### Train size',len(train_idx),'Valid size',len(valid_idx))
        print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
        print('#'*25)

        X_train = train.loc[train_idx, FEATURES]
        y_train = train.loc[train_idx, 'target']
        X_valid = train.loc[valid_idx, FEATURES]
        y_valid = train.loc[valid_idx, 'target']

        model = KNeighborsClassifier(
            n_neighbors=40,
            weights='distance',
            n_jobs=-1
        ).fit(X_train, y_train)

        oof_preds = model.predict(X_valid)
        acc = amex_metric_mod(y_valid.values, oof_preds)
        print('Kaggle Metric =',acc,'\n')
