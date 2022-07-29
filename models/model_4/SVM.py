import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

from dotenv import dotenv_values
from sklearn.model_selection import KFold
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.svm import SVR, NuSVR, LinearSVR

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

if __name__ == '__main__':
    print("Read Train Data")
    config = dotenv_values('../../.env')
    train = pd.read_parquet(config["ENGINEERED_DATA"] + "default_train_features.parquet")

    print("Drop 80% NA and fill the rest")
    train=train.dropna(axis=1, thresh=int(0.80*len(train)))
    print(train.shape)

    train=train.set_index(['customer_ID'])
    train=train.ffill().bfill()
    train=train.reset_index()
    train=train.groupby('customer_ID').tail(1)
    train=train.set_index(['customer_ID'])
    
    print("Drop low correlation")
    train.drop(train.columns[train.corrwith(train['target']).abs()<0.3],axis=1,inplace=True)
    print(train.shape)
    # print(train.isnull().sum())
    train = train.sort_index().reset_index()
    FEATURES = train.columns[1:-1]

    oof = []

    skf = KFold(n_splits=5, shuffle=True, random_state=824)
    for fold,(train_idx, valid_idx) in enumerate(skf.split(
                train, train.target )):

        print('#'*25)
        print('### Fold',fold+1)
        print('### Train size',len(train_idx),'Valid size',len(valid_idx))
        # print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
        print('#'*25)

        X_train = train.loc[train_idx, FEATURES]
        y_train = train.loc[train_idx, 'target']
        X_valid = train.loc[valid_idx, FEATURES]
        y_valid = train.loc[valid_idx, 'target']

        model=LinearSVR(
            loss='squared_epsilon_insensitive',
            dual=False,
            C=y_train.mean()
        )
        # print(y_train.mean())
        # model=NuSVR(
        #     C=y_train.mean(),
        #     verbose=True,
        #     max_iter=1000
        # )
        model.fit(X_train, y_train)

        oof_preds = model.predict(X_valid)
        acc = amex_metric_mod(y_valid.values, oof_preds)
        print('Kaggle Metric =',acc,'\n')

        df = train.loc[valid_idx, ['customer_ID','target'] ].copy()
        df['oof_pred'] = oof_preds
        oof.append( df )

        del X_train, y_train, X_valid, y_valid, model, df
        _ = gc.collect()
    
    print('#'*25)
    oof = pd.concat(oof,axis=0,ignore_index=True).set_index('customer_ID')
    acc = amex_metric_mod(oof.target.values, oof.oof_pred.values)
    print('OVERALL CV Kaggle Metric =',acc)