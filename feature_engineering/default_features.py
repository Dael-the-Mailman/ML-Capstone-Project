import pandas as pd
import numpy as np

from dotenv import dotenv_values

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


if __name__ == '__main__':
    config = dotenv_values('../.env')
    
    # Load Target Data
    print("Loading Target Data")
    targets = pd.read_csv(config["TRAIN_LABELS_PATH"])
    targets = targets.set_index('customer_ID')

    # Process Training Data
    print("Processing Training Data")
    train = pd.read_parquet(config["INT_TRAIN_PARQUET"])
    train = process_and_feature_engineer(train)
    train = train.merge(targets, left_index=True, right_index=True, how='left')
    train = train.sort_index().reset_index()
    train.to_parquet(config["ENGINEERED_DATA"] + "default_train_features.parquet")
    del train

    # Process Testing Data
    print("Processing Test Data")
    test = pd.read_parquet(config["INT_TEST_PARQUET"])
    test = process_and_feature_engineer(test)
    test = test.merge(targets, left_index=True, right_index=True, how='left')
    test = test.sort_index().reset_index()
    test.to_parquet(config["ENGINEERED_DATA"] + "default_test_features.parquet")
    del test

