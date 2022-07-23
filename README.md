# UCSD Extension ML Capstone Project
> Author: Kaleb Ugalde

![American Express Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/American_Express_logo_(2018).svg/1200px-American_Express_logo_(2018).svg.png "American Express - Default Prediction")

## About the Competition
Given the data provided by American Express predict whether or not a customer will default on their credit card. Applications for this technology are risk management and creating the infrastructure for large scale data analytics.

### Format of the data
The features of the dataset are split up into 5 groups
- D_* = Delinquency variables
- S_* = Spend variables
- P_* = Payment variables
- B_* = Balance variables
- R_* = Risk variables

The following features are categorical while the rest are normalized continuous data:

` ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'] `

### Data Used During Development
I utilized the [AMEX Integer Parquet Dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format) by Raddar. This dataset reduces the size of the original dataset considerably. Not only does this make training much faster
## Technologies Utilized
- Python
- PySpark
- NumPy
- Matplotlib
- Scikit Learn
- XGBoost
- CatBoost

## Insights from the data
Gave up EDA in favor of looking at other people's EDA. 🙂

## Models Tested
#### Model 1: XGBoost Kaggle Starter
> Score: 0.793

I followed the [XGBoost Kaggle Starter](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793) by Chris Deotte as my first submission. I used this as a way to get a first submission in fast. I will use the general structure of the notebook as the basis for future models.

#### Model 2:

## Feature Engineering

#### Featuretools

#### Autofeat

#### Tsfresh

#### Stumpy