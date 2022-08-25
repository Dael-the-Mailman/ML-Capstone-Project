# UCSD Extension ML Capstone Project
> Author: Kaleb Ugalde

> Rank: 1272/4935 

![American Express Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/American_Express_logo_(2018).svg/1200px-American_Express_logo_(2018).svg.png "American Express - Default Prediction")

## About the Competition
Given the data provided by American Express predict whether or not a customer will default on their credit card. Applications for this technology are risk management and creating the infrastructure for large scale data analytics.

## About the Application
==TODO==

## Table of Contents
==TODO==

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
I utilized the [AMEX Integer Parquet Dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format) by Raddar. This dataset reduces the size of the original dataset considerably. Not only does this make training much faster but it saves on space as well. Performance was also increased as when I tested an XGBoost model on the original dataset the model performed about 30% worse.
## Technologies Utilized
#### Machine Learning Technologies
- Python
- PySpark
- NumPy
- Matplotlib
- Scikit Learn
- XGBoost
- CatBoost

#### Cloud Technologies
- Databricks?
- AWS?

#### Front End Technologies
- React?
- Flutter?

#### Back End Technologies
- ???

## Insights from the data
Gave up EDA in favor of looking at other people's EDA. 🙂

## Models Tested
#### Model 1: XGBoost Kaggle Starter
> Score: 0.793

I followed the [XGBoost Kaggle Starter](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793) by Chris Deotte as my first submission. I used this as a way to get a first submission in fast. I will use the general structure of the notebook as the basis for future models.

#### Model 2: CatBoost Model
> Score: 0.792

I used the same features as the [XGBoost Kaggle Starter](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793) but replaced XGBoost with CatBoost. To my surprise the CatBoost model performed worse than the XGBoost model. I tried the default CatBoost model, CatBoostClassifier, and CatBoostRegressor models without any hyperparameter tuning. None of these models outperformed XGBoost. Perhaps with hyperparameter tuning they will work better but as of now I want to focus on engineering more than choosing models. 

#### Model 3: Rank and Mean Ensemble
> Score: 0.799

I used the data from [AMEX Rank Ensemble](https://www.kaggle.com/code/finlay/amex-rank-ensemble) by MAXXX and submitted his output to the competition. This made my position jump from 1823 to 257 at the time of writing this entry. I didn't write the code for this since the code seems pretty simple. This inspires me to look into ensemble methods for drastic increases in performance. I will look further into how people ensemble data in this competition and other competitions.

#### Model 4: FLAML AutoML
> Score: 0.784

I used the Fast Library for Automated Machine Learning & Tuning(FLAML) by Microsoft. I tested other local AutoML libraries like AutoGluon and EvalML. Both libraries ran out of memory and failed to train some or all models. FLAML also performs hyperparameter tuning for the models as well. One observation is that the model almost always settles on LGBM after an hour of train time.

#### Model 5: FLAML + Extended Training Set
> Score: 0.725

The extended training set was created using the top scoring submissions from the kaggle forums. I find which customer_ID's have over 90% consensus between the top submissions and concatenated them to the training dataset. The idea came from a podcast episode between Demis Hassabis and Lex Fridman and I wanted to see if there would be a performance increase. Even though I didn't see a performance increase in the FLAML library, the library did say that if given more time it might perform better.

#### Model 6: AutoVIML + Hyperparameter Tuning
> Score: 0.610

Model 6 ended up as an ultimate failure as the model seemed to overfit the data it was trained on. This will be my final model for the competition.

