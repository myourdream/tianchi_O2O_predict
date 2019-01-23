import datetime
import os
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
import xgboost as xgb


def get_processed_data():
    dataset1 = pd.read_csv('data_preprocessed_2/ProcessDataSet1.csv')
    dataset2 = pd.read_csv('data_preprocessed_2/ProcessDataSet2.csv')
    dataset3 = pd.read_csv('data_preprocessed_2/ProcessDataSet3.csv')

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset3.drop_duplicates(inplace=True)

    dataset12 = pd.concat([dataset1, dataset2], axis=0)

    dataset12.fillna(0, inplace=True)
    dataset3.fillna(0, inplace=True)

    dataset12.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Date', 'Coupon_id'],
        inplace=True)
    dataset3.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Coupon_id'], inplace=True)

    train_data, valid_data = train_test_split(dataset12, train_size=100000, random_state=0)
    _, valid_data = train_test_split(dataset12, random_state=0)
    # train_data_x = train_data.copy().drop(columns='Coupon_id')
    # train_data_y = train_data_x.pop('label')
    predict_data = dataset3

    # dataset12.to_csv(dataset12, index=False)
    #
    # dataset12_y = dataset12.label
    # dataset12_x = dataset12.drop(['user_id', 'label', 'day_gap_before', 'coupon_id', 'day_gap_after'], axis=1)
    #
    # dataset3.drop_duplicates(inplace=True)
    # dataset3_preds = dataset3[['user_id', 'coupon_id', 'date_received']]
    # dataset3_x = dataset3.drop(['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'], axis=1)

    # dataTrain = xgb.DMatrix(dataset12_x, label=dataset12_y)
    # dataTest = xgb.DMatrix(dataset3_x)
    return train_data, valid_data, predict_data


def train_xgb(train_data, valid_data, predict_data):
    params = {
        'learning_rate': 1e-2,
        'n_estimators': 1260,
        'max_depth': 8,
        'min_child_weight': 4,
        'gamma': .2,
        'subsample': .6,
        'colsample_bytree': .8,
        'scale_pos_weight': 1,
        'reg_alpha': 0,
        'n_jobs': cpu_jobs,
        'seed': 0
    }
    model = XGBClassifier().set_params(**params)

    train_data_x = train_data.drop('label')
    train_data_y = train_data.pop('label')
    model = model.fit(train_data_x, train_data_y, eval_metric='auc')

    valid_data_x = valid_data.copy().drop(columns='Coupon_id')
    valid_data_y = valid_data.pop('label')
    valid_data_y, valid_data_y_predict = valid_data_y, model.predict(valid_data_x)
    return model


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    train_data, valid_data, predict_data = get_processed_data()
    # analysis()
    # detect_duplicate_columns()
    # feature_importance_score()

    # grid_search_gbdt()
    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    train_xgb(train_data, valid_data, predict_data)

    print('predict: start predicting......')
    # predict('xgb')
    print('predict: predicting finished.')

    log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    log += '----------------------------------------------------\n'
    open('%s.log' % os.path.basename(__file__), 'a').write(log)
    print(log)
