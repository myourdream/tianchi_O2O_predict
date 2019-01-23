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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve


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

    # dataset12_x = dataset12.copy().drop(
    #     columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
    #              'Date', 'Coupon_id', 'label'], inplace=True, axis=1)
    # dataset3_x = dataset3.copy().drop(
    #     columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
    #              'Coupon_id'], inplace=True, axis=1)
    #
    # train_dmatrix = xgb.DMatrix(dataset12_x, label=dataset12.label)
    # predict_dmatrix = xgb.DMatrix(dataset3_x)

    # train_data, valid_data = train_test_split(dataset12, train_size=100000, random_state=0)
    # _, valid_data = train_test_split(dataset12, random_state=0)
    # # train_data_x = train_data.copy().drop(columns='Coupon_id')
    # # train_data_y = train_data_x.pop('label')
    # predict_data = dataset3

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
    # return train_dmatrix, predict_dmatrix, predict_dataset, dataset12, dataset12_x
    return dataset12, dataset3


def train_xgb(dataset12, dataset3):

    predict_dataset = dataset3[['User_id', 'Coupon_id', 'Date_received']].copy()
    predict_dataset.Date_received = pd.to_datetime(predict_dataset.Date_received, format='%Y-%m-%d')
    predict_dataset.Date_received = predict_dataset.Date_received.dt.strftime('%Y%m%d')

    # 将数据转化为dmatric格式
    dataset12_x = dataset12.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Date', 'Coupon_id', 'label'], axis=1)
    dataset3_x = dataset3.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Coupon_id'], axis=1)

    train_dmatrix = xgb.DMatrix(dataset12_x, label=dataset12.label)
    predict_dmatrix = xgb.DMatrix(dataset3_x)

    # xgboost模型训练
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': cpu_jobs
              }

    # 使用xgb.cv优化num_boost_round参数
    # cvresult = xgb.cv(params, train_dmatrix, num_boost_round=8000, nfold=2, metrics='auc', seed=0, callbacks=[
    #     xgb.callback.print_evaluation(show_stdv=False),
    #     xgb.callback.early_stop(50)
    # ])
    # num_round_best = cvresult.shape[0] - 1
    # print('Best round num: ', num_round_best)

    # 使用优化后的num_boost_round参数训练模型
    watchlist = [(train_dmatrix, 'train')]
    model = xgb.train(params, train_dmatrix, num_boost_round=4500, evals=watchlist)

    model.save_model('train_dir_2/xgbmodel')
    model = xgb.Booster(params)
    model.load_model('train_dir_2/xgbmodel')

    # predict test set
    dataset3_predict = predict_dataset.copy()
    dataset3_predict['label'] = model.predict(predict_dmatrix)

    # 标签归一化
    dataset3_predict.label = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
        dataset3_predict.label.values.reshape(-1, 1))
    dataset3_predict.sort_values(by=['Coupon_id', 'label'], inplace=True)
    dataset3_predict.to_csv("train_dir_2/xgb_preds.csv", index=None, header=None)
    print(dataset3_predict.describe())

    # 在dataset12上计算auc
    model = xgb.Booster()
    model.load_model('train_dir_2/xgbmodel')

    temp = dataset12[['Coupon_id', 'label']].copy()
    temp['pred'] = model.predict(xgb.DMatrix(dataset12_x))
    temp.pred = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(temp['pred'].values.reshape(-1, 1))
    print(myauc(temp))


def train_xgb_classifier(dataset12, dataset3):

    predict_dataset = dataset3[['User_id', 'Coupon_id', 'Date_received']].copy()
    predict_dataset.Date_received = pd.to_datetime(predict_dataset.Date_received, format='%Y-%m-%d')
    predict_dataset.Date_received = predict_dataset.Date_received.dt.strftime('%Y%m%d')

    # 获取trainx，y，testx，y
    dataset12_x = dataset12.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Date', 'Coupon_id', 'label'], axis=1)
    dataset3_x = dataset3.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Coupon_id'], axis=1)

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


# 性能评价函数
def myauc(test):
    testgroup = test.groupby(['Coupon_id'])
    aucs = []
    for i in testgroup:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    return np.average(aucs)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    # log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    dataset12, dataset3 = get_processed_data()
    # analysis()
    # detect_duplicate_columns()
    # feature_importance_score()

    # grid_search_gbdt()
    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    train_xgb(dataset12, dataset3)

    # print('predict: start predicting......')
    # # predict('xgb')
    # print('predict: predicting finished.')

    # log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    # log += '----------------------------------------------------\n'
    # open('%s.log' % os.path.basename(__file__), 'a').write(log)
    # print(log)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %s s' % (datetime.datetime.now() - start).seconds)
