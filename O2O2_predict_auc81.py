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

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


def drop_columns(X, predict=False):
    columns = [
        'User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
        # 'u33', 'u34'
    ]

    if predict:
        columns.append('Coupon_id')
    else:
        columns.append('Date')

    X.drop(columns=columns, inplace=True)


def get_preprocess_data(predict=False):
    if predict:
        offline = pd.read_csv('dataset/ccf_offline_stage1_test_revised.csv', parse_dates=['Date_received'])
    else:
        offline = pd.read_csv('dataset/ccf_offline_stage1_train.csv', parse_dates=['Date_received', 'Date'])

    offline.Distance.fillna(11, inplace=True)
    offline.Distance = offline.Distance.astype(int)
    offline.Coupon_id.fillna(0, inplace=True)
    offline.Coupon_id = offline.Coupon_id.astype(int)
    offline.Date_received.fillna(date_null, inplace=True)

    offline[['discount_rate_x', 'discount_rate_y']] = offline[offline.Discount_rate.str.contains(':') == True][
        'Discount_rate'].str.split(':', expand=True).astype(int)
    offline['discount_rate'] = 1 - offline.discount_rate_y / offline.discount_rate_x
    offline.discount_rate = offline.discount_rate.fillna(offline.Discount_rate).astype(float)

    if predict:
        return offline

    offline.Date.fillna(date_null, inplace=True)

    # online
    online = pd.read_csv('dataset/ccf_online_stage1_train.csv', parse_dates=['Date_received', 'Date'])

    online.Coupon_id.fillna(0, inplace=True)
    # online.Coupon_id = online.Coupon_id.astype(int)
    online.Date_received.fillna(date_null, inplace=True)
    online.Date.fillna(date_null, inplace=True)

    return offline, online


def task(X_chunk, X, counter):
    print(counter, end=',', flush=True)
    X_chunk = X_chunk.copy()

    X_chunk['o17'] = -1
    X_chunk['o18'] = -1

    for i, user in X_chunk.iterrows():
        temp = X[X.User_id == user.User_id]

        temp1 = temp[temp.Date_received < user.Date_received]
        temp2 = temp[temp.Date_received > user.Date_received]

        # 用户此次之后/前领取的所有优惠券数目
        X_chunk.loc[i, 'o3'] = len(temp1)
        X_chunk.loc[i, 'o4'] = len(temp2)

        # 用户此次之后/前领取的特定优惠券数目
        X_chunk.loc[i, 'o5'] = len(temp1[temp1.Coupon_id == user.Coupon_id])
        X_chunk.loc[i, 'o6'] = len(temp2[temp2.Coupon_id == user.Coupon_id])

        # 用户上/下一次领取的时间间隔
        temp1 = temp1.sort_values(by='Date_received', ascending=False)
        if len(temp1):
            X_chunk.loc[i, 'o17'] = (user.Date_received - temp1.iloc[0].Date_received).days

        temp2 = temp2.sort_values(by='Date_received')
        if len(temp2):
            X_chunk.loc[i, 'o18'] = (temp2.iloc[0].Date_received - user.Date_received).days

    return X_chunk


def get_offline_features(X, offline):
    # X = X[:1000]

    print(len(X), len(X.columns))

    temp = offline[offline.Coupon_id != 0]
    coupon_consume = temp[temp.Date != date_null]
    coupon_no_consume = temp[temp.Date == date_null]

    user_coupon_consume = coupon_consume.groupby('User_id')

    X['weekday'] = X.Date_received.dt.weekday
    X['day'] = X.Date_received.dt.day

    # # 距离优惠券消费次数
    # temp = coupon_consume.groupby('Distance').size().reset_index(name='distance_0')
    # X = pd.merge(X, temp, how='left', on='Distance')
    #
    # # 距离优惠券不消费次数
    # temp = coupon_no_consume.groupby('Distance').size().reset_index(name='distance_1')
    # X = pd.merge(X, temp, how='left', on='Distance')
    #
    # # 距离优惠券领取次数
    # X['distance_2'] = X.distance_0 + X.distance_1
    #
    # # 距离优惠券消费率
    # X['distance_3'] = X.distance_0 / X.distance_2

    # temp = coupon_consume[coupon_consume.Distance != 11].groupby('Distance').size()
    # temp['d4'] = temp.Distance.sum() / len(temp)
    # X = pd.merge(X, temp, how='left', on='Distance')

    '''user features'''

    # 优惠券消费次数
    temp = user_coupon_consume.size().reset_index(name='u2')
    X = pd.merge(X, temp, how='left', on='User_id')
    # X.u2.fillna(0, inplace=True)
    # X.u2 = X.u2.astype(int)

    # 优惠券不消费次数
    temp = coupon_no_consume.groupby('User_id').size().reset_index(name='u3')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 使用优惠券次数与没使用优惠券次数比值
    X['u19'] = X.u2 / X.u3

    # 领取优惠券次数
    X['u1'] = X.u2.fillna(0) + X.u3.fillna(0)

    # 优惠券核销率
    X['u4'] = X.u2 / X.u1

    # 普通消费次数
    temp = offline[(offline.Coupon_id == 0) & (offline.Date != date_null)]
    temp1 = temp.groupby('User_id').size().reset_index(name='u5')
    X = pd.merge(X, temp1, how='left', on='User_id')

    # 一共消费多少次
    X['u25'] = X.u2 + X.u5

    # 用户使用优惠券消费占比
    X['u20'] = X.u2 / X.u25

    # 正常消费平均间隔
    temp = pd.merge(temp, temp.groupby('User_id').Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('User_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('User_id').size().reset_index(name='len'))
    temp['u6'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('User_id')
    X = pd.merge(X, temp[['User_id', 'u6']], how='left', on='User_id')

    # 优惠券消费平均间隔
    temp = pd.merge(coupon_consume, user_coupon_consume.Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('User_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('User_id').size().reset_index(name='len'))
    temp['u7'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('User_id')
    X = pd.merge(X, temp[['User_id', 'u7']], how='left', on='User_id')

    # 15天内平均会普通消费几次
    X['u8'] = X.u6 / 15

    # 15天内平均会优惠券消费几次
    X['u9'] = X.u7 / 15

    # 领取优惠券到使用优惠券的平均间隔时间
    temp = coupon_consume.copy()
    temp['days'] = (temp.Date - temp.Date_received).dt.days
    temp = (temp.groupby('User_id').days.sum() / temp.groupby('User_id').size()).reset_index(name='u10')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 在15天内使用掉优惠券的值大小
    X['u11'] = X.u10 / 15

    # 领取优惠券到使用优惠券间隔小于15天的次数
    temp = coupon_consume.copy()
    temp['days'] = (temp.Date - temp.Date_received).dt.days
    temp = temp[temp.days <= 15]
    temp = temp.groupby('User_id').size().reset_index(name='u21')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户15天使用掉优惠券的次数除以使用优惠券的次数
    X['u22'] = X.u21 / X.u2

    # 用户15天使用掉优惠券的次数除以领取优惠券未消费的次数
    X['u23'] = X.u21 / X.u3

    # 用户15天使用掉优惠券的次数除以领取优惠券的总次数
    X['u24'] = X.u21 / X.u1

    # 消费优惠券的平均折率
    temp = user_coupon_consume.discount_rate.mean().reset_index(name='u45')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券的最低消费折率
    temp = user_coupon_consume.discount_rate.min().reset_index(name='u27')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券的最高消费折率
    temp = user_coupon_consume.discount_rate.max().reset_index(name='u28')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销过的不同优惠券数量
    temp = coupon_consume.groupby(['User_id', 'Coupon_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='u32')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户领取所有不同优惠券数量
    temp = offline[offline.Date_received != date_null]
    temp = temp.groupby(['User_id', 'Coupon_id']).size().reset_index(name='u47')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Coupon_id'])

    # 用户核销过的不同优惠券数量占所有不同优惠券的比重
    X['u33'] = X.u32 / X.u47

    # 用户平均每种优惠券核销多少张
    X['u34'] = X.u2 / X.u47

    # 核销优惠券用户-商家平均距离
    temp = offline[(offline.Coupon_id != 0) & (offline.Date != date_null) & (offline.Distance != 11)]
    temp = temp.groupby('User_id').Distance
    temp = pd.merge(temp.count().reset_index(name='x'), temp.sum().reset_index(name='y'), on='User_id')
    temp['u35'] = temp.y / temp.x
    temp = temp[['User_id', 'u35']]
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券中的最小用户-商家距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('User_id').Distance.min().reset_index(name='u36')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券中的最大用户-商家距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('User_id').Distance.max().reset_index(name='u37')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 优惠券类型
    discount_types = [
        '0.2', '0.5', '0.6', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '30:20', '50:30', '10:5',
        '20:10', '100:50', '200:100', '50:20', '30:10', '150:50', '100:30', '20:5', '200:50', '5:1',
        '50:10', '100:20', '150:30', '30:5', '300:50', '200:30', '150:20', '10:1', '50:5', '100:10',
        '200:20', '300:30', '150:10', '300:20', '500:30', '20:1', '100:5', '200:10', '30:1', '150:5',
        '300:10', '200:5', '50:1', '100:1',
    ]
    X['discount_type'] = -1
    for k, v in enumerate(discount_types):
        X.loc[X.Discount_rate == v, 'discount_type'] = k

    # 不同优惠券领取次数
    temp = offline.groupby(['User_id', 'Discount_rate']).size().reset_index(name='u41')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Discount_rate'])

    # 不同优惠券使用次数
    temp = coupon_consume.groupby(['User_id', 'Discount_rate']).size().reset_index(name='u42')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Discount_rate'])

    # 不同优惠券不使用次数
    temp = coupon_no_consume.groupby(['User_id', 'Discount_rate']).size().reset_index(name='u43')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Discount_rate'])

    # 不同打折优惠券使用率
    X['u44'] = X.u42 / X.u41

    # 满减类型优惠券领取次数
    temp = offline[offline.Discount_rate.str.contains(':') == True]
    temp = temp.groupby('User_id').size().reset_index(name='u48')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 打折类型优惠券领取次数
    temp = offline[offline.Discount_rate.str.contains('\.') == True]
    temp = temp.groupby('User_id').size().reset_index(name='u49')
    X = pd.merge(X, temp, how='left', on='User_id')

    '''offline merchant features'''

    # 商户消费次数
    temp = offline[offline.Date != date_null].groupby('Merchant_id').size().reset_index(name='m0')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券被领取后核销次数
    temp = coupon_consume.groupby('Merchant_id').size().reset_index(name='m1')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商户正常消费笔数
    X['m2'] = X.m0.fillna(0) - X.m1.fillna(0)

    # 商家优惠券被领取次数
    temp = offline[offline.Date_received != date_null].groupby('Merchant_id').size().reset_index(name='m3')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券被领取后核销率
    X['m4'] = X.m1 / X.m3

    # 商家优惠券被领取后不核销次数
    temp = coupon_no_consume.groupby('Merchant_id').size().reset_index(name='m7')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商户当天优惠券领取次数
    temp = X[X.Date_received != date_null]
    temp = temp.groupby(['Merchant_id', 'Date_received']).size().reset_index(name='m5')
    X = pd.merge(X, temp, how='left', on=['Merchant_id', 'Date_received'])

    # 商户当天优惠券领取人数
    temp = X[X.Date_received != date_null]
    temp = temp.groupby(['User_id', 'Merchant_id', 'Date_received']).size().reset_index()
    temp = temp.groupby(['Merchant_id', 'Date_received']).size().reset_index(name='m6')
    X = pd.merge(X, temp, how='left', on=['Merchant_id', 'Date_received'])

    # 商家优惠券核销的平均消费折率
    temp = coupon_consume.groupby('Merchant_id').discount_rate.mean().reset_index(name='m8')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券核销的最小消费折率
    temp = coupon_consume.groupby('Merchant_id').discount_rate.max().reset_index(name='m9')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券核销的最大消费折率
    temp = coupon_consume.groupby('Merchant_id').discount_rate.min().reset_index(name='m10')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券核销不同的用户数量
    temp = coupon_consume.groupby(['Merchant_id', 'User_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='m11')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券领取不同的用户数量
    temp = offline[offline.Date_received != date_null].groupby(['Merchant_id', 'User_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='m12')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 核销商家优惠券的不同用户数量其占领取不同的用户比重
    X['m13'] = X.m11 / X.m12

    # 商家优惠券平均每个用户核销多少张
    X['m14'] = X.m1 / X.m12

    # 商家被核销过的不同优惠券数量
    temp = coupon_consume.groupby(['Merchant_id', 'Coupon_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='m15')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家领取过的不同优惠券数量的比重
    temp = offline[offline.Date_received != date_null].groupby(['Merchant_id', 'Coupon_id']).size()
    temp = temp.groupby('Merchant_id').count().reset_index(name='m18')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
    X['m19'] = X.m15 / X.m18

    # 商家被核销优惠券的平均时间
    temp = pd.merge(coupon_consume, coupon_consume.groupby('Merchant_id').Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('Merchant_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('Merchant_id').size().reset_index(name='len'))
    temp['m20'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('Merchant_id')
    X = pd.merge(X, temp[['Merchant_id', 'm20']], how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家平均距离
    temp = coupon_consume[coupon_consume.Distance != 11].groupby('Merchant_id').Distance
    temp = pd.merge(temp.count().reset_index(name='x'), temp.sum().reset_index(name='y'), on='Merchant_id')
    temp['m21'] = temp.y / temp.x
    temp = temp[['Merchant_id', 'm21']]
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家最小距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('Merchant_id').Distance.min().reset_index(name='m22')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家最大距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('Merchant_id').Distance.max().reset_index(name='m23')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    """offline coupon features"""

    # 此优惠券一共发行多少张
    temp = offline[offline.Coupon_id != 0].groupby('Coupon_id').size().reset_index(name='c1')
    X = pd.merge(X, temp, how='left', on='Coupon_id')

    # 此优惠券一共被使用多少张
    temp = coupon_consume.groupby('Coupon_id').size().reset_index(name='c2')
    X = pd.merge(X, temp, how='left', on='Coupon_id')

    # 优惠券使用率
    X['c3'] = X.c2 / X.c1

    # 没有使用的数目
    X['c4'] = X.c1 - X.c2

    # 此优惠券在当天发行了多少张
    temp = X.groupby(['Coupon_id', 'Date_received']).size().reset_index(name='c5')
    X = pd.merge(X, temp, how='left', on=['Coupon_id', 'Date_received'])

    # 优惠券类型(直接优惠为0, 满减为1)
    X['c6'] = 0
    X.loc[X.Discount_rate.str.contains(':') == True, 'c6'] = 1

    # 不同打折优惠券领取次数
    temp = offline.groupby('Discount_rate').size().reset_index(name='c8')
    X = pd.merge(X, temp, how='left', on='Discount_rate')

    # 不同打折优惠券使用次数
    temp = coupon_consume.groupby('Discount_rate').size().reset_index(name='c9')
    X = pd.merge(X, temp, how='left', on='Discount_rate')

    # 不同打折优惠券不使用次数
    temp = coupon_no_consume.groupby('Discount_rate').size().reset_index(name='c10')
    X = pd.merge(X, temp, how='left', on='Discount_rate')

    # 不同打折优惠券使用率
    X['c11'] = X.c9 / X.c8

    # 优惠券核销平均时间
    temp = pd.merge(coupon_consume, coupon_consume.groupby('Coupon_id').Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('Coupon_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('Coupon_id').size().reset_index(name='count'))
    temp['c12'] = ((temp['max'] - temp['min']).dt.days / (temp['count'] - 1))
    temp = temp.drop_duplicates('Coupon_id')
    X = pd.merge(X, temp[['Coupon_id', 'c12']], how='left', on='Coupon_id')

    '''user merchant feature'''

    # 用户领取商家的优惠券次数
    temp = offline[offline.Coupon_id != 0]
    temp = temp.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um1')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后不核销次数
    temp = coupon_no_consume.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um2')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后核销次数
    temp = coupon_consume.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um3')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后核销率
    X['um4'] = X.um3 / X.um1

    # 用户对每个商家的不核销次数占用户总的不核销次数的比重
    temp = coupon_no_consume.groupby('User_id').size().reset_index(name='temp')
    X = pd.merge(X, temp, how='left', on='User_id')
    X['um5'] = X.um2 / X.temp
    X.drop(columns='temp', inplace=True)

    # 用户在商店总共消费过几次
    temp = offline[offline.Date != date_null].groupby(['User_id', 'Merchant_id']).size().reset_index(name='um6')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户在商店普通消费次数
    temp = offline[(offline.Coupon_id == 0) & (offline.Date != date_null)]
    temp = temp.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um7')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户当天在此商店领取的优惠券数目
    temp = offline[offline.Date_received != date_null]
    temp = temp.groupby(['User_id', 'Merchant_id', 'Date_received']).size().reset_index(name='um8')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id', 'Date_received'])

    # 用户领取优惠券不同商家数量
    temp = offline[offline.Coupon_id == offline.Coupon_id]
    temp = temp.groupby(['User_id', 'Merchant_id']).size().reset_index()
    temp = temp.groupby('User_id').size().reset_index(name='um9')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券不同商家数量
    temp = coupon_consume.groupby(['User_id', 'Merchant_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='um10')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销过优惠券的不同商家数量占所有不同商家的比重
    X['um11'] = X.um10 / X.um9

    # 用户平均核销每个商家多少张优惠券
    X['um12'] = X.u2 / X.um9

    '''other feature'''

    # 用户领取的所有优惠券数目
    temp = X.groupby('User_id').size().reset_index(name='o1')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户领取的特定优惠券数目
    temp = X.groupby(['User_id', 'Coupon_id']).size().reset_index(name='o2')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Coupon_id'])

    # multiple threads
    # data split
    stop = len(X)
    step = int(ceil(stop / cpu_jobs))

    X_chunks = [X[i:i + step] for i in range(0, stop, step)]
    X_list = [X] * cpu_jobs
    counters = [i for i in range(cpu_jobs)]

    start = datetime.datetime.now()
    with ProcessPoolExecutor() as e:
        X = pd.concat(e.map(task, X_chunks, X_list, counters))
        print('time:', str(datetime.datetime.now() - start).split('.')[0])
    # multiple threads

    # 用户领取优惠券平均时间间隔
    temp = pd.merge(X, X.groupby('User_id').Date_received.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('User_id').Date_received.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('User_id').size().reset_index(name='len'))
    temp['o7'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('User_id')
    X = pd.merge(X, temp[['User_id', 'o7']], how='left', on='User_id')

    # 用户领取特定商家的优惠券数目
    temp = X.groupby(['User_id', 'Merchant_id']).size().reset_index(name='o8')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取的不同商家数目
    temp = X.groupby(['User_id', 'Merchant_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='o9')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户当天领取的优惠券数目
    temp = X.groupby(['User_id', 'Date_received']).size().reset_index(name='o10')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Date_received'])

    # 用户当天领取的特定优惠券数目
    temp = X.groupby(['User_id', 'Coupon_id', 'Date_received']).size().reset_index(name='o11')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Coupon_id', 'Date_received'])

    # 用户领取的所有优惠券种类数目
    temp = X.groupby(['User_id', 'Coupon_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='o12')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 商家被领取的优惠券数目
    temp = X.groupby('Merchant_id').size().reset_index(name='o13')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被领取的特定优惠券数目
    temp = X.groupby(['Merchant_id', 'Coupon_id']).size().reset_index(name='o14')
    X = pd.merge(X, temp, how='left', on=['Merchant_id', 'Coupon_id'])

    # 商家被多少不同用户领取的数目
    temp = X.groupby(['Merchant_id', 'User_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='o15')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家发行的所有优惠券种类数目
    temp = X.groupby(['Merchant_id', 'Coupon_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='o16')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    print(len(X), len(X.columns))

    return X


def get_online_features(online, X):
    # temp = online[online.Coupon_id == online.Coupon_id]
    # coupon_consume = temp[temp.Date == temp.Date]
    # coupon_no_consume = temp[temp.Date != temp.Date]

    # 用户线上操作次数
    temp = online.groupby('User_id').size().reset_index(name='on_u1')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上点击次数
    temp = online[online.Action == 0].groupby('User_id').size().reset_index(name='on_u2')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上点击率
    X['on_u3'] = X.on_u2 / X.on_u1

    # 用户线上购买次数
    temp = online[online.Action == 1].groupby('User_id').size().reset_index(name='on_u4')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上购买率
    X['on_u5'] = X.on_u4 / X.on_u1

    # 用户线上领取次数
    temp = online[online.Coupon_id != 0].groupby('User_id').size().reset_index(name='on_u6')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上领取率
    X['on_u7'] = X.on_u6 / X.on_u1

    # 用户线上不消费次数
    temp = online[(online.Date == date_null) & (online.Coupon_id != 0)]
    temp = temp.groupby('User_id').size().reset_index(name='on_u8')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上优惠券核销次数
    temp = online[(online.Date != date_null) & (online.Coupon_id != 0)]
    temp = temp.groupby('User_id').size().reset_index(name='on_u9')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上优惠券核销率
    X['on_u10'] = X.on_u9 / X.on_u6

    # 用户线下不消费次数占线上线下总的不消费次数的比重
    X['on_u11'] = X.u3 / (X.on_u8 + X.u3)

    # 用户线下的优惠券核销次数占线上线下总的优惠券核销次数的比重
    X['on_u12'] = X.u2 / (X.on_u9 + X.u2)

    # 用户线下领取的记录数量占总的记录数量的比重
    X['on_u13'] = X.u1 / (X.on_u6 + X.u1)

    # # 消费优惠券的平均折率
    # temp = coupon_consume.groupby('User_id').discount_rate.mean().reset_index(name='ou14')
    # X = pd.merge(X, temp, how='left', on='User_id')
    #
    # # 用户核销优惠券的最低消费折率
    # temp = coupon_consume.groupby('User_id').discount_rate.min().reset_index(name='ou15')
    # X = pd.merge(X, temp, how='left', on='User_id')
    #
    # # 用户核销优惠券的最高消费折率
    # temp = coupon_consume.groupby('User_id').discount_rate.max().reset_index(name='ou16')
    # X = pd.merge(X, temp, how='left', on='User_id')
    #
    # # 不同打折优惠券领取次数
    # temp = online.groupby('Discount_rate').size().reset_index(name='oc1')
    # X = pd.merge(X, temp, how='left', on='Discount_rate')
    #
    # # 不同打折优惠券使用次数
    # temp = coupon_consume.groupby('Discount_rate').size().reset_index(name='oc2')
    # X = pd.merge(X, temp, how='left', on='Discount_rate')
    #
    # # 不同打折优惠券不使用次数
    # temp = coupon_no_consume.groupby('Discount_rate').size().reset_index(name='oc3')
    # X = pd.merge(X, temp, how='left', on='Discount_rate')
    #
    # # 不同打折优惠券使用率
    # X['oc4'] = X.oc2 / X.oc1

    print(len(X), len(X.columns))
    print('----------')

    return X


def get_train_data():
    path = 'data_preprocessed_4/cache_%s_train.csv' % os.path.basename(__file__)

    if os.path.exists(path):
        data = pd.read_csv(path)
    else:
        offline, online = get_preprocess_data()

        # date received 2016-01-01 - 2016-06-15
        # date consumed 2016-01-01 - 2016-06-30

        # train data 1
        # 2016-04-16 ~ 2016-05-15
        data_1 = offline[('2016-04-16' <= offline.Date_received) & (offline.Date_received <= '2016-05-15')].copy()
        data_1['label'] = 0
        data_1.loc[
            (data_1.Date != date_null) & (data_1.Date - data_1.Date_received <= datetime.timedelta(15)), 'label'] = 1

        # feature data 1
        # 领券 2016-01-01 ~ 2016-03-31
        end = '2016-03-31'
        data_off_1 = offline[offline.Date_received <= end]
        data_on_1 = online[online.Date_received <= end]

        # 普通消费 2016-01-01 ~ 2016-04-15
        end = '2016-04-15'
        data_off_2 = offline[(offline.Coupon_id == 0) & (offline.Date <= end)]
        data_on_2 = online[(online.Coupon_id == 0) & (online.Date <= end)]

        data_1 = get_offline_features(data_1, pd.concat([data_off_1, data_off_2]))
        data_1 = get_online_features(pd.concat([data_on_1, data_on_2]), data_1)

        # train data 2
        # 2016-05-16 ~ 2016-06-15
        data_2 = offline['2016-05-16' <= offline.Date_received].copy()
        data_2['label'] = 0
        data_2.loc[
            (data_2.Date != date_null) & (data_2.Date - data_2.Date_received <= datetime.timedelta(15)), 'label'] = 1

        # feature data 2
        # 领券
        start = '2016-02-01'
        end = '2016-04-30'
        data_off_1 = offline[(start <= offline.Date_received) & (offline.Date_received <= end)]
        data_on_1 = online[(start <= online.Date_received) & (online.Date_received <= end)]

        # 普通消费
        start = '2016-02-01'
        end = '2016-05-15'
        data_off_2 = offline[(offline.Coupon_id == 0) & (start <= offline.Date) & (offline.Date <= end)]
        data_on_2 = online[(online.Coupon_id == 0) & (start <= online.Date) & (online.Date <= end)]

        data_2 = get_offline_features(data_2, pd.concat([data_off_1, data_off_2]))
        data_2 = get_online_features(pd.concat([data_on_1, data_on_2]), data_2)

        data = pd.concat([data_1, data_2])

        # undersampling
        # if undersampling:
        #     temp = X_1[X_1.label == 1].groupby('User_id').size().reset_index()
        #     temp = X_1[X_1.User_id.isin(temp.User_id)]
        #     X_1 = pd.concat([temp, X_1[~X_1.User_id.isin(temp.User_id)].sample(4041)])

        # data.drop_duplicates(inplace=True)
        drop_columns(data)
        data.fillna(0, inplace=True)
        data.to_csv(path, index=False)

    return data


def analysis():
    offline, online = get_preprocess_data()

    # t = offline.groupby('Discount_rate').size().reset_index(name='receive_count')
    # t1 = offline[(offline.Coupon_id != 0) & (offline.Date != date_null)]
    # t1 = t1.groupby('Discount_rate').size().reset_index(name='consume_count')
    # t = pd.merge(t, t1, on='Discount_rate')
    # t['consume_rate'] = t.consume_count / t.receive_count

    # t = offline.groupby('Merchant_id').size().reset_index(name='receive_count')
    # t1 = offline[(offline.Coupon_id != 0) & (offline.Date != date_null)]
    # t1 = t1.groupby('Merchant_id').size().reset_index(name='consume_count')
    # t = pd.merge(t, t1, on='Merchant_id')
    # t['consume_rate'] = t.consume_count / t.receive_count

    t = offline.groupby('Distance').size().reset_index(name='receive_count')
    t1 = offline[(offline.Coupon_id != 0) & (offline.Date != date_null)]
    t1 = t1.groupby('Distance').size().reset_index(name='consume_count')
    t = pd.merge(t, t1, on='Distance')
    t['consume_rate'] = t.consume_count / t.receive_count

    t.to_csv('data_preprocessed_4/note.csv')

    # plt.bar(temp.Discount_rate.values, temp.total.values)
    # plt.bar(range(num), y1, bottom=y2, fc='r')
    # plt.show()

    exit()


def detect_duplicate_columns():
    X = get_train_data()
    X = X[:1000]

    for index1 in range(len(X.columns) - 1):
        for index2 in range(index1 + 1, len(X.columns)):
            column1 = X.columns[index1]
            column2 = X.columns[index2]
            X[column1] = X[column1].astype(str)
            X[column2] = X[column2].astype(str)
            temp = len(X[X[column1] == X[column2]])
            if temp == len(X):
                print(column1, column2, temp)
    exit()


def feature_importance_score():
    clf = train_xgb()
    fscores = pd.Series(clf.get_booster().get_fscore()).sort_values(ascending=False)
    fscores.plot(kind='bar', title='Feature Importance')
    plt.ylabel('Feature Importance Score')
    plt.show()
    exit()


def feature_selection():
    data = get_train_data()

    train_data, test_data = train_test_split(data,
                                             train_size=100000,
                                             random_state=0
                                             )

    X = train_data.copy().drop(columns='Coupon_id')
    y = X.pop('label')

    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # X = sel.fit_transform(X)
    # print(X.shape)
    # Create the RFE object and rank each pixel


def fit_eval_metric(estimator, X, y, name=None):
    if name is None:
        name = estimator.__class__.__name__

    if name is 'XGBClassifier' or name is 'LGBMClassifier':
        estimator.fit(X, y, eval_metric='auc')
    else:
        estimator.fit(X, y)

    return estimator


def grid_search(estimator, param_grid):
    start = datetime.datetime.now()

    print('--------------------------------------------')
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    print(param_grid)
    print()

    data = get_train_data()

    data, _ = train_test_split(data, train_size=100000, random_state=0)

    X = data.copy().drop(columns='Coupon_id')
    y = X.pop('label')

    estimator_name = estimator.__class__.__name__
    n_jobs = cpu_jobs
    if estimator_name is 'XGBClassifier' or estimator_name is 'LGBMClassifier' or estimator_name is 'CatBoostClassifier':
        n_jobs = 1

    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs
                       # cv=5
                       )

    clf = fit_eval_metric(clf, X, y, estimator_name)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.5f (+/-%0.05f) for %r' % (mean, std * 2, params))
    print()
    print('best params', clf.best_params_)
    print('best score', clf.best_score_)
    print('time: %s' % str((datetime.datetime.now() - start)).split('.')[0])
    print()

    return clf.best_params_, clf.best_score_


def grid_search_auto(steps, params, estimator):
    global log

    old_params = params.copy()

    while 1:
        for name, step in steps.items():
            score = 0

            start = params[name] - step['step']
            if start <= step['min']:
                start = step['min']

            stop = params[name] + step['step']
            if step['max'] != 'inf' and stop >= step['max']:
                stop = step['max']

            while 1:

                if str(step['step']).count('.') == 1:
                    stop += step['step'] / 10
                else:
                    stop += step['step']

                param_grid = {
                    name: np.arange(start, stop, step['step']),
                }

                best_params, best_score = grid_search(estimator.set_params(**params), param_grid)

                if best_params[name] == params[name] or score > best_score:
                    print(estimator.__class__.__name__, params)
                    break

                direction = (best_params[name] - params[name]) // abs(best_params[name] - params[name])
                start = stop = best_params[name] + step['step'] * direction

                score = best_score
                params[name] = best_params[name]
                print(estimator.__class__.__name__, params)

                if best_params[name] - step['step'] < step['min'] or (
                        step['max'] != 'inf' and best_params[name] + step['step'] > step['max']):
                    break

        if old_params == params:
            break
        old_params = params
        print('--------------------------------------------')
        print('new grid search')

    print('--------------------------------------------')
    log += 'grid search: %s\n%r\n' % (estimator.__class__.__name__, params)


def grid_search_gbdt(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1900,
        'max_depth': 9,
        'min_samples_split': 200,
        'min_samples_leaf': 50,
        'subsample': .8,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 200,
        # 'max_depth': 8,
        # 'min_samples_split': 200,
        # 'min_samples_leaf': 50,
        # 'subsample': .8,

        'random_state': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 10, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 10, 'min': 1, 'max': 'inf'},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, GradientBoostingClassifier())


def grid_search_xgb(get_param=False):
    params = {
        # all
        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1260,
        'max_depth': 8,
        'min_child_weight': 4,
        'gamma': .2,
        'subsample': .6,
        'colsample_bytree': .8,
        'scale_pos_weight': 1,
        'reg_alpha': 0,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,
        # 'scale_pos_weight': 1,
        # 'reg_alpha': 0,

        'n_jobs': cpu_jobs,
        'seed': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_child_weight': {'step': 1, 'min': 1, 'max': 'inf'},
        'gamma': {'step': .1, 'min': 0, 'max': 1},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
        'scale_pos_weight': {'step': 1, 'min': 1, 'max': 10},
        'reg_alpha': {'step': .1, 'min': 0, 'max': 1},
    }

    grid_search_auto(steps, params, XGBClassifier())


def grid_search_lgb(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1200,
        'num_leaves': 51,
        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'min_child_samples': 22,
        'subsample': .8,
        'colsample_bytree': .8,

        # 'learning_rate': .1,
        # 'n_estimators': 90,
        # 'num_leaves': 50,
        # 'min_split_gain': 0,
        # 'min_child_weight': 1e-3,
        # 'min_child_samples': 21,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        'n_jobs': cpu_jobs,
        'random_state': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'num_leaves': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_split_gain': {'step': .1, 'min': 0, 'max': 1},
        'min_child_weight': {'step': 1e-3, 'min': 1e-3, 'max': 'inf'},
        'min_child_samples': {'step': 1, 'min': 1, 'max': 'inf'},
        # 'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, LGBMClassifier())


def grid_search_cat(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 3600,
        'max_depth': 8,
        'max_bin': 127,
        'reg_lambda': 2,
        'subsample': .7,

        # 'learning_rate': 1e-1,
        # 'iterations': 460,
        # 'depth': 8,
        # 'l2_leaf_reg': 8,
        # 'border_count': 37,

        # 'ctr_border_count': 16,
        'one_hot_max_size': 2,
        'bootstrap_type': 'Bernoulli',
        'leaf_estimation_method': 'Newton',
        'random_state': 0,
        'verbose': False,
        'eval_metric': 'AUC',
        'thread_count': cpu_jobs
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'max_bin': {'step': 1, 'min': 1, 'max': 255},
        'reg_lambda': {'step': 1, 'min': 0, 'max': 'inf'},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'one_hot_max_size': {'step': 1, 'min': 0, 'max': 255},
    }

    grid_search_auto(steps, params, CatBoostClassifier())


def grid_search_rf(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3090,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0
        }
    else:
        params = {
            'n_estimators': 3110,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, RandomForestClassifier())


def grid_search_et(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3060,
            'max_depth': 22,
            'min_samples_split': 12,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0,
        }
    else:
        params = {
            'n_estimators': 3100,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, ExtraTreesClassifier())


def train_gbdt(model=False):
    global log

    params = grid_search_gbdt(True)
    clf = GradientBoostingClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'gbdt'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', subsample: %.1f' % params['subsample']
    log += '\n\n'

    return train(clf)


def train_xgb(model=False):
    global log

    params = grid_search_xgb(True)

    clf = XGBClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'xgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_child_weight: %d' % params['min_child_weight']
    log += ', gamma: %.1f' % params['gamma']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_lgb(model=False):
    global log

    params = grid_search_lgb(True)

    clf = LGBMClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'lgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', num_leaves: %d' % params['num_leaves']
    log += ', min_split_gain: %.1f' % params['min_split_gain']
    log += ', min_child_weight: %.4f' % params['min_child_weight']
    log += ', min_child_samples: %d' % params['min_child_samples']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_cat(model=False):
    global log

    params = grid_search_cat(True)

    clf = CatBoostClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'cat'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', iterations: %d' % params['iterations']
    log += ', depth: %d' % params['depth']
    log += ', l2_leaf_reg: %d' % params['l2_leaf_reg']
    log += ', border_count: %d' % params['border_count']
    log += ', subsample: %d' % params['subsample']
    log += ', one_hot_max_size: %d' % params['one_hot_max_size']
    log += '\n\n'

    return train(clf)


def train_rf(clf):
    global log

    params = clf.get_params()
    log += 'rf'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_rf_gini(model=False):
    clf = RandomForestClassifier().set_params(**grid_search_rf('gini', True))
    if model:
        return clf
    return train_rf(clf)


def train_rf_entropy():
    clf = RandomForestClassifier().set_params(**grid_search_rf('entropy', True))

    return train_rf(clf)


def train_et(clf):
    global log

    params = clf.get_params()
    log += 'et'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_et_gini(model=False):
    clf = ExtraTreesClassifier().set_params(**grid_search_et('gini', True))
    if model:
        return clf
    return train_et(clf)


def train_et_entropy():
    clf = ExtraTreesClassifier().set_params(**{
        'n_estimators': 3100,
        'max_depth': 13,
        'min_samples_split': 70,
        'min_samples_leaf': 10,
        'criterion': 'entropy',
        'random_state': 0
    })

    return train_et(clf)


def train(clf):
    global log

    data = get_train_data()

    train_data, test_data = train_test_split(data,
                                             train_size=100000,
                                             random_state=0
                                             )

    _, test_data = train_test_split(data, random_state=0)

    X_train = train_data.copy().drop(columns='Coupon_id')
    y_train = X_train.pop('label')

    clf = fit_eval_metric(clf, X_train, y_train)

    X_test = test_data.copy().drop(columns='Coupon_id')
    y_test = X_test.pop('label')

    y_true, y_pred = y_test, clf.predict(X_test)
    # log += '%s\n' % classification_report(y_test, y_pred)
    log += '  accuracy: %f\n' % accuracy_score(y_true, y_pred)
    y_score = clf.predict_proba(X_test)[:, 1]
    log += '       auc: %f\n' % roc_auc_score(y_true, y_score)

    # coupon average auc
    coupons = test_data.groupby('Coupon_id').size().reset_index(name='total')
    aucs = []
    for _, coupon in coupons.iterrows():
        if coupon.total > 1:
            X_test = test_data[test_data.Coupon_id == coupon.Coupon_id].copy()
            X_test.drop(columns='Coupon_id', inplace=True)

            if len(X_test.label.unique()) != 2:
                continue

            y_true = X_test.pop('label')
            y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_true, y_score))

    log += 'coupon auc: %f\n\n' % np.mean(aucs)

    return clf


def predict(model):
    path = 'data_preprocessed_4/cache_%s_predict.csv' % os.path.basename(__file__)

    if os.path.exists(path):
        X = pd.read_csv(path, parse_dates=['Date_received'])
    else:
        offline, online = get_preprocess_data()

        # 2016-03-16 ~ 2016-06-30
        start = '2016-03-16'
        offline = offline[(offline.Coupon_id == 0) & (start <= offline.Date) | (start <= offline.Date_received)]
        online = online[(online.Coupon_id == 0) & (start <= online.Date) | (start <= online.Date_received)]

        X = get_preprocess_data(True)
        X = get_offline_features(X, offline)
        X = get_online_features(online, X)
        X.drop_duplicates(inplace=True)
        X.fillna(0, inplace=True)
        X.to_csv(path, index=False)

    sample_submission = X[['User_id', 'Coupon_id', 'Date_received']].copy()
    sample_submission.Date_received = sample_submission.Date_received.dt.strftime('%Y%m%d')
    drop_columns(X, True)

    if model is 'blending':
        predict = blending(X)
    else:
        clf = eval('train_%s' % model)()
        predict = clf.predict_proba(X)[:, 1]

    sample_submission['Probability'] = predict
    sample_submission.to_csv('train_dir_4/submission_%s.csv' % model,
                             #  float_format='%.5f',
                             index=False, header=False)


def blending(predict_X=None):
    global log
    log += '\n'

    X = get_train_data().drop(columns='Coupon_id')
    y = X.pop('label')

    X = np.asarray(X)
    y = np.asarray(y)

    _, X_submission, _, y_test_blend = train_test_split(X, y,
                                                        random_state=0
                                                        )

    if predict_X is not None:
        X_submission = np.asarray(predict_X)

    X, _, y, _ = train_test_split(X, y,
                                  train_size=100000,
                                  random_state=0
                                  )

    # np.random.seed(0)
    # idx = np.random.permutation(y.size)
    # X = X[idx]
    # y = y[idx]

    skf = StratifiedKFold()
    clfs = ['gbdt', 'xgb', 'lgb', 'cat',
            # 'rf_gini', 'et_gini'
            ]

    blend_X_train = np.zeros((X.shape[0], len(clfs)))
    blend_X_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, v in enumerate(clfs):
        clf = eval('train_%s' % v)(True)

        aucs = []
        dataset_blend_test_j = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = fit_eval_metric(clf, X_train, y_train)

            y_submission = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_submission))

            blend_X_train[test_index, j] = y_submission
            dataset_blend_test_j.append(clf.predict_proba(X_submission)[:, 1])

        log += '%7s' % v + ' auc: %f\n' % np.mean(aucs)
        blend_X_test[:, j] = np.asarray(dataset_blend_test_j).T.mean(1)

    print('blending')
    clf = LogisticRegression()
    # clf = GradientBoostingClassifier()
    clf.fit(blend_X_train, y)
    y_submission = clf.predict_proba(blend_X_test)[:, 1]

    # Linear stretch of predictions to [0,1]
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    if predict_X is not None:
        return y_submission
    log += '\n  blend auc: %f\n\n' % roc_auc_score(y_test_blend, y_submission)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    # analysis()
    # detect_duplicate_columns()
    # feature_importance_score()

    # grid_search_gbdt()
    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    train_xgb()
    predict('xgb')

    # grid_search_lgb()
    # train_lgb()
    # predict('lgb')

    # grid_search_cat()
    # train_cat()
    # predict('cat')

    # grid_search_rf()
    # train_rf_gini()
    # predict('rf_gini')

    # grid_search_rf('entropy')
    # train_rf_entropy()
    # predict('rf_entropy')

    # grid_search_et()
    # train_et_gini()
    # predict('et_gini')

    # grid_search_et('entropy')
    # train_et_entropy()
    # predict('et_entropy')

    # blending()
    # predict('blending')

    log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    log += '----------------------------------------------------\n'
    open('%s.log' % os.path.basename(__file__), 'a').write(log)
    print(log)
