import datetime
import os
from concurrent.futures import ProcessPoolExecutor
from math import ceil
import pandas as pd


# In[] 读入源数据
def get_source_data():
    # 源数据路径
    DataPath = 'dataset'

    # 读入源数据
    off_train = pd.read_csv(os.path.join(DataPath, 'ccf_offline_stage1_train.csv'),
                            parse_dates=['Date_received', 'Date'])
    off_train.columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received', 'Date']

    on_train = pd.read_csv(os.path.join(DataPath, 'ccf_online_stage1_train.csv'), parse_dates=['Date_received', 'Date'])
    on_train.columns = ['User_id', 'Merchant_id', 'Action', 'Coupon_id', 'Discount_rate', 'Date_received', 'Date']

    off_test = pd.read_csv(os.path.join(DataPath, 'ccf_offline_stage1_test_revised.csv'), parse_dates=['Date_received'])
    off_test.columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received']

    print(off_train.info())
    print(off_train.head(5))
    return off_train, on_train, off_test


# In[] null,na 特殊处理
def null_process_offline(dataset, predict=False):
    dataset.Distance.fillna(11, inplace=True)
    dataset.Distance = dataset.Distance.astype(int)
    dataset.Coupon_id.fillna(0, inplace=True)
    dataset.Coupon_id = dataset.Coupon_id.astype(int)
    dataset.Date_received.fillna(date_null, inplace=True)

    dataset[['discount_rate_x', 'discount_rate_y']] = dataset[dataset.Discount_rate.str.contains(':') == True][
        'Discount_rate'].str.split(':', expand=True).astype(int)
    dataset['discount_rate'] = 1 - dataset.discount_rate_y / dataset.discount_rate_x
    dataset.discount_rate = dataset.discount_rate.fillna(dataset.Discount_rate).astype(float)
    if predict:
        return dataset
    else:
        dataset.Date.fillna(date_null, inplace=True)
        return dataset


def null_process_online(dataset):
    dataset.Coupon_id.fillna(0, inplace=True)
    # online.Coupon_id = online.Coupon_id.astype(int)
    dataset.Date_received.fillna(date_null, inplace=True)
    dataset.Date.fillna(date_null, inplace=True)
    return dataset


# In[] 生成交叉训练集
def data_process(off_train, on_train, off_test):
    # train feature split
    # 交叉训练集一：收到券的日期大于4月14日和小于5月14日
    time_range = ['2016-04-16', '2016-05-15']
    dataset1 = off_train[(off_train.Date_received >= time_range[0]) & (off_train.Date_received <= time_range[1])].copy()
    dataset1['label'] = 0
    dataset1.loc[
        (dataset1.Date != date_null) & (dataset1.Date - dataset1.Date_received <= datetime.timedelta(15)), 'label'] = 1
    # 交叉训练集一特征offline：线下数据中领券和用券日期大于1月1日和小于4月13日
    time_range_date_received = ['2016-01-01', '2016-03-31']
    time_range_date = ['2016-01-01', '2016-04-15']
    feature1_off = off_train[(off_train.Date >= time_range_date[0]) & (off_train.Date <= time_range_date[1]) | (
            (off_train.Coupon_id == 0) & (off_train.Date_received >= time_range_date_received[0]) & (
            off_train.Date_received <= time_range_date_received[1]))]
    # 交叉训练集一特征online：线上数据中领券和用券日期大于1月1日和小于4月13日[on_train.date == 'null' to on_train.coupon_id == 0]
    feature1_on = on_train[(on_train.Date >= time_range_date[0]) & (on_train.Date <= time_range_date[1]) | (
            (on_train.Coupon_id == 0) & (on_train.Date_received >= time_range_date_received[0]) & (
            on_train.Date_received <= time_range_date_received[1]))]

    # 交叉训练集二：收到券的日期大于5月15日和小于6月15日
    time_range = ['2016-05-16', '2016-06-15']
    dataset2 = off_train[(off_train.Date_received >= time_range[0]) & (off_train.Date_received <= time_range[1])]
    dataset2['label'] = 0
    dataset2.loc[
        (dataset2.Date != date_null) & (dataset2.Date - dataset2.Date_received <= datetime.timedelta(15)), 'label'] = 1
    # 交叉训练集二特征offline：线下数据中领券和用券日期大于2月1日和小于5月14日
    time_range_date_received = ['2016-02-01', '2016-04-30']
    time_range_date = ['2016-02-01', '2016-05-15']
    feature2_off = off_train[(off_train.Date >= time_range_date[0]) & (off_train.Date <= time_range_date[1]) | (
            (off_train.Coupon_id == 0) & (off_train.Date_received >= time_range_date_received[0]) & (
            off_train.Date_received <= time_range_date_received[1]))]
    # 交叉训练集二特征online：线上数据中领券和用券日期大于2月1日和小于5月14日
    feature2_on = on_train[(on_train.Date >= time_range_date[0]) & (on_train.Date <= time_range_date[1]) | (
            (on_train.Coupon_id == 0) & (on_train.Date_received >= time_range_date_received[0]) & (
            on_train.Date_received <= time_range_date_received[1]))]

    # 测试集
    dataset3 = off_test
    # 测试集特征offline :线下数据中领券和用券日期大于3月15日和小于6月30日的
    time_range = ['2016-03-16', '2016-06-30']
    feature3_off = off_train[((off_train.Date >= time_range[0]) & (off_train.Date <= time_range[1])) | (
            (off_train.Coupon_id == 0) & (off_train.Date_received >= time_range[0]) & (
            off_train.Date_received <= time_range[1]))]
    # 测试集特征online :线上数据中领券和用券日期大于3月15日和小于6月30日的
    feature3_on = on_train[((on_train.Date >= time_range[0]) & (on_train.Date <= time_range[1])) | (
            (on_train.Coupon_id == 0) & (on_train.Date_received >= time_range[0]) & (
            on_train.Date_received <= time_range[1]))]

    # get train feature
    ProcessDataSet1 = get_features(dataset1, feature1_off, feature1_on)
    ProcessDataSet2 = get_features(dataset2, feature2_off, feature2_on)
    ProcessDataSet3 = get_features(dataset3, feature3_off, feature3_on)

    return ProcessDataSet1, ProcessDataSet2, ProcessDataSet3


def get_features(dataset, feature_off, feature_on):
    dataset = get_offline_features(dataset, feature_off)
    return get_online_features(feature_on, dataset)


# In[] 定义获取feature的函数
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


# In[]
if __name__ == '__main__':
    # 程序开始时间打印
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)

    # 预处理后数据存放路径
    FeaturePath = 'data_preprocessed_2'

    # 读入源数据
    off_train, on_train, off_test = get_source_data()

    # 源数据null处理
    off_train = null_process_offline(off_train, predict=False)
    on_train = null_process_online(on_train)
    off_test = null_process_offline(off_test, predict=True)

    # 获取训练特征集，测试特征集
    ProcessDataSet1, ProcessDataSet2, ProcessDataSet3 = data_process(off_train, on_train, off_test)

    # 源数据处理后的数据保存为文件
    # dataset_1 = get_offline_features(dataset1, feature1_off)
    # ProcessDataSet1 = get_online_features(feature1_on, dataset_1)
    ProcessDataSet1.to_csv(os.path.join(FeaturePath, 'ProcessDataSet1.csv'), index=None)

    # dataset_2 = get_offline_features(dataset2, feature2_off)
    # ProcessDataSet2 = get_online_features(feature2_on, dataset_2)
    ProcessDataSet2.to_csv(os.path.join(FeaturePath, 'ProcessDataSet2.csv'), index=None)

    # dataset_3 = get_offline_features(dataset3, feature3_off)
    # ProcessDataSet3 = get_online_features(feature3_on, dataset_3)
    ProcessDataSet3.to_csv(os.path.join(FeaturePath, 'ProcessDataSet3.csv'), index=None)

    # 花费时间打印
    print((datetime.datetime.now() - start).seconds)
