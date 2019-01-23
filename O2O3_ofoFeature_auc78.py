#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
import os

# 如果没有keep_default_na=False，加载后空值处就是NAN，且类似coupon_id等处的类型都是float
# 判断是否是NAN的话是：off_train.date!=off_train.date结果是True即为NAN，否则是非空值
# 这里使用了keep_default_na=False，使coupon_id等字段的数据类型转化为object可以简单看作是字符串，空值变为null
# 这时候判断是否是空值便可用off_train.date=='null'

# 源数据路径
DataPath = 'D:/MachineLearning/ofo/ofoOptimization'
# 预处理后数据存放路径
FeaturePath = 'D:/MachineLearning/ofo/ofoOptimization'

off_train = pd.read_csv(os.path.join(DataPath, 'ccf_offline_stage1_train.csv'), header=0, keep_default_na=False)
off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

off_test = pd.read_csv(os.path.join(DataPath, 'ccf_offline_stage1_test_revised.csv'), header=0, keep_default_na=False)
off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

print(off_train.info())
print(off_train.head(5))

# In[2]:


# 交叉训练集一：收到券的日期大于4月14日和小于5月14日
dataset1 = off_train[(off_train.date_received >= '201604014') & (off_train.date_received <= '20160514')]
# 交叉训练集一特征：线下数据中领券和用券日期大于1月1日和小于4月13日
feature1 = off_train[(off_train.date >= '20160101') & (off_train.date <= '20160413') | (
            (off_train.date == 'null') & (off_train.date_received >= '20160101') & (
                off_train.date_received <= '20160413'))]

# 交叉训练集二：收到券的日期大于5月15日和小于6月15日
dataset2 = off_train[(off_train.date_received >= '20160515') & (off_train.date_received <= '20160615')]
# 交叉训练集二特征：线下数据中领券和用券日期大于2月1日和小于5月14日
feature2 = off_train[(off_train.date >= '20160201') & (off_train.date <= '20160514') | (
            (off_train.date == 'null') & (off_train.date_received >= '20160201') & (
                off_train.date_received <= '20160514'))]

# 测试集
dataset3 = off_test
# 测试集特征 :线下数据中领券和用券日期大于3月15日和小于6月30日的
feature3 = off_train[((off_train.date >= '20160315') & (off_train.date <= '20160630')) | (
            (off_train.date == 'null') & (off_train.date_received >= '20160315') & (
                off_train.date_received <= '20160630'))]


# In[3]:


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        # 将时间差转化为天数
        this_gap = (dt.date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - dt.date(
            int(d[0:4]), int(d[4:6]), int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8])) - dt.datetime(int(date_received[0:4]),
                                                                                     int(date_received[4:6]),
                                                                                     int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def GetOtherFeature(dataset):
    t = dataset[['user_id']].copy()
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()

    t1 = dataset[['user_id', 'coupon_id']].copy()
    t1['this_month_user_receive_same_coupn_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

    t2 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t2.date_received = t2.date_received.astype('str')
    # 如果出现相同的用户接收相同的优惠券在接收时间上用‘：’连接上第n次接受优惠券的时间
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    # 将接收时间的一组按着':'分开，这样就可以计算接受了优惠券的数量,apply是合并
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    # 最大接受的日期

    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    # 最小的接收日期
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset[['user_id', 'coupon_id', 'date_received']]
    # 将两表融合只保留左表数据,这样得到的表，相当于保留了最近接收时间和最远接受时间
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    # 这个优惠券最近接受时间
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype(int)
    # 这个优惠券最远接受时间
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype(int) - t3.min_date_received

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    # 提取第四个特征,一个用户所接收到的所有优惠券的数量
    t4 = dataset[['user_id', 'date_received']].copy()
    t4['this_day_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    # 提取第五个特征,一个用户不同时间所接收到不同优惠券的数量
    t5 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()
    # 一个用户不同优惠券 的接受时间
    t6 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

    other_feature = pd.merge(t1, t, on='user_id')
    other_feature = pd.merge(other_feature, t3, on=['user_id', 'coupon_id'])
    other_feature = pd.merge(other_feature, t4, on=['user_id', 'date_received'])
    other_feature = pd.merge(other_feature, t5, on=['user_id', 'coupon_id', 'date_received'])
    other_feature = pd.merge(other_feature, t7, on=['user_id', 'coupon_id', 'date_received'])
    return other_feature


# In[4]:


def calc_discount_rate(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return float(s[0])
    else:
        return 1.0 - float(s[1]) / float(s[0])


def get_discount_man(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[0])


def get_discount_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[1])


def is_man_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 0
    else:
        return 1


def GetCouponRelatedFeature(dataset, feature):
    # 为了求得每个feature中date最大的日期，其会被用在求days_distance字段
    t = feature[feature['date'] != 'null']['date'].unique()
    t = max(t)

    dataset['day_of_week'] = dataset.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    # 显示时间是几月
    dataset['day_of_month'] = dataset.date_received.astype('str').apply(lambda x: int(x[6:8]))
    # 显示时期和截止日之间的天数
    dataset['days_distance'] = dataset.date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(int(t[0:4]), int(t[4:6]), int(t[6:8]))).days)
    # 显示满了多少钱后开始减
    dataset['discount_man'] = dataset.discount_rate.apply(get_discount_man)
    # 显示满减的减少的钱
    dataset['discount_jian'] = dataset.discount_rate.apply(get_discount_jian)
    # 返回优惠券是否是满减券
    dataset['is_man_jian'] = dataset.discount_rate.apply(is_man_jian)
    # 显示打折力度
    dataset['discount_rate'] = dataset.discount_rate.apply(calc_discount_rate)
    d = dataset[['coupon_id']]
    d['coupon_count'] = 1
    # 显示每一种优惠券的数量
    d = d.groupby('coupon_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, d, on='coupon_id', how='left')
    return dataset


# In[5]:


def GetMerchantRelatedFeature(feature):
    merchant = feature[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']].copy()
    t = merchant[['merchant_id']].copy()
    # 删除重复行数据
    t.drop_duplicates(inplace=True)

    # 卖出的商品
    t1 = merchant[merchant.date != 'null'][['merchant_id']].copy()
    t1['total_sales'] = 1
    # 每个商品的销售数量
    t1 = t1.groupby('merchant_id').agg('sum').reset_index()

    # 使用了优惠券消费的商品，正样本
    t2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id']].copy()
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('merchant_id').agg('sum').reset_index()

    # 商品的优惠券的总数量
    t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']].copy()
    t3['total_coupon'] = 1
    t3 = t3.groupby('merchant_id').agg('sum').reset_index()

    # 商品销量和距离的关系
    t4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id', 'distance']].copy()
    # 下面三行代码的主要作用就是为了将distance字段的数据类型转化为int
    # 把数据中的null值全部替换为-1
    t4.replace('null', -1, inplace=True)
    t4.distance = t4.distance.astype('int')
    # 再把数据中的-1全部替换为NaN
    t4.replace(-1, np.nan, inplace=True)

    # 返回用户离商品的距离最小值
    t5 = t4.groupby('merchant_id').agg('min').reset_index()
    t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

    # 返回用户离商品的距离最大值
    t6 = t4.groupby('merchant_id').agg('max').reset_index()
    t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)
    # print(t6)

    # 返回距离的平均值
    t7 = t4.groupby('merchant_id').agg('mean').reset_index()
    t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)
    # 返回距离的中位值
    t8 = t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    merchant_feature = pd.merge(t, t1, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t2, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t3, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t5, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t6, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t7, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t8, on='merchant_id', how='left')

    # 将数据中的NaN用0来替换
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan, 0)
    # 优惠券的使用率
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype(
        'float') / merchant_feature.total_coupon
    # 即卖出商品中使用优惠券的占比
    merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_sales
    # 将数据中的NaN用0来替换
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan, 0)

    return merchant_feature


# In[6]:


def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                        int(s[1][6:8]))).days


def GetUserRelatedFeature(feature):
    # for dataset3
    user = feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']].copy()

    t = user[['user_id']].copy()
    t.drop_duplicates(inplace=True)

    # 客户一共买的商品
    t1 = user[user.date != 'null'][['user_id', 'merchant_id']].copy()
    t1.drop_duplicates(inplace=True)
    t1.merchant_id = 1
    t1 = t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的最小距离
    t2 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id', 'distance']]
    t2.replace('null', -1, inplace=True)
    t2.distance = t2.distance.astype('int')
    t2.replace(-1, np.nan, inplace=True)
    t3 = t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的最大距离
    t4 = t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的平均距离
    t5 = t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的中间距离
    t6 = t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

    # 客户使用优惠券购买的次数
    t7 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id').agg('sum').reset_index()

    # 客户使购买的总次数
    t8 = user[user.date != 'null'][['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id').agg('sum').reset_index()

    # 客户收到优惠券的总数
    t9 = user[user.coupon_id != 'null'][['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id').agg('sum').reset_index()

    # 客户从收优惠券到消费的时间间隔
    t10 = user[(user.date_received != 'null') & (user.date != 'null')][['user_id', 'date_received', 'date']]
    t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10 = t10[['user_id', 'user_date_datereceived_gap']]

    # 客户从收优惠券到消费的平均时间间隔
    t11 = t10.groupby('user_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    # 客户从收优惠券到消费的最小时间间隔
    t12 = t10.groupby('user_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    # 客户从收优惠券到消费的最大时间间隔
    t13 = t10.groupby('user_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    user_feature = pd.merge(t, t1, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t3, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t4, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t5, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t6, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t7, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t8, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t9, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t11, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t12, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t13, on='user_id', how='left')
    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan, 0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan, 0)
    user_feature['buy_use_coupon_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.buy_total.astype(
        'float')
    user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype(
        'float') / user_feature.coupon_received.astype('float')
    user_feature.buy_total = user_feature.buy_total.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan, 0)
    return user_feature


# In[7]:


def GetUserAndMerchantRelatedFeature(feature):
    all_user_merchant = feature[['user_id', 'merchant_id']].copy()
    all_user_merchant.drop_duplicates(inplace=True)

    # 一个客户在一个商家一共买的次数
    t = feature[['user_id', 'merchant_id', 'date']].copy()
    t = t[t.date != 'null'][['user_id', 'merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    # 一个客户在一个商家一共收到的优惠券
    t1 = feature[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    # 一个客户在一个商家使用优惠券购买的次数
    t2 = feature[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    # 一个客户在一个商家浏览的次数
    t3 = feature[['user_id', 'merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    # 一个客户在一个商家没有使用优惠券购买的次数
    t4 = feature[['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    user_merchant = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t1, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t2, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t3, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t4, on=['user_id', 'merchant_id'], how='left')
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_received.astype('float')
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype(
        'float') / user_merchant.user_merchant_any.astype('float')
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')
    return user_merchant


# In[8]:


def get_label(s):
    s = s.split(':')
    if s[0] == 'null':
        return 0
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                      int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return -1


def DataProcess(dataset, feature, TrainFlag):
    other_feature = GetOtherFeature(dataset)
    merchant = GetMerchantRelatedFeature(feature)
    user = GetUserRelatedFeature(feature)
    user_merchant = GetUserAndMerchantRelatedFeature(feature)
    coupon = GetCouponRelatedFeature(dataset, feature)

    dataset = pd.merge(coupon, merchant, on='merchant_id', how='left')
    dataset = pd.merge(dataset, user, on='user_id', how='left')
    dataset = pd.merge(dataset, user_merchant, on=['user_id', 'merchant_id'], how='left')
    dataset = pd.merge(dataset, other_feature, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset.drop_duplicates(inplace=True)

    dataset.user_merchant_buy_total = dataset.user_merchant_buy_total.replace(np.nan, 0)
    dataset.user_merchant_any = dataset.user_merchant_any.replace(np.nan, 0)
    dataset.user_merchant_received = dataset.user_merchant_received.replace(np.nan, 0)
    dataset['is_weekend'] = dataset.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    dataset = pd.concat([dataset, weekday_dummies], axis=1)
    if TrainFlag:
        dataset['date'] = dataset['date'].fillna('null');
        dataset['label'] = dataset.date.astype('str') + ':' + dataset.date_received.astype('str')
        dataset.label = dataset.label.apply(get_label)
        dataset.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_count'], axis=1, inplace=True)
    else:
        dataset.drop(['merchant_id', 'day_of_week', 'coupon_count'], axis=1, inplace=True)
    dataset = dataset.replace('null', np.nan)
    return dataset


# In[9]:


ProcessDataSet1 = DataProcess(dataset1, feature1, True)
ProcessDataSet1.to_csv(os.path.join(DataPath, 'ProcessDataSet1.csv'), index=None)
print('---------------ProcessDataSet1 done-------------------')
ProcessDataSet2 = DataProcess(dataset2, feature2, True)
ProcessDataSet2.to_csv(os.path.join(DataPath, 'ProcessDataSet2.csv'), index=None)
print('---------------ProcessDataSet2 done-------------------')
ProcessDataSet3 = DataProcess(dataset3, feature3, False)
ProcessDataSet3.to_csv(os.path.join(DataPath, 'ProcessDataSet3.csv'), index=None)
print('---------------ProcessDataSet3 done-------------------')

# In[ ]:


# In[ ]:
