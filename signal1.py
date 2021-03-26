# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:39:48 2021

@author: jiajia.xu
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
%matplotlib inline

import itertools
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
# from mlxtend.plotting import plot_learning_curves
# from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


#减少内存
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# 读取数据
train = pd.read_csv('./train.csv')
test = pd.read_csv('./testA.csv')

# 简单预处理
train_list = []
for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])
    
test_list = []
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])

train = pd.DataFrame(np.array(train_list))
test = pd.DataFrame(np.array(test_list))

# id列不算入特征
features = ['s_'+str(i) for i in range(len(train_list[0])-2)] 
train.columns = ['id'] + features + ['label']
test.columns = ['id'] + features

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# 根据8：2划分训练集和校验集
X_train = train.drop(['id','label'], axis=1)
y_train = train['label']

# 测试集
X_test = test.drop(['id'], axis=1)

# 划分训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


#定义评价指标
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(4, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True
#贝叶斯调参


from sklearn.model_selection import cross_val_score

"""定义优化函数"""
def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf, 
              min_child_weight, min_split_gain, reg_lambda, reg_alpha):
    # 建立模型
    model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=4,
                                   learning_rate=0.1, n_estimators=5000,
                                   num_leaves=int(num_leaves), max_depth=int(max_depth), 
                                   bagging_fraction=round(bagging_fraction, 2), feature_fraction=round(feature_fraction, 2),
                                   bagging_freq=int(bagging_freq), min_data_in_leaf=int(min_data_in_leaf),
                                   min_child_weight=min_child_weight, min_split_gain=min_split_gain,
                                   reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                   n_jobs= 8
                                  )
    f1 = make_scorer(f1_score, average='micro')
    val = cross_val_score(model_lgb, X_train, y_train, cv=5, scoring=f1).mean()

    return val
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
"""定义优化参数"""
bayes_lgb = BayesianOptimization(
    rf_cv_lgb, 
    {
        'num_leaves':(10, 200),
        'max_depth':(3, 20),
        'bagging_fraction':(0.5, 1.0),
        'feature_fraction':(0.5, 1.0),
        'bagging_freq':(0, 100),
        'min_data_in_leaf':(10,100),
        'min_child_weight':(0, 10),
        'min_split_gain':(0.0, 1.0),
        'reg_alpha':(0.0, 10),
        'reg_lambda':(0.0, 10),
    }
)

"""开始优化"""
bayes_lgb.maximize(n_iter=10)


bayes_lgb.max


"""调整一个较小的学习率，并通过cv函数确定当前最优的迭代次数"""
base_params_lgb = {
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'num_class': 4,
                    'learning_rate': 0.01,
                    'num_leaves': 124,
                    'max_depth': 13,
                    'min_data_in_leaf': 86,
                    'min_child_weight':8.8,
                    'bagging_fraction': 0.66,
                    'feature_fraction': 0.88,
                    'bagging_freq': 66,
                    'reg_lambda': 3,
                    'reg_alpha': 6.69,
                    'min_split_gain': 0.02,
                    'nthread': 10,
                    'verbose': -1,
}
train_matrix=lgb.Dataset(X_train,label=y_train)
cv_result_lgb = lgb.cv(
    train_set=train_matrix,
    early_stopping_rounds=1000, 
    num_boost_round=20000,
    nfold=5,
    stratified=True,
    shuffle=True,
    params=base_params_lgb,
    feval=f1_score_vali,
    seed=0
)
print('迭代次数{}'.format(len(cv_result_lgb['f1_score-mean'])))#3231
print('最终模型的f1为{}'.format(max(cv_result_lgb['f1_score-mean'])))#0.9552

#stack
from sklearn.model_selection import KFold
from mlxtend.classifier import StackingClassifier


train_X = train.drop(['id','label'], axis=1)
train_y = train['label']

model_final=RandomForestClassifier(n_estimators = 100)
model_final.fit(train_stack,train_y)
pre=model_final.predict_proba(test_stack)
    
model1=RandomForestClassifier(n_estimators = 100)
model2=lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=4,
                                   learning_rate=0.1, n_estimators=100,
                                   num_leaves=124, max_depth=13, 
                                   bagging_fraction=0.66, feature_fraction=0.88,
                                   bagging_freq=66, min_data_in_leaf=86,
                                   min_child_weight=8.8, min_split_gain=0.02,
                                   reg_lambda=3, reg_alpha=6.7,
                                   n_jobs= 8
                                  )   
model3= MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='sgd')

model_final=RandomForestClassifier(n_estimators = 100)

sclf = StackingClassifier(classifiers=[model1, model2, model3], 
                          meta_classifier=model_final,use_probas=True,average_probas=False)
sclf.fit(train_X,train_y)

temp=sclf.predict_proba(X_test)

# 输出预测结果
result=pd.read_csv('sample_submit.csv')
result['label_0']=temp[:,0]
result['label_1']=temp[:,1]
result['label_2']=temp[:,2]
result['label_3']=temp[:,3]
result.to_csv('submit1.csv',index=False)
