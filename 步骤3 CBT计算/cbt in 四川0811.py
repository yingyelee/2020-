#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import scipy.spatial.distance as dist
import catboost as cbt
import json
from sklearn.metrics import f1_score
import time

import gc
import math
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from six.moves import reduce
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime,timedelta


# In[2]:


os.listdir(r'H:\compitition\sichuan-fusai\cbt')
submission = pd.read_csv(r'H:\compitition\sichuan\cbt\submit_example.csv')
train = pd.read_csv(r'H:\compitition\sichuan-fusai\cbt/train_user4.csv')
test = pd.read_csv(r'H:\compitition\sichuan-fusai\cbt/test_user4.csv')


# In[3]:


data = train.append(test)


# In[4]:


data.shape


# In[5]:


cat_list = [i for i in train.columns if i in ['city_name','county_name','idcard_cnt']]
for i in tqdm(cat_list):
    data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['ID'].transform('count')
feature_name = [i for i in data.columns if i not in ['ID','Label']]


# In[7]:


tr_index = ~data['Label'].isnull()
X_train = data[tr_index][list(set(feature_name))].reset_index(drop=True)
y = data[tr_index]['Label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][list(set(feature_name))].reset_index(drop=True)
print(X_train.shape,X_test.shape)
oof = np.zeros(X_train.shape[0])
prediction = np.zeros(X_test.shape[0])
seeds = [19970412, 2019 * 2 + 1024, 4096, 2018, 1024]
num_model_seed = 5
for model_seed in range(num_model_seed):
    oof_cat = np.zeros(X_train.shape[0])
    prediction_cat=np.zeros(X_test.shape[0])
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        cbt_model = cbt.CatBoostClassifier(iterations=8000,learning_rate=0.08,max_depth=8,verbose=100,
                                       early_stopping_rounds=500,task_type='GPU',eval_metric='AUC',
                                       cat_features=cat_list)
        cbt_model.fit(train_x[feature_name], train_y,eval_set=(test_x[feature_name],test_y))
        gc.collect()    
        oof_cat[test_index] += cbt_model.predict_proba(test_x)[:,1]
        prediction_cat += cbt_model.predict_proba(X_test[feature_name])[:,1]/5   
    oof += oof_cat / num_model_seed
    prediction += prediction_cat / num_model_seed
print('score',f1_score(y, np.round(oof)))    


# In[8]:


submit = test[['ID']]
sub["label"] = sub["prob"] > round(np.percentile(sub["prob"], threshold), 4)
submit['label'] = (prediction>=0.499).astype(int)  #以0.5为界
submit['label'].replace(0,'FALSE',inplace = True)
submit['label'].replace(1,'True',inplace = True)
submit.rename(columns={'ID':'phone_no_m'})
print(submit['label'].value_counts())
submit.to_csv("submission.csv",index=False)


# In[9]:


import matplotlib.pyplot as plt 


plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

fea_ = cbt_model.feature_importances_
fea_name = cbt_model.feature_names_
plt.figure(figsize=(10, 100))
plt.barh(fea_name,fea_,height =1)


# In[12]:



submission['Label'] = prediction

submission.to_csv("submission0416cat.csv",index=False)

