#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb


# ### Gather Data

# In[2]:


df = pd.read_csv('MLClassData.csv')
df.shape


# ### Visualize Data ###

# In[5]:


df.iloc[:5,20:30]


# In[6]:


df.columns


# In[8]:


### Define Features to Train Model on ###
features = list(pd.read_csv('FilteredClfFeatsBF100_50.csv')['Clf Features'])
features


# In[9]:


# Check type of each feature is float
print('Data Types:')
for i in features:
    t = df[i].dtype
    if t != float:
        print(i, t)
        features.remove(i)


# In[13]:


# Check for null values 
print('Null Columns:')
for i in features:
    c = df[i].isnull().sum() / float(len(df))
    if c > 0.95:
        print(i, c)
        features.remove(i)


# In[14]:


# Check for zero values
print('Zero Columns:')
for i in features:
    n = (df[i] == 0).astype(int).sum() / float(len(df))
    if n > 0.95:
        print(i, n)
        features.remove(i)


# In[15]:


# Check unique values in each feature
print('Unique Values:')
for i in features:
    u = len(df[i].unique()) / float(len(df))
    if u < 0.1:
        print(i, u)


# ### Label Data with Different Classes

# In[16]:


### Define Objective and Label ###
df['maxreturn'] = (df.high - df.entry_price) / df.zv_length
df['label'] = (df.maxreturn > 0.4).astype(int)


# In[21]:


### Look at return details ###
print('Percent of Target Trades:', df.label.sum() / float(len(df)))


# ### Split Data into Training and Testing Sets

# In[36]:


### Split the data into training, testing sets ###
train = df[(pd.to_datetime(df.entry_time) < pd.to_datetime('2017-01-01')) & 
           (pd.to_datetime(df.entry_time) >= pd.to_datetime('2014-01-01'))]
test = df[pd.to_datetime(df.entry_time) >= pd.to_datetime('2017-01-01')]

X_train = train.loc[:,features]
y_train = train.loc[:,'label']

X_test = test.loc[:,features]
y_test = test.loc[:,'label']

print('Length of Training Set: ' + str(len(train)))
print('Length of Testing Set: ' + str(len(test)))


# ### Clean and Standardize Data

# In[23]:


### Remove null values and replace with the median of each column ###
transform = Imputer(missing_values='NaN',strategy='median')

### Find median of each column in training set and replace null values ###
X_train = transform.fit_transform(X_train)

### Apply median from training set to null values of test set ###
X_test = transform.transform(X_test)


# ### Train Random Forest

# In[29]:


### Train and fit classifier to Training Data ###
clf = RandomForestClassifier(n_estimators=200, max_depth=10, max_leaf_nodes=100, n_jobs=-1, verbose=1)

clf.fit(X_train, y_train)


# In[30]:


y_pred = clf.predict_proba(X_test)
y_pred.sum()


# In[31]:


trades = test.assign(pred = y_pred)
trades = trades[trades.pred > 0.5]


# In[32]:


print('Original Max Return - No ML')
print(test.maxreturn.describe())
print('---------------')
print('Max Return with ML')
print(trades.maxreturn.describe())


# In[33]:


trades = trades.sort_values('entry_time')
trades['cpl'] = trades.mtm_pl.cumsum()
test = test.sort_values('entry_time')
test['cpl'] = test.mtm_pl.cumsum()
print('Original PnL - No ML')
test.plot(x='entry_time',y='cpl',figsize=[12,8])
plt.show()
print('----------------')
print('PnL with ML')
trades.plot(x='entry_time',y='cpl',figsize=[12,8])
plt.show()


# In[34]:


### Feature Importances ###
importances = sorted(list(zip(clf.feature_importances_, features)),reverse=True)
for i, f in importances:
    print(f, i)


# ## Gradient Boosted Model

# In[106]:


data = lgb.Dataset(X_train, y_train)
param = {'max_depth':6, 'num_leaves':10, 'learning_rate':0.01, 'num_threads':-1,
         'min_data_in_leaf':100, 'objective':'binary', 'metric':'binary_logloss','verbosity':-1}
bst = lgb.train(param, data, 500)


# In[107]:


y_pred = bst.predict(X_test)
y_pred.sum()


# In[ ]:



