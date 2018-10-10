#Original file and other CSVs can be found at: https://info.cloudquant.com/2018/05/machine-learning-fxcm-webinar-with-trevor-trinkino-of-cloudquant-part-2-3/
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

#==========================================
# ### Gather Data
# ### Visualize Data ###
#==========================================

df = pd.read_csv('./ml-trevor-trinko/MLClassData.csv')
df.shape

df.iloc[:5,20:30]

df.columns


### Define Features to Train Model on ###
features = list(pd.read_csv('./ml-trevor-trinko/FilteredClfFeatsBF100_50.csv')['Clf Features'])
features



#==========================================
## Check type of each feature is float
## Check for null values 
## Check for zero values
## Check unique values in each feature
#==========================================

# print('Data Types:')
# for i in features:
#     t = df[i].dtype
#     if t != float:
#         print(i, t)
#         features.remove(i)

# print('Null Columns:')
# for i in features:
#     c = df[i].isnull().sum() / float(len(df))
#     if c > 0.95:
#         print(i, c)
#         features.remove(i)

# print('Zero Columns:')
# for i in features:
#     n = (df[i] == 0).astype(int).sum() / float(len(df))
#     if n > 0.95:
#         print(i, n)
#         features.remove(i)


# print('Unique Values:')
# for i in features:
#     u = len(df[i].unique()) / float(len(df))
#     if u < 0.1:
#         print(i, u)


#=========================
# ###                                    Label Data with Different Classes
#Unsupervised ML: No labeling, just finds patterns, clusters etc
#Supervised ML: Classifies everything to 'good' or 'bad'

#Here the goal is to classify all trades to good or bad trades.
#=========================


### 1st: We Define Objective and Label ###

## Our objective is: find any rebounding greater than 40% of the daily move. 
df['maxreturn'] = (df.high - df.entry_price) / df.zv_length
## To classify: If it has, label as a 1, if not, label as a 0.
df['label'] = (df.maxreturn > 0.4).astype(int)

##If we want to see a table of our results, where 'label' column is the labels we made
#print df[['symbol', 'entry_time', 'label']]


##Furthermore, if we want to see what percent 'passed' from the total amount of trades
#print('Percent of Target Trades:', df.label.sum() / float(len(df)))


### 2nd: Split the data into training, testing sets ###
train = df[(pd.to_datetime(df.entry_time) < pd.to_datetime('2017-01-01')) & 
           (pd.to_datetime(df.entry_time) >= pd.to_datetime('2014-01-01'))]
test = df[pd.to_datetime(df.entry_time) >= pd.to_datetime('2017-01-01')]


##.loc: In train.loc[:,'label'], it is setting the entire column labeled 'label' equal to the value of y=train
##What it learns on:
X_train = train.loc[:,features]
##What it splits on:
y_train = train.loc[:,'label']

##What it predicts on:
X_test = test.loc[:,features]
##Will validate how well we did:
y_test = test.loc[:,'label']

##Show the number of trades in training set and testing set
#print('Length of Training Set: ' + str(len(train)))
#print('Length of Testing Set: ' + str(len(test)))
print 'x test df:', X_test
#print 'y test df:', y_test.head()

# ### Clean and Standardize Data



### Remove null values and replace with the median of each column so it doesn't throw errors; in reality you should
# IRL: go through the each column and determine which is the best null value replacement
transform = Imputer(missing_values='NaN',strategy='median')

##Fit(): Method calculates the parameters mean and s.d. and saves them as internal objects.
##Transform(): Method using these calculated parameters apply the transformation to a particular dataset.
##Fit_transform(): joins the fit() and transform() method for transformation of dataset.
### Find median of each column in training set and replace null values ###
X_train = transform.fit_transform(X_train)
### Apply median from training set to null values of test set ###
##Don't use fit_transform because it would find new medians for 2017+ data. IRL this is not possible since
##it would involve looking into the future.
X_test = transform.transform(X_test)


# ### Train Random Forest

### Train and fit classifier to Training Data ###
##Create a random forest classifier object with clf = RandomForestClassifier()
# 200 decision trees (with say 7 features and 50 000 trades each), the results being averaged out to get final prediction
# Greater the depth, greater the risk of overfitting
# Nodes: the amount of splits creats 2x nodes for every rule set
# n_jobs: amount of cores   
clf = RandomForestClassifier(n_estimators=200, max_depth=10, max_leaf_nodes=100, n_jobs=-1, verbose=1)

##This is telling it to find patterns in the X training data set that allow it to split on Y train label most effectively
clf.fit(X_train, y_train)

##After running, clf will have all the nested 'if' statements
##With .predict we are telling it to take the rules saved and found from the X_train data and using the new data (X_test) assign
##either a 0 or 1 and match the label that it trained on in the .fit line
y_pred = clf.predict_proba(X_test)
##This results in ~11000 positive results, out of ~69000 entries in the testing set. This ratio compares to the
##27% returned in line 89 ("print('Percent of Target Trades:'...").
y_pred.sum()


trades = test.assign(pred = y_pred)
trades = trades[trades.pred > 0.5]

## (For RandomForestClassifier without hyper parameters) This will print mean = 0.309639 which means in the 69200 trades,
## every stock had rebounded at an average max bounce of 30.9% of it's daily range on average sometime during the day
# print('Original Max Return - No ML')
# print(test.maxreturn.describe())
# print('---------------')

## (For RandomForestClassifier without hyper parameters) This will print a mean = 0.456872 (in 11511 trades), an improvement thanks to m.l.
# print('Max Return with ML')
# print(trades.maxreturn.describe())

trades = trades.sort_values('entry_time')
trades['cpl'] = trades.mtm_pl.cumsum()
test = test.sort_values('entry_time')
test['cpl'] = test.mtm_pl.cumsum()
#print('Original PnL - No ML')
test.plot(x='entry_time',y='cpl',figsize=[12,8])
#plt.show()
#print('----------------')
#print('PnL with ML')
trades.plot(x='entry_time',y='cpl',figsize=[12,8])
plt.show()

### Feature Importances ###
importances = sorted(list(zip(clf.feature_importances_, features)),reverse=True)
for i, f in importances:
    print(f, i)


# ## Gradient Boosted Model

data = lgb.Dataset(X_train, y_train)
param = {'max_depth':6, 'num_leaves':10, 'learning_rate':0.01, 'num_threads':-1,
         'min_data_in_leaf':100, 'objective':'binary', 'metric':'binary_logloss','verbosity':-1}
bst = lgb.train(param, data, 500)

y_pred = bst.predict(X_test)
y_pred.sum()