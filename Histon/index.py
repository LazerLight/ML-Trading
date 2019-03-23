# This Python file uses the following encoding: utf-8
# https://old.reddit.com/r/stocks/comments/5mfdjk/howto_technical_trading_using_python_and_machine/
import pandas as pd
import numpy as np
import talib
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


import datetime

'''
===========================
1) Gather Financial Data
===========================
'''
#Download data from yahoo finance
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2014,3,24)
ticker = "AAPL"
f=web.DataReader(ticker,'yahoo',start,end)
f.Volume = f.Volume.astype(float)


'''
===========================
2) Select the features
===========================
'''

f['SMA_20'] = talib.SMA(np.asarray(f['Close']), 20)
f['SMA_50'] = talib.SMA(np.asarray(f['Close']), 50)
f['UBB'], f['MBB'], f['LBB'] = talib.BBANDS(np.asarray(f['Close']), timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
f.plot(y= ['UBB','MBB', 'LBB'], title='AAPL Close & Moving Averages')


'''
===========================
3) Select the target
===========================
'''

# Create forward looking column (because we are trying to predict the next day price)
f['NextDayPrice'] = f['Close'].shift(-1)

# print(f[['Close','NextDayPrice']])


'''
===========================
4) Clean up your data
===========================
'''
#Multiple ways of cleaning up. For starters, remove NA values
f_cleanData = f.copy()
f_cleanData.dropna(inplace=True)

#f_cleanData.to_csv('output', sep='\t', encoding='utf-8')

'''
===========================
5) Split Data into Training and Testing Set
===========================
'''
#Since we are trying to predict a continuous value from labeled data, this is considered a supervised learning regression model. 
#If we were trying to predict a label or categorical data, it would be considered classification.  

X_all = f_cleanData.ix[:, f_cleanData.columns != 'NextDayPrice']
y_all = f_cleanData['NextDayPrice']
#print (y_all.head()) #Print the first 5 rows

#Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.30, random_state=35)


'''
===========================
6) Train the Model
===========================
'''


#Create a decision tree regressor and fit it to the training set
regressor = LinearRegression()

regressor.fit(X_train,y_train)

print ("Training set: {} samples".format(X_train.shape[0]))
print ("Test set: {} samples".format(X_test.shape[0]))
print(X_test.head())


'''
===========================
7) Evaluate the Model
===========================
'''
# Evaluate the performance of our model. The metrics you use is up to you. 
#Accuracy and Mean Squared Error are shown below.  
# scores = cross_val_score(regressor, X_test, y_test, cv=10)
# print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))    


# mse = mean_squared_error(y_test, regressor.predict(X_test))
# print("MSE: %.4f" % mse)

'''
===========================
7) Predict
===========================
'''

# X=f_cleanData[-1:]
# print(regressor.predict(X))

from iexfinance.stocks import get_historical_data 
from datetime import datetime

start = datetime(2017, 1, 1) # starting date: year-month-date
end = datetime(2018, 1, 1) # ending date: year-month-date

data = get_historical_data('ERIC', start=start, end=end, output_format='pandas') 
print(data.head())