#Min-Max Scaling: x(i,j) = x(i,j)-x(min j)/ x(max j) - x(min j)
#An alternative to normalizing through s.d.

import pandas as pd
import plotly as py
from plotly import tools
import plotly.graph_objs as go

df= pd.read_csv('./Data/EURUSD_Candlestick_1_Hour_ASK_31.12.2016-12.05.2017.csv')
df= pd.read_csv('./Data/USDTRY_Candlestick_1_Hour_ASK_31.12.2016-15.10.2018.csv')

df.columns = ['date','open','high','low','close','volume']

#Format date, set it as the index and remove it as a column
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]

#The market isn't always live, so it leads to many hours with identical data. Remove this.
df = df.drop_duplicates(keep=False)
#print(df.head())

#Moving Average
ma = df.close.rolling(center=False, window=30).mean()

#Trace: a set of data for plottimg

trace = go.Ohlc(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name="Currency Quote")
trace1 = go.Scatter(x=df.index, y=ma)
trace2 = go.Bar(x=df.index,y=df.volume)

data = [trace,trace1,trace2]

fig = tools.make_subplots(rows=2,cols=1, shared_xaxes=True)
fig.append_trace(trace,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)


py.offline.plot(fig, filename='tutorial.html')