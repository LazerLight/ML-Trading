import pandas as pd 
import matplotlib.pyplot as plt 

def test_run():
    df = pd.read_csv("ANF.csv")
    print df[['Adj Close','Close']]
    df[['Adj Close','Close']].plot()
    plt.show()


# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np

# x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
# plt.plot(x, np.sin(x))       # Plot the sine of each x point
# plt.show()                   # Display the plot


#Create an Empty Data Frame

def test_run1():
    start_date='2010-01-22'
    end_date='2010-01-26'
    dates=pd.date_range(start_date,end_date)
    df1=pd.DataFrame(index=dates)
    print(df1)

#test_run1()

#Join SPY Data

def test_run2():
    #Define date range
    start_date='2018-08-31'
    end_date='2018-09-06'
    dates=pd.date_range(start_date,end_date)

    #Create an empty dataframe
    df1=pd.DataFrame(index=dates)

    #Read SPY data into temporary dataframe.
    #index_col sets the dates as the index, eliminating array index numbering
    #parse_dates makes the dates into date time index objects
    #na_values gives values to NaN
    dfSPY = pd.read_csv("ANF.csv", index_col="Date",parse_dates=True,usecols=['Date','Adj Close'], na_values=['nan'])

    #Join the two dataframes using DataFrame.join()
    df1=df1.join(dfSPY)

    # Drop NaN Values
    df1 = df1.dropna()
    print df1

#test_run2()

#Read Multiple Stocks
def test_run3():
    #Define date range
    start_date='2018-08-31'
    end_date='2018-09-06'
    dates=pd.date_range(start_date,end_date)

    #Create an empty dataframe
    df1=pd.DataFrame(index=dates)

    #Read SPY data into temporary dataframe.
    #index_col sets the dates as the index, eliminating array index numbering
    #parse_dates makes the dates into date time index objects
    #na_values gives values to NaN
    dfSPY = pd.read_csv("ANF.csv", index_col="Date",parse_dates=True,usecols=['Date','Adj Close'], na_values=['nan'])

    #Rename 'Adj Close' column to 'SPY' to prevent clash
    dfSPY = dfSPY.rename(columns={'Adj Close': 'ANF'})

    #Join the two dataframes using DataFrame.join() with 'how' set to 'inner'
    df1=df1.join(dfSPY,how='inner')

    # Read in more stocks
    symbols = ['JWN']
    for symbol in symbols:
        df_temp=pd.read_csv("{}.csv".format(symbol), index_col='Date', parse_dates=True,usecols=['Date','Adj Close'],na_values=['nan'])

        #Rename to prevent column name overlap
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df1=df1.join(df_temp) #use default how='left'

    print df1

#
#Utility Functions
#

import os

def symbol_to_path(symbol, base_dir="data"):
    #Return CSV file path given ticker symbol
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols,dates):
    #Read stock data for given symbols from CSV files
    df =pd.DataFrame(index=dates)

    if 'SPY' not in symbols: #add SPY for reference,absent
        symbols.insert(0,'SPY')
    
    for symbol in symbols:
        df_temp=pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True,usecols=['Date','Adj Close'],na_values=['nan'])

        #Rename to prevent column name overlap
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df=df.join(df_temp)
        if symbol == 'SPY': #Drop dates SPY didn't trade
            df = df.dropna(subset=['SPY'])
    return df

#Slicing By Rows and/or Columns
def test_run4():
    #Define a date range
    dates = pd.date_range('2017-09-01', '2018-09-01')

    #Pick the symbols
    symbols= ['G', 'GE', 'PVH', 'V']

    #Get Stock data
    df = get_data(symbols, dates)
  
    #Slice by rows (dates) using DataFrame.ix[] selector
    return df['2017-10-01':'2018-08-01']

    #Slice by column (symbols)
    # print df['V']
    # print df[['PVH','GE']]

    # #Slice by row and column
    # return df.ix['2018-01-15':'2018-02-15', ['V','PVH']]

# test_run4()


# Normalizing Stocks
def normalize_data(df):
    return df/df.ix[0,:]
    
#Plotting Multiple Stocks
def plot_data(df, title="Stock Prices"):

    ax = df.plot(title=title,fontsize=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

plot_data(test_run4())

