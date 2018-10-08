#Computing Global Statistics for Each Stock
import pandas as pd 
import matplotlib.pyplot as plt 
import os

#==============================
# Start Aux. Functions
#==============================

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



def plot_data(df, title="Stock Prices"):

    ax = df.plot(title=title,fontsize=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

#==============================
# End Aux. Functions
#==============================

def test_run():
    #Read data
    dates = pd.date_range('2017-09-01', '2018-09-01')
    symbols= ['G', 'GE', 'PVH', 'V']
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute global stats for each stock
    #Get mean, median, s.d. for each stock
    print df.mean()
    print df.median()
    print df.std()

#Compute Rolling Statistics
    #rolling_mean isn't a data frame method, it's a pandas method
    #Therefore you can't just call df.rolling_mean
def test_run1():
    #Read data
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols= ['G', 'GE', 'PVH', 'V']
    df = get_data(symbols, dates)

    #Plot SPY data, retain matplotlib axis object
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')

    #Compute rolling mean using a 20-day window
    rm_SPY = df['SPY'].rolling(20).mean()

    #Add rolling mean to the same plot
    rm_SPY.plot(label='Rolling mean', ax=ax)

    #Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    plt.show()


#Compute Bollinger Bands

    #==============================
    # Start Aux. Functions
    #==============================
def get_rolling_mean(values, window):
    #Return rolling mean of given values, using specified window size
    return values.rolling(window).mean()

def get_rolling_std(values, window):
    #Return rolling s.d of given values, give a specified window size
    return values.rolling(window).std()

def get_bollinger_bands(rm, rstd):
    # Return upper and lower Bollinger Bands (+/- 2 s.d) 
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2

    return upper_band, lower_band

    #==============================
    # End Aux. Functions
    #==============================


def test_run2():
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols= ['SPY']
    df = get_data(symbols, dates)
    
    #Step one: Compute rolling mean
    rm_SPY = get_rolling_mean(df['SPY'], 20)
    #Step two: Compute rolling std
    rstd_SPY = get_rolling_std(df['SPY'], 20)

    #Step three: Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    #Plot SPY values, rolling mean and Bollinger Bands
    ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
    rm_SPY.plot(label='Rolling Mean', ax=ax)
    upper_band.plot(label='Upper B.B.', ax=ax)
    lower_band.plot(label='Lowe B.B', ax=ax)

    #Add axis labels and a legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    plt.show()

#Daily Returns
# Equal to (stock price) / (stock price from day before) - 1
def compute_daily_returns_alternative(df):
    daily_returns = df.copy()
    # Compute daily returns from row 1 onwards. 
    # Denominator indicates the range from beginning to end, excluding the final value
    daily_returns[1:0] = (df[1:0] / df[:-1].values) - 1
    daily_returns.ix[0,:] = 0 #Set daily returns for first row to 0
    daily_returns = daily_returns.dropna()

    return daily_returns

def compute_daily_returns(df):
    #Using Pandas

    daily_returns = (df/df.shift(1)) - 1 #
    return daily_returns

def cdr():
    dates = pd.date_range('2012-07-03', '2012-07-30')
    symbols= ['SPY']
    df = get_data(symbols, dates)
    plot_data(compute_daily_returns(df))


#Cumulative Returns
# Equal to the stock price at a certain date / stock price on a start day of the year

def compute_cumulative_returns(df):
    #Using Pandas
    print df[1:0].values
    cumulative_returns = (df/df[0]) - 1 #
    return cumulative_returns

def ccr():
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols= ['SPY']
    df = get_data(symbols, dates)
    plot_data(compute_cumulative_returns(df))

ccr()