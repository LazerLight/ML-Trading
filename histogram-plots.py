import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#==============================
# Start Aux. Functions
#==============================

def symbol_to_path(symbol, base_dir="data"):
    #Return CSV file path given ticker symbol
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols,dates):
    #Read stock data for given symbols from CSV files
    df =pd.DataFrame(index=dates)

    if 'SPY2018' not in symbols: #add SPY for reference,absent
        symbols.insert(0,'SPY2018')
    
    for symbol in symbols:
        df_temp=pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True,usecols=['Date','Adj Close'],na_values=['nan'])

        #Rename to prevent column name overlap
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df=df.join(df_temp)
        if symbol == 'SPY2018': #Drop dates SPY didn't trade
            df = df.dropna(subset=['SPY2018'])
    return df


def plot_data(df, title="Stock Prices", ylabel="Price"):

    ax = df.plot(title=title,fontsize=5)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.show()

def compute_daily_returns(df):
    #Using Pandas

    daily_returns = (df/df.shift(1)) - 1 #
    daily_returns=daily_returns.dropna()
    return daily_returns


#==============================
# End Aux. Functions
#=============================


#Histograms
    #For daily returns, it is a normal/gaussian curve
    #Kurtosis: describes the tails. 
    # Positive Kurtosis; 'Fat tails', Many occurences
    # Negative Kurtosis; Skinny tails': Less occurences

#Plotting Histograms of Daily Returns

def test_run():
    dates = pd.date_range('2017-09-01', '2018-09-01')
    symbols= ['SPY2018']
    df = get_data(symbols, dates)
    #plot_data(df)

    #Compute daily returns
    daily_returns = compute_daily_returns(df)
    #plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

    #Plot a histogram, changing the number of 'bins' to 20
    daily_returns.hist(bins=40)
    #plt.show()

    #Get mean and standard deviation
    mean = daily_returns['SPY2018'].mean()
    print "mean=",mean
    std = daily_returns['SPY2018'].std()
    print "std=,",std

    #Insert a vertical line indicating the mean
    plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.show()

    #Compute kurtosis
    print daily_returns.kurtosis()

#Plotting Two Histograms of Daily Returns

def test_run1():

    # Read data
    dates = pd.date_range('2017-09-01', '2018-09-01')
    symbols= ['SPY2018', 'GE']
    df = get_data(symbols, dates)

    #Compute daily returns
    daily_returns = compute_daily_returns(df)

    #Plot
    daily_returns['SPY2018'].hist(bins=20, label="SPY2018", alpha=0.5)
    daily_returns['GE'].hist(bins=20, label="GE", alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()






#Scatterplots
#The difference between daily returns between SPY and a stock is plotted.
#The scatterplot (SPY as the x-axies) will have: 
#Alpha: the y-intercept. A positive value means it's outperforming the market
#Beta: the slope of the line. Beta greater than one means its more volatile than the market and vice-versa.
#Beta/slope isn't not the correlation. Correlation is the tightness of the line.

def test_run2():

    # Read data
    dates = pd.date_range('2017-09-27', '2018-09-23')
    symbols= ['SPY2018', 'GE']
    df = get_data(symbols, dates)

    #Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns", ylabel="Daily returns")

def test_run3():

    # Read data
    dates = pd.date_range('2017-09-27', '2018-09-23')
    symbols= ['SPY2018', 'V','GE']
    df = get_data(symbols, dates)

    #Compute daily returns
    daily_returns = compute_daily_returns(df)

    #Scatterplot SPY vs Visa
    daily_returns.plot(kind='scatter', x='SPY2018', y='V')
    # Polynomial fit of degree 1. Require x and y coordinate. 
    # Returns polynomial coefficient and y intercept. In y = mx + b, m and b respectively 
    beta_V, alpha_V = np.polyfit(daily_returns['SPY2018'], daily_returns['V'], 1)
    print "beta_V=", beta_V
    print "alpha_V=", alpha_V
    plt.plot(daily_returns['SPY2018'], beta_V*daily_returns['SPY2018'] + alpha_V, '-', color='r')
    #plt.show()

    #Scatterplot SPY vs GE
    daily_returns.plot(kind='scatter', x='SPY2018', y='GE')
    beta_GE, alpha_GE = np.polyfit(daily_returns['SPY2018'], daily_returns['GE'], 1)
    print "beta_GE=", beta_GE
    print "alpha_GE=", alpha_GE
    plt.plot(daily_returns['SPY2018'], beta_GE*daily_returns['SPY2018'] + alpha_GE, '-', color='r')
    #plt.show()

    #Calculate correlation coefficient
    print daily_returns.corr(method='pearson')
test_run3()