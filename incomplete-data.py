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



def plot_data(df, title="Stock Prices"):

    ax = df.plot(title=title,fontsize=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

#==============================
# End Aux. Functions
#==============================

#Using fillna() to Fill In Missing Data
symbols = ["FAKE"]
start_date = "2017-09-25"
end_date = "2018-09-24"

idx=pd.date_range(start_date, end_date)

df_data=get_data(symbols,idx)

#This will 'forward fill' missing data, then 'back fill'
#Do forward then backward fill to avoid 'peeping into the future' as much as possible
df_data.fillna(method="ffill", inplace=True)
df_data.fillna(method="bfill", inplace=True)


plot_data(df_data)