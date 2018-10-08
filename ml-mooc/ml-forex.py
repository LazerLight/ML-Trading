import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import os

def graphRaw():
    date, AdjClose = np.loadtxt('./data/XAU.csv', unpack=True, delimiter=',', converters = {0: mdates.datestr2num})

    fig = plt.figure(figsize=(10,7))
    ax1 = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40)
    
    ax1.plot(Date,AdjClose)

    plt.grid(True)
    plt.show()


graphRaw()