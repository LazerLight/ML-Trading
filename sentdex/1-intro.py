#Probably one of the largest drawbacks to the Python programming languages is that it is single-threaded. 
#Python may take a very long time to reach a solution, There are quite a few solutions to this problem, like 
#threading, multiprocessing, and GPU programming. All of these are possible with Python, and today we will be 
#covering threading. 
#https://pythonprogramming.net/threading-tutorial-python/

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from matplotlib import style
style.use("ggplot")




def graphRawFX():
    date,bid,ask = np.loadtxt('./data/GBPUSD1d.txt', unpack=True,
                              delimiter=',',
                              converters={0:mdates.strpdate2num('%Y%m%d%H%M%S')}) #Convert date stamps

    fig=plt.figure(figsize=(10,7))

    ax1 = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40)
    ax1.plot(date,bid)
    ax1.plot(date,ask)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)


    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))#Format x-axis for dates

    for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)

    ax_2 = ax1.twinx()
    ax_2.fill_between(date, 0, (ask-bid),facecolor='g', alpha=0.3)
    #0 refers to the minimum it would fill under
    #ask-bid is the 'spread'
    #facecolor=g is the color green

    plt.subplots_adjust(bottom=.23)
    plt.grid(True)
    plt.show()

graphRawFX()