import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time

# Since the file is very big, storing it in memory is costly. Therefore make it global and store it only once.
date,bid,ask = np.loadtxt('../data/GBPUSD1d.txt', unpack=True, delimiter=',', converters={0:mdates.strpdate2num('%Y%m%d%H%M%S')}) #Convert date stamps

def percentChange(startPt, currentPt):
    return ((float(currentPt) - startPt)/startPt)*100 #Return a float, not int


def patternStoring():
    '''
    The goal of patternFinder is to begin collection of %change patterns
    in the tick data. From there, we also collect the short-term outcome
    of this pattern. Later on, the length of the pattern, how far out we
    look to, compare to, and the length of the compared range be changed,
    and even THAT can be machine learned to find the best of all 3 by
    comparing success rates.
    '''
    
    #Simple Average of every bid/ask of every entry
    #To simplify referencing bid/ask 
    avgLine = ((bid+ask)/2)
    
    #This finds the length of the total array for us
    #Amount of entries in the text file
    x = len(avgLine)-30
    #This will be our starting point, allowing us to compare to the
    #past 10 % changes. 
    y = 11
    # where we are in a trade. #
    # can be none, buy,
    currentStance = 'none'
    while y < x:
        
        p1 = percentChange(avgLine[y-10], avgLine[y-9])
        p2 = percentChange(avgLine[y-10], avgLine[y-8])
        p3 = percentChange(avgLine[y-10], avgLine[y-7])
        p4 = percentChange(avgLine[y-10], avgLine[y-6])
        p5 = percentChange(avgLine[y-10], avgLine[y-5])
        p6 = percentChange(avgLine[y-10], avgLine[y-4])
        p7 = percentChange(avgLine[y-10], avgLine[y-3])
        p8 = percentChange(avgLine[y-10], avgLine[y-2])
        p9 = percentChange(avgLine[y-10], avgLine[y-1])
        p10= percentChange(avgLine[y-10], avgLine[y])
        
        #Outcome: Average of 20-30 points in the future.
        outcomeRange = avgLine[y+20:y+30]
        currentPoint = avgLine[y]

        #function to account for the average of the items in the array
        print reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)

        
        print currentPoint
        print '_______'
        print p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
        time.sleep(55)
        
        y+=1

patternStoring()