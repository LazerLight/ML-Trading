'''
so we suspect that possibly 

To compare patterns:
use a % change calculation to calculate similarity between each %change
movement in the pattern finder. From those numbers, subtract them from 100, to
get a "how similar" #. From this point, take all 10 of the how similars,
and average them. Whichever pattern is MOST similar, is the one we will assume
we have found. 
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from numpy import loadtxt
import time

# Since the file is very big, storing it in memory is costly. Therefore make it global and store it only once.
date,bid,ask = np.loadtxt('../data/GBPUSD1d.txt', unpack=True, delimiter=',', converters={0:mdates.strpdate2num('%Y%m%d%H%M%S')}) #Convert date stamps

avgLine = ((bid+ask)/2)


#As we run pattern finder, every pattern will be stored here
patternAr = []
#Store outcomes
performanceAr = []
#Pattern for recognition 
patForRec = []


def percentChange(startPoint,currentPoint):
    
    return ((float(currentPoint)-startPoint)/abs(startPoint))*100.00



def patternStorage():
    '''
    The goal of patternStorage is to collect %change in the tick data. From there, we also collect the short-term outcome
    of this pattern. 
    The length of the pattern, how far out we look/compare to, 
    and the length of the compared range be changed.
    All that can be machine learned to find the best of all 3 by comparing success rates.
    '''

    #For displaying calculation time
    startTime = time.time()

    # required to do a pattern array, because the liklihood of an identical
    # %change across millions of patterns is fairly likely and would
    # cause problems. IF it was a problem of identical patterns,
    # then it wouldnt matter, but the % change issue
    # would cause a lot of harm. Cannot have a list as a dictionary Key.
    
    #MOVE THE ARRAYS THEMSELVES#
    
    avgLine = ((bid+ask)/2)
    x = len(avgLine)-30
    y = 11
    currentStance = 'none'
    
    while y < x:
        pattern = []
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

        outcomeRange = avgLine[y+20:y+30]
        currentPoint = avgLine[y]


        try:
            avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
        #Exception to avoid negative infinities
        except Exception, e:
            print str(e)
            avgOutcome = 0
        #Return futureOutcome as a percent change instead of an actual number
        futureOutcome = percentChange(currentPoint, avgOutcome)

        #print some logics
        '''
        print 'where we are historically:',currentPoint
        print 'soft outcome of the horizon:',avgOutcome
        print 'This pattern brings a future change of:',futureOutcome
        '''

        #This will append to the empty array started in the line after the while loop.
        pattern.append(p1)
        pattern.append(p2)
        pattern.append(p3)
        pattern.append(p4)
        pattern.append(p5)
        pattern.append(p6)
        pattern.append(p7)
        pattern.append(p8)
        pattern.append(p9)
        pattern.append(p10)

        patternAr.append(pattern)
        performanceAr.append(futureOutcome) #patternAr and performanceAr should be of = length
        
        y+=1
    #####
    endTime = time.time()
    print len(patternAr), len(performanceAr), 'Pattern storing took:', endTime-startTime
    #####


        #can use .index to find the index value, then search for that value to get the matching information.
        # so like, performanceAr.index(12341)


def currentPattern():
    #mostRecentPoint = avgLine[-1]

    cp1 = percentChange(avgLine[-11],avgLine[-10])
    cp2 = percentChange(avgLine[-11],avgLine[-9])
    cp3 = percentChange(avgLine[-11],avgLine[-8])
    cp4 = percentChange(avgLine[-11],avgLine[-7])
    cp5 = percentChange(avgLine[-11],avgLine[-6])
    cp6 = percentChange(avgLine[-11],avgLine[-5])
    cp7 = percentChange(avgLine[-11],avgLine[-4])
    cp8 = percentChange(avgLine[-11],avgLine[-3])
    cp9 = percentChange(avgLine[-11],avgLine[-2])
    cp10= percentChange(avgLine[-11],avgLine[-1])

    patForRec.append(cp1)
    patForRec.append(cp2)
    patForRec.append(cp3)
    patForRec.append(cp4)
    patForRec.append(cp5)
    patForRec.append(cp6)
    patForRec.append(cp7)
    patForRec.append(cp8)
    patForRec.append(cp9)
    patForRec.append(cp10)

    print patForRec

def patternRecognition():
    #To compare
    for eachPattern in patterAr:
        #How similar are we
        sim1 = 100.00 - abs(percentChange(eachPattern[0], patForRec[0]))
        sim2 = 100.00 - abs(percentChange(eachPattern[1], patForRec[1]))
        sim3 = 100.00 - abs(percentChange(eachPattern[2], patForRec[2]))
        sim4 = 100.00 - abs(percentChange(eachPattern[3], patForRec[3]))
        sim5 = 100.00 - abs(percentChange(eachPattern[4], patForRec[4]))
        sim6 = 100.00 - abs(percentChange(eachPattern[5], patForRec[5]))
        sim7 = 100.00 - abs(percentChange(eachPattern[6], patForRec[6]))
        sim8 = 100.00 - abs(percentChange(eachPattern[7], patForRec[7]))
        sim9 = 100.00 - abs(percentChange(eachPattern[8], patForRec[8]))
        sim10 = 100.00 - abs(percentChange(eachPattern[9], patForRec[9]))

        #Average them
        howSim = (sim1+sim2+sim3+sim4+sim5+sim6+sim7+sim8+sim9+sim10)/10.00


    #We only want similarities over 70%
    if howSim > 70:
        #Find the index
        patdex = patternAr.index(eachPattern)
        print patdex
        
        print '##################################'
        print '##################################'
        print '##################################'
        print '##################################'
        print patForRec
        print '==================================='
        print '==================================='
        print eachPattern
        print '----------'
        print 'predicted outcome:',performanceAr[patdex]
        print '##################################'
        print '##################################'
        print '##################################'
        print '##################################'
            



            
patternStorage()
currentPattern()
patternRecognition()