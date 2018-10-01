#Optimizers
#What's an optimizer?: 1)Find minimum values of functions 2)Build parameterized models based on data 3)Refine allocations to stock in portfolios

#How to use optimizer: 1)Provide a function to minimize (ex.: f(x)=x^2+0.5) 2)Provide an initial guess 3) Call the optimizer

#Problematic functions: Several local minima, areas with 0 slope, discontinuity
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.optimize as spo

def f(x):
    #Given a scale X, return some value (a real number).
    y = (x - 1.5)**2+0.5
    print "x = {}, y = {}".format(x,y) #for tracing
    return y

#Minimizer
def test_run():
    Xguess = 2.0 #Initial guess
    
    #Parameters: the function, initial guess, solving algorithm, verbosity of results
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
    print 'Minima found at:'
    print "x = {}, y = {}".format(min_result.x,min_result.fun) #for tracing

    #Plot function values and mark minima
    Xplot = np.linspace(0.5,2.5,21)
    Yplot = f(Xplot)
    plt.plot(Xplot, Yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title("Minima of an objective function")
    plt.show()

test_run()


def error(line, data): # error function- compute error between given line model and observed data.
    '''Compute error between given line model and observed data.
    Parameters:
    line: tuple/list/array (C0,C1) where C0 is the slope and C1 the y-intercept
    data: 2D array where each row is a point (x,y)
    
    returns error as a single real value
    '''

    #Metric: Sum of squared Y-axis differences
    err = np.sum((data[:,1] - (line[0]*data[:,0] + line[1])) **2)
    return err