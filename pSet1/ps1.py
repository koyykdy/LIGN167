# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:17:21 2018
LIGN167 Assignment 1
@author: Dongyoung Kim
PID: A13216053
"""
import numpy as np
import math
import matplotlib as mp
# Problem 1
def compute_slope_estimator(x,y):
    if x.size == y.size:
        if x.ndim == 1 and y.ndim == 1:
            #function here
            n = x.size
            for i in range(x.size):
                a=( ((x[i]*y[i])-(n*np.mean(x)*np.mean(y)))/((math.pow(x[i],2))-(n*math.pow(np.mean(x),2))) )
        else:
            print('Input Error: Dimension greater than 1')
            return None
    else:
        print('Input Error: Size Mismatch')
        return None
    #print("The estimated slope value is {:0.2f}".format(a))
    return a
# Problem 2
def compute_intercept_estimator(x,y):
    if x.size == y.size:
        if x.ndim == 1 and y.ndim == 1:
            #function here
            #n = x.size
            for i in range(x.size):
                b=( np.mean(y) - (compute_slope_estimator(x,y) * np.mean(x)) )
        else:
            print('Input Error: Dimension greater than 1')
            return None
    else:
        print('Input Error: Size Mismatch')
        return None
    #print("The estimated intercept is {:0.2f}".format(b))
    return b
# Problem 3
def train_model(x,training_set):
    slope = compute_slope_estimator(x,training_set)
    intercept = compute_intercept_estimator(x,training_set)
    #print("The estimated slope value is {:0.2f}".format(slope))
    #print("The estimated intercept is {:0.2f}".format(intercept))
    return (slope,intercept)
# Problem 4
def sample_linear_model(x,slope,intercept,sd):
    if x.ndim == 1:
        #function here
        #n = x.size
        yi = np.array([])
        for i in range(x.size):
            yi = np.append(yi, [slope*x[i] + intercept + np.random.normal(0,sd)])
    else:
        print('Input Error: Dimension greater than 1')
        return None
    return yi
# Problem 5
def sample_datasets(x,slope,intercept,sd,n):
    l = []
    for i in range(n):
        l.append(sample_linear_model(x,slope,intercept,sd))
    return l
# Problem 6
def compute_average_estimated_slope(x_vals,a=1,b=1,sd=1):
    n=1000 #this denotes the number of models being trained
    #x_vals = np.linspace(0,1,num=5)
    l = sample_datasets(x_vals,a,b,sd,n) # this takes n number of training sample datasets
    t=0
    for i in range(n):
        t = t + compute_slope_estimator(x_vals,l[i])
    avg = (1/n)*t
    return avg
# Problem 7: Free response answer here.
    # As the number of x_vals increased, so did compute time.
    # As the number of x_vals increased, the avg estimated slope decreased as it approached 3.
# Problem 8
# Problem 8: Free response answer here.
    # As the number of x_vals increased, so did compute time.
    # As the number of x_vals increased, the avg estimates slope error decreased as it approached 4.
def compute_estimated_slope_error(x_vals,a=1,b=1,sd=1):
    n=1000 #this denotes the number of models being trained
    #x_vals = np.linspace(0,1,num=5)
    l = sample_datasets(x_vals,a,b,sd,n)
    t=0
    for i in range(n):
        t = t + (math.pow(1-(compute_slope_estimator(x_vals,l[i])),2))
    error = (1/n)*t
    return error
# Problem 9: Include a pyplot graph as an additional file.
    # The histograms take longer to compute with larger x_vals.
    # The histograms appear to be slightly more normal with greater x_vals, but are generally normal at any
    # given x_vals size.
def slope_sampler(x_vals, a=1, b=1, sd=1):
    n=1000 #this denotes the number of models being trained
    #x_vals = np.linspace(0,1,num=5)
    l = sample_datasets(x_vals,a,b,sd,n)
    t=[]
    for i in range(n):
        t.append(compute_slope_estimator(x_vals,l[i]))
    mp.pyplot.hist(t)
    return t
# Problem 10
def calculate_prediction_error(y,y_hat):
    if y.size == y_hat.size:
        n = y.size
        t=0
        for i in range(n):
            t=t+math.pow((y[i]-y_hat[i]),2)
        perror = (1/n)*t
    else:
        print('Input Error: Size Mismatch')
        return None
    #print("The estimated slope value is {:0.2f}".format(a))
    return perror
# Problem 11
# Problem 11: Free response answer here.
    # As the number of x_vals increases, the training set error decreases; from ~8.5 at 5 to ~1.33 at 100.
    # The compute time becomes incredibly long as x_vals approaches 1000.
def average_training_set_error(x_vals,a=1,b=1,sd=1):
    n=1000 #this denotes the number of models being trained
    #x_vals = np.linspace(0,1,num=5)
    l = sample_datasets(x_vals,a,b,sd,n)
    p=0
    for i in range(n):
        p = p + calculate_prediction_error(l[i],(compute_slope_estimator(x_vals,l[i])*x_vals+compute_intercept_estimator(x_vals,l[i])))
    terror = (1/n)*p
    return terror
# Problem 12
# Problem 12: Free response answer here.
    # On average, compared to the training set errors, the errors are very slightly higher on the test set.
    # However, the trends of lower error with increasing x-vals and increasing compute times remain the same.
    # at x_vals=100: train_error= 1.33, test_error= 1.37; at x_vals=10: train_error= 1.83, test_error= 1.94
def average_test_set_error(x_vals,a=1,b=1,sd=1):
    n=1000
    l = sample_datasets(x_vals,a,b,sd,n)
    p=0
    for i in range(n):
        p = p + calculate_prediction_error(sample_linear_model(x_vals, a, b, sd),(compute_slope_estimator(x_vals,l[i])*x_vals+compute_intercept_estimator(x_vals,l[i])))
    aterror = (1/n)*p
    return aterror