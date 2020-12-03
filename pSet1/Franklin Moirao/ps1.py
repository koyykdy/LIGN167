# My name: Franklin Moirao
# Partners' names: None :(
# My contributions: All of the contributions
# Partners' contributions: I have no partner :(

import numpy as np

# Problem 1
def compute_slope_estimator(x,y):
    x = np.array(n, )
    y = np.array(n, )

    #to get xbar and y bar
    for i in x:
        x_bar_temp = x_bar_temp + x[i]
        y_bar_temp = y_bar_temp + y[i]

    xbar = (1/n) * x_bar_temp
    ybar = (1/n) * y_bar_temp

    #a(x,y) equation
    for j in x and y:
        a_top_temp = a_top_temp + (x[i] * y[i])
        a_bot_temp = a_bot_temp + (x[i]^2)

    a_top = a_top_temp - (n * xbar * ybar)
    a_bot = a_bot_temp - (n * xbar^2)

    a = a_top / a_bot

return a


# Problem 2
def compute_intercept_estimator(x,y):
    x = np.array(n, )
    y = np.array(n, )

    #to get xbar and y bar
    for i in x:
        x_bar_temp = x_bar_temp + x[i]
        y_bar_temp = y_bar_temp + y[i]

    xbar = (1/n) * x_bar_temp
    ybar = (1/n) * y_bar_temp


    #b(x,y) equation
    b = ybar - (compute_slope_estimator * xbar)

return b


# Problem 3
def train_model(x,training_set):
    slope = compute_slope_estimator(x,y)
    intercept = compute_intercept_estimator(x,y)

return slope and intercept



# Problem 4
def sample_linear_model(x,slope,intercept,sd):
    y = np.array(n ,)

    for i in x:
        y[i] = (compute_slope_estimator * x[i]) + compute_intercept_estimator + np.random.normal

return y[i]

# Problem 5
def sample_datasets(x,slope,intercept,sd,n):
    dataset = np.array(n, )

    for j in dataset:
        dataset[i] = sample_linear_model

return dataset[i]

# Problem 6
def compute_average_estimated_slope(x_vals,a=1,b=1,sd=1):


# Problem 7: Free response answer here.

# Problem 8
# Problem 8: Free response answer here.
def compute_estimated_slope_error(x_vals,a=1,b=1,sd=1):

# Problem 9: Include a pyplot graph as an additional file.

# Problem 10
def calculate_prediction_error(y,y_hat):

# Problem 11
# Problem 11: Free response answer here.
def average_training_set_error(x_vals,a=1,b=1,sd=1):

# Problem 12
# Problem 12: Free response answer here.
def average_test_set_error(x_vals,a=1,b=1,sd=1):