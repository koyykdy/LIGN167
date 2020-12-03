# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:28:51 2018
LIGN167 Assignment 1
@author: Dongyoung Kim
PID: A13216053
"""
import numpy as np
import torch


######################################## BEGIN STARTER CODE ########################################

def relu(x):
	if x<0:
		return 0
	else:
		return x

def loss(y_predicted, y_observed):
	return (y_predicted - y_observed)**2

def mlp(x,W0,W1,W2):
	
	r0_0 = x*W0[0]
	r0_1 = x*W0[1]
	r0_2 = x*W0[2]
	r0 = np.array([r0_0,r0_1,r0_2])

	h0_0 = relu(r0_0)
	h0_1 = relu(r0_1)
	h0_2 = relu(r0_2)
	h0 = np.array([h0_0,h0_1,h0_2])

	r1_0 = h0_0*W1[0,0] + h0_1*W1[0,1]+ h0_2*W1[0,2]
	r1_1 = h0_0*W1[1,0] + h0_1*W1[1,1]+ h0_2*W1[1,2]
	r1_2 = h0_0*W1[2,0] + h0_1*W1[2,1]+ h0_2*W1[2,2]
	r1 = np.array([r1_0,r1_1,r1_2])

	h1_0 = relu(r1_0)
	h1_1 = relu(r1_1)
	h1_2 = relu(r1_2)
	h1 = np.array([h1_0,h1_1,h1_2])

	y_predicted = h1_0*W2[0] + h1_1*W2[1]+ h1_2*W2[2]

	variable_dict = {}
	variable_dict['x'] = x
	variable_dict['r0'] = r0
	variable_dict['h0'] = h0
	variable_dict['r1'] = r1
	variable_dict['h1'] = h1
	variable_dict['y_predicted'] = y_predicted

	return variable_dict

x = 10
W0 = np.array([1,2,3])
W1 = np.array([[3,4,5],[-5,4,3],[3,4,1]])
W2 = np.array([1,3,-3])

#print(mlp(x,W0,W1,W2))

###########  PyTorch code   ###########
def torch_mlp(x,W0,W1,W2):
	m = torch.nn.ReLU()
	h0 = m(torch.mul(W0,x))

	h1 = m(torch.matmul(W1,h0))

	y_predicted = torch.dot(W2,h1)

	return y_predicted

def torch_loss(y_predicted,y_observed):
	return torch.pow(y_predicted-y_observed,2)


x_torch = torch.tensor(x,dtype=torch.float)
W0_torch = torch.tensor(W0,dtype=torch.float,requires_grad=True)
W1_torch = torch.tensor(W1,dtype=torch.float,requires_grad=True)
W2_torch = torch.tensor(W2,dtype=torch.float,requires_grad=True)
output = torch_mlp(x_torch,W0_torch,W1_torch,W2_torch)

########### END PyTorch code  ###########




######################################## END STARTER CODE ########################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES


#PROBLEM 1
def d_loss_d_ypredicted(variable_dict,y_observed):
    y_pred = variable_dict['y_predicted']
    derivLoss = 2*(y_pred - y_observed)
    return derivLoss

#PROBLEM 2
def d_loss_d_W2(variable_dict,y_observed):
    derivLoss = d_loss_d_ypredicted(variable_dict,y_observed)
    derivW2_0 = derivLoss*variable_dict['h1'][0]
    derivW2_1 = derivLoss*variable_dict['h1'][1]
    derivW2_2 = derivLoss*variable_dict['h1'][2]
    derivW2 = [derivW2_0,derivW2_1,derivW2_2]
    return derivW2

#PROBLEM 3
def d_loss_d_h1(variable_dict,W2,y_observed):
    derivLoss = d_loss_d_ypredicted(variable_dict,y_observed)
    derivh1_0 = derivLoss*W2[0]
    derivh1_1 = derivLoss*W2[1]
    derivh1_2 = derivLoss*W2[2]
    derivh1  = [derivh1_0,derivh1_1,derivh1_2]
    return derivh1

#PROBLEM 4
def relu_derivative(x):
    if x <= 0:
        dReLU = 0
    else:
        dReLU = 1
    return dReLU

#PROBLEM 5
def d_loss_d_r1(variable_dict,W2,y_observed):
    r1 = variable_dict['r1'][:]
    derivh1 = d_loss_d_h1(variable_dict,W2,y_observed)
    derivr1_0 = derivh1[0]*relu_derivative(r1[0])
    derivr1_1 = derivh1[1]*relu_derivative(r1[1])
    derivr1_2 = derivh1[2]*relu_derivative(r1[2])
    derivr1 = [derivr1_0,derivr1_1,derivr1_2]
    return derivr1

#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
	derivW1 = np.outer(d_loss_d_r1(variable_dict,W2,y_observed),variable_dict['h0'])
	return derivW1

#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
	derivh0 = np.dot(d_loss_d_r1(variable_dict,W2,y_observed),W1)
	return derivh0

#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
	r0 = variable_dict['r0']
	derivh0 = d_loss_d_h0(variable_dict,W1,W2,y_observed)
	derivr0_0 = derivh0[0]*relu_derivative(r0[0])
	derivr0_1 = derivh0[1]*relu_derivative(r0[1])
	derivr0_2 = derivh0[2]*relu_derivative(r0[2])
	derivr0 = [derivr0_0,derivr0_1,derivr0_2]
	return derivr0

#PROBLEM 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
	derivr0 = d_loss_d_r0(variable_dict,W1,W2,y_observed)
	x = variable_dict['x']
	derivW0 = [derivr0[0]*x,derivr0[1]*x,derivr0[2]*x]
	return derivW0