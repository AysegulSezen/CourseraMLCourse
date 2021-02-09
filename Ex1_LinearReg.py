#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:28:48 2021
Homework of Andrew Ng. Converted from Octave to Python. Ex3 Logistic Reg vs Neural Network
@author: aysegulsezen
"""
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def  computeCost(X, y, theta):
    m = len(y); # number of training examples

    J = 0;

    # ====================== YOUR CODE HERE ======================
    predictions=np.dot(X,theta);
    rslt=pow(( np.array([x - y for x, y in zip(predictions, y)]) ),2);
    J=1 / (2*m) * np.sum(rslt);
    # =========================================================================

    return J 

def  gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y); # number of training examples
    J_history = np.zeros(shape=( num_iters, 1));

    for iter in range( 1,num_iters):
    # ====================== YOUR CODE HERE ======================
        predictions=np.dot(X , theta) 
        grad1=  np.array([x - y for x, y in zip(predictions, y)])
        grad2= ((1/m) * np.dot(grad1.transpose() , X).transpose())
        theta = theta - (grad2 * alpha);
   # ============================================================
        J_history[iter] = computeCost(X, y, theta); # degistir unutma theta degil theta 0 ve 1 olacak..
    
    return theta, J_history


def Ex1_LinearReg():
    ##############-1-
    print('Plotting Data ...')
    file = open("ex1data1.txt", "r")
    X=[]
    y=[]
    for line in file:
        str=line.split(',')
        X.append(float(str[0]))
        y.append(float(str[1].replace('\n','')))
        
    m = len(y); # number of training examples
    plt.scatter(X, y )#, c = y)

    
    ###############-2-
    oneColumn = np.ones(shape=(m,1))
    X = np.hstack((oneColumn,np.array(X)[:, None]))  #[np.ones(m, 1) X]; # X matrixinin başına 1 lerden oluşan bir kolon ekle
    theta = np.zeros(shape=(2, 1)); # initialize fitting parameters

    iterations = 1500;
    alpha = 0.01;

    print('Testing the cost function ...')
    J = computeCost(X, y, theta);
    print('With theta = [0 ; 0] Cost computed =', J);
    print('Expected cost value (approx) 32.07');
    
    J = computeCost(X, y, np.array([-1 , 2]).transpose());
    print('With theta = [-1 ; 2] Cost computed =', J);
    print('Expected cost value (approx) 54.24');
    
    print('Running Gradient Descent ...')
    theta = gradientDescent(X, y, theta, alpha, iterations)[0];

    # print theta to screen
    print('Theta found by gradient descent:', theta);
    print('Expected theta values (approx) -3.6303  1.1664');
    
    plt.plot(X[:,1], np.dot(X,theta))
    #legend('Training data', 'Linear regression')
    plt.show()
    
    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5] ,theta);
    print('For population = 35,000, we predict a profit of ',predict1*10000);
    predict2 = np.dot([1, 7] , theta);
    print('For population = 70,000, we predict a profit of ',predict2*10000);
    
    ###############-3-
    print('Visualizing J(theta_0, theta_1) ...')
    theta0_vals = np.linspace(-10, 10, 100);
    theta1_vals = np.linspace(-1, 4, 100);
    J_vals = np.zeros(shape=( len(theta0_vals), len(theta1_vals)));
    
    for i in range(1,len(theta0_vals)):
        for j in range(1,len(theta1_vals)):
            t = [theta0_vals[i], theta1_vals[j]];
            J_vals[i,j] = computeCost(X, y, t);
    J_vals = J_vals.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    #plt.show()
    plt.contour(theta0_vals, theta1_vals, J_vals ,levels=[-2, 3, 20])#,colors='k',
     #          levels=[ 0], alpha=0.5,
      #         linestyles=[ '-'])
    


 


    
Ex1_LinearReg()


