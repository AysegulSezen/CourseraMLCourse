#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:24:39 2021
Homework of Andrew Ng. Converted from Octave to Python.
Handwritten digit recognition by Logictic Regression
@author: aysegulsezen
"""
import scipy.io
import numpy as np
import skimage.io as io 
import math 
import scipy.optimize as op 

def  displayData(X, example_width):

    example_width = round( math.sqrt(X.shape[1]));

    m= X.shape[0]
    n =X.shape[1]
    example_height = int( (n / example_width));

    # Compute number of items to display
    display_rows = math.floor(math.sqrt(m));
    display_cols = math.ceil(m / display_rows);

    # Between images padding
    pad = 1;

    # Setup blank display
    w1=pad + display_rows * (example_height + pad)
    h1=int(pad + display_cols * (example_width + pad))
    display_array = - np.ones( shape=(w1 ,h1 ) );
    

    # Copy each example into a patch on the display array
    curr_ex = 0;
    for j in range( 1,display_rows+1):
    	for i in range(1,display_cols+1):
            max_val = max( abs( X[curr_ex, :] ) )
            row0=pad + (j - 1) * (example_height + pad) 
            row1=pad + (j - 1) * (example_height + pad) + example_height
            col0=pad + (i - 1) * (example_width + pad) 
            col1=pad + (i - 1) * (example_width + pad) + example_width
            display_array[row0:row1,col0:col1]=X[curr_ex, :].reshape( example_height, example_width,order='F') / max_val;
            
            curr_ex = curr_ex + 1;
		
    # Display Image
    h=io.imshow(display_array,cmap='gray')
    io.show()

    return h, display_array

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def lrCostFunction(theta, X, y, lambd):
            
    m = len(y); # number of training examples

    J = 0;
    grad = np.zeros( theta.shape);

    # ====================== YOUR CODE HERE ======================
        
    predictions=sigmoid( np.dot(X , theta) );
    lambdaSumC = lambd/(2*m) * sum(theta[1:] ** 2) ;
    J= np.sum(((-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))/m ) )+ lambdaSumC;
    
    lambdaSumG = lambd/m * theta;  
    grad= (1/m * np.dot( X.transpose() , ( np.array([x - y for x, y in zip(predictions, y)]) ))) 
    grad[0,0] = grad[0,0]-lambdaSumG[0];
    grad=np.concatenate(grad)

    return J, grad

def costFunction(theta, X, y, lambd):
            
    m = len(y); # number of training examples

    J = 0;
    predictions=sigmoid( np.dot(X , theta) );
    lambdaSumC = lambd/(2*m) * sum(theta[1:] ** 2) ;
    J= np.sum(((-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))/m ) )+ lambdaSumC;
    return J

def gradFunction(theta, X, y, lambd):
            
    m = len(y); # number of training examples

    J = 0;
    grad = np.zeros( theta.shape);

    
    predictions=sigmoid( np.dot(X , theta) );
    lambdaSumC = lambd/(2*m) * sum(theta[1:] ** 2) ;
    J= np.sum(((-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))/m ) )+ lambdaSumC;
    
    lambdaSumG = lambd/m * theta;  
    grad= (1/m * np.dot( X.transpose() , ( np.array([x - y for x, y in zip(predictions, y)]) ))) 
    #grad[0,0] = grad[0,0]-lambdaSumG[0];
    grad=np.concatenate(grad)
    return grad


def oneVsAll(X,y,num_labels,l):
    m=X.shape[0]  # 5000 olmalı
    n=X.shape[1]  # 400 olmalı
    all_theta= np.zeros( shape=(num_labels,n+1)) # 10 satır 401 sütunlu 0 1 matrix

    a = np.ones(shape=(m,1))
    X = np.hstack((a,X))  #[np.ones(m, 1) X]; # X matrixinin başına 1 lerden oluşan bir kolon ekle
    
    # ====================== YOUR CODE HERE ======================        
    initial_theta = np.zeros(shape=(n + 1, 1));
    num_iters=120;

    for c  in range(1,num_labels+1):
        # method TNC: %70 acc, 9 minute;  L-BFGS-B %69 acc 8 minute ; SLSQP %75 acc 1 minute 
        #fmin = op.minimize(fun=costFunction, x0=initial_theta, args=(X, np.where(y==c,1,0), l), method='TNC', jac=gradFunction) 
        #fmin = op.minimize(fun=costFunction, x0=initial_theta, args=(X, np.where(y==c,1,0), l), method='L-BFGS-B', jac=gradFunction)
        fmin = op.minimize(fun=costFunction, x0=initial_theta, args=(X, np.where(y==c,1,0), l), method='SLSQP', jac=gradFunction)
        if c==10 : # number 10 save all_theta 0 
            c=0
        all_theta[c,:] = fmin.x


    return all_theta #np.array(all_theta).reshape(num_labels,n+1), all_J

def  predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros( shape= (X.shape[0], 1));

    # Add ones to the X data matrix
    a = np.ones(shape=(m,1))
    X = np.hstack((a,X))  #[np.ones(m, 1) X]; # X matrixinin başına 1 lerden oluşan bir kolon ekle


    # ====================== YOUR CODE HERE ======================
    predictions = sigmoid( np.dot(X , all_theta.transpose())); # 5000 rows, 10 columns matrix

    for i in range(1,m):
        p[i]=np.argmax( predictions[i], axis=0) ;
    
    # 0 is used for number 10. index zero is equal 10.
    p=np.where(p==0,10,p)

    # =========================================================================
    return p 


def Ex3_LogReg():
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    num_labels = 10;          # 10 labels, from 1 to 10
    
    ################-1-
    print('Loading and Visualizing Data ...\n')


    mat = scipy.io.loadmat('ex3data1.mat') 
    X= mat['X'] # 5000 handwritten digit image(rows), 400 piksel of every image (columns)
    y= mat['y'] # 5000 image's answer of which number it is. 0-9
    
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices= np.random.permutation(m)
    sel = X[rand_indices[1:100], :];

    displayData(sel,5);
    
    ##################-2-
    print('Testing lrCostFunction() with regularization');

    theta_t = np.array([ [-2, -1, 1, 2]]).transpose();
    
    a = np.ones(shape=(5,1))
    X_t = np.hstack(( a , np.array(range(1,16)).reshape(5,3,order='F')/10 ));
    y_t = np.where(  np.array([[1,0,1,0,1]]).transpose() >= 0.5,1,0);
    lambda_t = 3;
    J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t);
    
    print('J:',lrCostFunction(theta_t, X_t, y_t, lambda_t)[0])
    print('grad:',lrCostFunction(theta_t, X_t, y_t, lambda_t)[1])

    print('Cost:', J);
    print('Expected cost: 2.534819');
    print('Gradients:', grad);
    print('Expected gradients: 0.146561 -0.548558 0.724722 1.398003');
    
    #################-3-
    print('Training One-vs-All Logistic Regression...')

    lambd = 0.1;
    all_theta = oneVsAll(X, y, num_labels, lambd);
    #print('all_theta:',all_theta.shape)
    
    ################-4-
    pred = predictOneVsAll(all_theta, X);
    print('pred:',pred[1:5], ' real:',mat['y'][1:5])
    print('pred:',pred[500:505], ' real:',mat['y'][500:505])
    print('pred:',pred[1000:1005], ' real:',mat['y'][1000:1005])
    print('pred:',pred[1500:1505], ' real:',mat['y'][1500:1505])
    print('pred:',pred[2000:2005], ' real:',mat['y'][2000:2005])
    print('pred:',pred[2500:2505], ' real:',mat['y'][2500:2505])
    print('pred:',pred[4500:4505], ' real:',mat['y'][4500:4505])

    print('Training Set Accuracy:', np.mean(np.double(np.where(pred == y,1,0))) * 100);
    
Ex3_LogReg()



