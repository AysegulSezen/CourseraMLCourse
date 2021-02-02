#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:14:07 2021
Homework of Andrew Ng. Converted from Octave to Python. Ex3 Logistic Reg vs Neural Network
@author: aysegulsezen
"""
import scipy.io
import math 
import numpy as np
import skimage.io as io 

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
    
    #display_array[0:32,0:32]=X[0, :].reshape( example_height, example_width,order='F') #/ max_val;
    #display_array[0:32,32:64]=X[1, :].reshape( example_height, example_width,order='F')


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


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros( shape=( X.shape[0], 1));

    # ====================== YOUR CODE HERE ======================
    # Add ones to the X data matrix
    a = np.ones(shape=(m,1))
    X = np.hstack((a,X))  #[np.ones(m, 1) X]; # X matrixinin başına 1 lerden oluşan bir kolon ekle

    HiddenLayerResult=sigmoid(np.dot(X,Theta1.transpose()));
    oneColumn=np.ones(shape=(m,1))
    HiddenLayerResult= np.hstack((oneColumn,HiddenLayerResult)) # [np.ones(m,1) HiddenLayerResult];
    OutputLayerResult=sigmoid(np.dot(HiddenLayerResult,Theta2.transpose()));

    for i in range( 1,m+1):
        #[val, index] = max(OutputLayerResult[i,:]);
        #p[i] = index;
        p[i-1]=np.argmax( OutputLayerResult[i-1,:], axis=0) +1;

    # =========================================================================

    return p 

def Ex3_LogReg_NN():
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    hidden_layer_size = 25;   # 25 hidden units
    num_labels = 10;          # 10 labels, from 1 to 10
    
    ###################-1-
    print('Loading and Visualizing Data ...')
    mat = scipy.io.loadmat('ex3data1.mat')
    X = mat['X']
    y = mat['y']
    m = X.shape[0]
    
    rand_indices= np.random.permutation(m)
    sel = X[rand_indices[1:100], :];

    displayData(sel,5);
    
    ##################-2-
    print('Loading Saved Neural Network Parameters ...')
    mat1 = scipy.io.loadmat('ex3weights.mat')
    Theta1 = mat1['Theta1']
    Theta2 = mat1['Theta2']
    
    ##################-3-
    pred = predict(Theta1, Theta2, X);
    #print('pred:',pred[1:5], ' real:',mat['y'][1:5])
    #print('pred:',pred[500:505], ' real:',mat['y'][500:505])
    #print('pred:',pred[1000:1005], ' real:',mat['y'][1000:1005])
    #print('pred:',pred[1500:1505], ' real:',mat['y'][1500:1505])
    #print('pred:',pred[2000:2005], ' real:',mat['y'][2000:2005])
    #print('pred:',pred[2500:2505], ' real:',mat['y'][2500:2505])
    #print('pred:',pred[4500:4505], ' real:',mat['y'][4500:4505])


    print('Training Set Accuracy: ', np.mean(np.double( np.where( pred == y,1,0))) * 100);
    
    #  Randomly permute examples
    #rp = np.random.permutation(m);
    #for i in range( 1,m):
        # Display 
        #print('Displaying Example Image');
        #imagePixel=X[rp[0:i], :]
        #displayData( imagePixel[ np.newaxis,:],5);

    imageNo= 1234  # give an image number between 1-5000
    x1=X[imageNo, :][ np.newaxis,:]
    displayData(x1,5);

    pred = predict(Theta1, Theta2, x1);
    print('Neural Network Prediction:',pred,' (digit)', np.mod(pred, 10));
    


    
Ex3_LogReg_NN()


