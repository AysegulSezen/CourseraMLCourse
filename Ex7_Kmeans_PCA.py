#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:02:28 2021
Homework of Andrew Ng. Converted from Octave to Python. PCA - Dimension reduction
@author: aysegulsezen
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io 
from PIL import Image
import pylab as pl
import math  

# from sklearn import preprocessing

def featureNormalize(X):
    mu = np.mean(X,axis=0)  #[np.mean(X[:,0]),np.mean(X[:,1])] #mean(X);
    X_norm = X- mu

    sigma = np.std(X_norm,axis=0, dtype=np.float32)#,dtype=np.float64);
    X_norm=X_norm/sigma

    return X_norm, mu, sigma

def  pca(X):
    m = X.shape[0]
    n=  X.shape[1]

    U = np.zeros(shape=n);
    S = np.zeros(shape=n);

    # ====================== YOUR CODE HERE ======================

    sigma= (1/m) * np.dot( X.transpose() , X );
    [U,S,v]=np.linalg.svd(sigma);
    S=np.diag(S)

    # =========================================================================
    return U, S


def  projectData(X, U, K):
    Z = np.zeros( shape=( X.shape[0], K));
    
    # ====================== YOUR CODE HERE ======================
    for i in range(0,X.shape[0]):
        for k in range(0,K):
            x = X[i, :].transpose();
            projection_k = sum(x.transpose() * U[:, k]);
            Z[i,k]=projection_k;
    # =============================================================
    
    return Z 


def recoverData(Z, U, K):
    X_rec = np.zeros( shape=(  Z.shape[0], U.shape[0]));

    # ====================== YOUR CODE HERE ======================

    for i in range(0,Z.shape[0]) :
        for j in range(0,U.shape[0]):
            v = Z[i, :].transpose();
            recovered_j = v.transpose() * U[j, 0:K].transpose();
            X_rec[i,j]=recovered_j;

    # =============================================================

    return X_rec 

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



def Ex7_Kmeans_PCA():
    #################-1-
    print('Visualizing example dataset for PCA.');

    mat = scipy.io.loadmat('ex7data1.mat') 
    X= mat['X']

    #  Visualize the example dataset
    plt.scatter(X[:, 0], X[:, 1])
    
    
    #################-2-
    print('Running PCA on example dataset.');

    [X_norm, mu, sigma] = featureNormalize(X);

    #  Run PCA
    [U, S] = pca(X_norm);

    
    y0=mu + 1.5 * S[0,0] * U[:,0].transpose() 
    yy0=[mu[1],y0[1]]
    xx0=[mu[0],y0[0]]
    h0 = plt.plot(xx0,yy0 , '-k')

    y1=mu + 1.5 * S[1,1] * U[:,1].transpose()
    yy1=[mu[1],y1[1]]
    xx1=[mu[0],y1[0]]

    h1 = plt.plot(xx1,yy1, '-k')

    plt.legend()
    plt.show()


    print('Top eigenvector:');
    print(' U(:,1) =', U[0,0], U[1,0]);
    print('(you should expect to see -0.707107 -0.707107)');
    
    #################-3-
    print('Dimension reduction on example dataset.');

    plt.scatter(X_norm[:,0],X_norm[:,1]);

    #  Project the data onto K = 1 dimension
    K = 1;
    Z = projectData(X_norm, U, K);
    print('Projection of the first example:', Z[0]);
    print('(this value should be about 1.481274)');

    X_rec  = recoverData(Z, U, K);
    print('Approximation of the first example:', X_rec[0, 0], X_rec[0, 1]);
    print('(this value should be about  -1.047419 -1.047419)');

    #  Draw lines connecting the projected points to the original points
    plt.plot(X_rec[:, 0], X_rec[:, 1]);
    for i in range(0,X_norm.shape[0]):
        xx=[X_norm[i,:][0],X_rec[i,:][0]]
        yy=[X_norm[i,:][1],X_rec[i,:][1]]
        plt.plot(xx,yy, '--k', 'LineWidth', 1);
    
    plt.legend()
    plt.show()
    
    ################-4-
    print('Loading face dataset.');

    matF = scipy.io.loadmat('ex7faces.mat') 
    Xf= matF['X']
    #  Display the first 100 faces in the dataset
    displayData(Xf[0:100, :],5);
            
    ################-5-
    print('\nRunning PCA on face dataset.(this might take a minute or two ...)');
    [X_norm, mu, sigma] = featureNormalize(Xf);

    #  Run PCA
    [U, S] = pca(X_norm);
    
    print('U:',U.shape)
    print('S:',S.shape)

    #  Visualize the top 36 eigenvectors found
    displayData(U[:, 1:36].transpose(),5);
    #io.imshow(U[0].reshape(32,32,order='F'),cmap='gray')
    
    ################-6-
    print('Dimension reduction for face dataset.');

    K = 100;
    Z = projectData(X_norm, U, K);

    print('The projected data Z has a size of: ', Z.shape);
    
    ################-7-
    K = 100;

    # Display normalized data
    displayData(X_norm[1:100,:],5);
    #title('Original faces');

    # Display reconstructed data from only k eigenfaces
    displayData(X_rec[1:100,:],5);
    #title('Recovered faces');

    
Ex7_Kmeans_PCA()
    