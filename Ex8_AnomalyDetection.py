#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:16:15 2021
Homework of Andrew Ng. Converted from Octave to Python. Anormaly Detection
@author: aysegulsezen
"""
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math  

def estimateGaussian(X):
    m = X.shape[0]
    n = X.shape[0]

    mu = np.zeros( shape=(n, 1));
    sigma2 = np.zeros( shape=( n, 1));

    # ====================== YOUR CODE HERE ======================
    mu = 1/m * sum(X);
    sigma2= 1/m*sum( np.square( (X-mu)) );
    # =============================================================

    return mu, sigma2

def multivariateGaussian(X, mu, Sigma2):
    k = len(mu);

    if (Sigma2.ndim == 1):
        Sigma2 = np.diag(Sigma2);

    X =  X- mu[:].transpose()
    p1 = pow((2 * math.pi) , (- k / 2)) * pow( np.linalg.det(Sigma2) , (-0.5)) 
    p2=np.exp(-0.5 *  np.multiply( np.dot( X , np.linalg.pinv(Sigma2)), X).sum( axis=1));
    p=p1*p2
    return p

def visualizeFit(X, mu, sigma2):

    x = np.linspace(0, 35, 71)
    y = np.linspace(0, 35, 71)
    X1, X2 = np.meshgrid(x, y) 
    x1x2=np.array([np.concatenate(X2),np.concatenate(X1)]).transpose()
    Z = multivariateGaussian(x1x2,mu,sigma2);

    Z = Z.reshape(X1.shape,order='F');
    plt.scatter(X[:, 0], X[:, 1],marker='x');
    plt.contour(X1, X2, Z,levels=(pow(10,-20), pow(10,-17), pow(10,-14), pow(10,-11), pow(10,-8), pow(10,-5), pow(10,-2)))
    #plt.legend()
    #plt.show()
    
def selectThreshold(yval, pval):
    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;

    stepsize = (max(pval) - min(pval)) / 1000;

    #for epsilon in range( min(pval),max(pval),stepsize):
    epsilon=min(pval)
    while epsilon<max(pval):
    
    # ====================== YOUR CODE HERE ======================
        
        predictions = np.where(pval < epsilon,1,0) #(pval < epsilon);

        truePos=sum(sum(np.where( np.where( yval==1,1,0).transpose() & np.where(predictions==1,1,0) ,1,0) )) 
        trueNeg=sum(sum(np.where( np.where( yval==0,1,0).transpose() & np.where(predictions==0,1,0) ,1,0) ) )
        falsePos=sum(sum(np.where( np.where( yval==0,1,0).transpose() & np.where(predictions==1,1,0) ,1,0) )) 
        falseNeg=sum(sum(np.where( np.where( yval==1,1,0).transpose() & np.where(predictions==0,1,0) ,1,0) )) 

        prec=truePos/(truePos+falsePos);
        rec =truePos/(truePos+falseNeg);

        F1= 2 * prec * rec / (prec + rec ) ;

    # =============================================================
        if ( F1 > bestF1 ) :
            bestF1 = F1;
            bestEpsilon = epsilon;
            
        epsilon = epsilon+stepsize

    return bestEpsilon, bestF1 




def Ex8_AnomalyDetection():  # main function
    ###############-1-
    print('Visualizing example dataset for outlier detection.');
    mat = scipy.io.loadmat('ex8data1.mat') 
    X= mat['X']

    #  Visualize the example dataset
    plt.scatter(X[:, 0], X[:, 1])
    plt.legend()
    plt.show()
    
    ###############-2-
    print('Visualizing Gaussian fit.\n\n');

    mu, sigma2 = estimateGaussian(X);
    p = multivariateGaussian(X, mu, sigma2);
    visualizeFit(X,  mu, sigma2);
    
    ###############-3-
    Xval= mat['Xval']
    yval= mat['yval']

    pval = multivariateGaussian(Xval, mu, sigma2);
    epsilon, F1 = selectThreshold(yval, pval);
    
    print('Best epsilon found using cross-validation:', epsilon);
    print('Best F1 on Cross Validation Set:', F1);
    print('   (you should see a value epsilon of about 8.99e-05)');
    print('   (you should see a Best F1 value of  0.875000)');
    
    #  Find the outliers in the training set and plot the
    outliers =np.where( p < epsilon)[0]

    #  Draw a red circle around those outliers
    plt.scatter(X[outliers, 0], X[outliers, 1], s=50, alpha=0.5,edgecolors='r')#color='r')#  , linewidths=) #, 'ro', 'LineWidth', 2, 'MarkerSize', 10);
    plt.legend()
    plt.show()
    
    ################-4-
    mat2 = scipy.io.loadmat('ex8data2.mat') 
    X2= mat2['X']
    X2val=mat2['Xval']
    y2val=mat2['yval']

    mu2, sigma2_2 = estimateGaussian(X2);

    #  Training set 
    p2 = multivariateGaussian(X2, mu2, sigma2_2);

    #  Cross-validation set
    p2val = multivariateGaussian(X2val, mu2, sigma2_2);

    #  Find the best threshold
    epsilon2, F1_2 = selectThreshold(y2val, p2val);
    

    print('Best epsilon found using cross-validation: ', epsilon2);
    print('Best F1 on Cross Validation Set:', F1_2);
    print('   (you should see a value epsilon of about 1.38e-18)');
    print('   (you should see a Best F1 value of 0.615385)');
    print('number Outliers found:', sum(np.where( p2 < epsilon2,1,0)));

    

Ex8_AnomalyDetection()