#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:39:00 2020
Homework Andrew Ng Ex6 SVM
@author: aysegulsezen
"""

import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn import metrics

def gaussianKernel(x1, x2, sigma):
    x1 = x1[:]
    x2 = x2[:]
    sim = 0;
    sim=  np.exp(-sum( np.square( ( np.subtract( x1,x2) )))/  (2 * np.square(sigma ) )  )

    return sim

def dataset3Params(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    
      
    Cclass=np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]);
    Sigmaclass=np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]);
    error=np.zeros(shape=(Cclass.shape[0],Sigmaclass.shape[0]));
    
    for c in range(0 ,Cclass.shape[0]):
        for s in range(0,Sigmaclass.shape[0]):
            model = svm.SVC(kernel='rbf', C = Cclass[c],gamma=1/Sigmaclass[s])
            model.fit(X,y)
            y_pred=model.predict(Xval)
            error[c,s]=metrics.accuracy_score(yval, y_pred)

    
    maxIndex= np.where(error == np.amax(error)) 
    
    
    C = Cclass[maxIndex[0][0]];
    sigma = Sigmaclass[maxIndex[1][1]];

    return C,sigma

#https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[ 0], alpha=0.5,
               linestyles=[ '-'])
    
    # plot support vectors
    #if plot_support:
   #     ax.scatter(model.support_vectors_[:, 0],
   #                model.support_vectors_[:, 1],
   #                s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    


def ex6_SVM():  # main function
    ############-1-
    mat = scipy.io.loadmat('ex6data1.mat')  # loading our training data.
    X=mat['X']
    y=mat['y'][:,0]
    #xx=np.linspace(1, 4.2)
    #yy=np.linspace(4.5, 1.5)
    #plt.plot(xx, yy, '-k')
    #plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
    plt.scatter(X[:,0],X[:,1],c=y, s=50, cmap='autumn')
    plt.legend()
    plt.show()

    ##############-2-
    print('Training Linear SVM ...')
    C = 1.0;
    clf = svm.SVC(kernel='linear', C = C)
    clf.fit(X,y)
    #print('predict:',clf.predict([3,3.2,1]))
    support_vector_indices = clf.support_
    support_vectors_per_class = clf.n_support_
    support_vectors = clf.support_vectors_
    #plt.scatter(support_vectors[:,0], support_vectors[:,1], color='black')
    #plt.show()

    w = clf.coef_[0]
    a = -w[0] / w[1]

    xx = np.linspace(0,4)
    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.legend()
    plt.show()

    ################-3-
    print('Evaluating the Gaussian Kernel ...')

    x1 = [1, 2, 1]; 
    x2 = [0, 4, -1]; 
    sigma = 2;
    sim = gaussianKernel(x1, x2, sigma);

    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = :', sigma)
    print('(for sigma = 2, this value should be about 0.324652):',  sim);
    
    ##############-4-
    print('Loading and Visualizing Data ...')


    mat1 = scipy.io.loadmat('ex6data2.mat') 
    X1= mat1['X']
    y1= mat1['y'][:,0]
    print('X:',mat1['X'].shape, ' y:', mat1['y'].shape)
    plt.scatter(X1[:,0],X1[:,1],c=y1, s=50, cmap='autumn')
    plt.legend()
    plt.show()
    
    ##############-5-
    print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...');
    # SVM Parameters
    C = 1; sigma = 0.01; 

    clfG = svm.SVC(kernel='rbf', C = C,gamma=1/sigma)
    clfG.fit(X1,y1)
    
    #visualize
    plt.scatter(X1[:,0],X1[:,1],c=y1, s=50, cmap='autumn')
    plot_svc_decision_function(clfG)
    plt.show()
    
    ##############-6-
    print('Loading and Visualizing Data ...')

    mat2 = scipy.io.loadmat('ex6data3.mat') 
    X2= mat2['X']
    y2= mat2['y'][:,0]
    Xval=mat2['Xval']
    yval=mat2['yval']


    plt.scatter(X2[:,0],X2[:,1],c=y2, s=50, cmap='autumn')
    plt.legend()
    plt.show()
    
    ##############-7-
    [C, sigma] = dataset3Params(X2, y2, Xval, yval);

    #C=0.1
    clfG2 = svm.SVC(kernel='rbf', C = C,gamma=1/sigma)
    clfG2.fit(X2,y2)
    
    #visualize
    plt.scatter(X2[:,0],X2[:,1],c=y2, s=50, cmap='autumn')
    plot_svc_decision_function(clfG2)
    




    
    
ex6_SVM()

