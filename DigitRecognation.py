#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:36:21 2020
@author: aysegulsezen
"""
import scipy.io
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.optimize as op
from PIL import Image
from skimage.io import imread
#import cv2

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def costFunction(theta,X,y,l):
    m=len(X)
    
    
    predictions=sigmoid(np.dot(X,theta)) #sigmoid(X * theta );
    cost= ((-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))/m ) #+ lambdaSumC;
    regCost= sum(cost) + l/(2*m) * sum(theta[1:]**2)



    #lG=(l/m * theta[1:])
    grad= (1/m * np.dot( X.transpose() , (predictions-y))) #+ lambdaSumG;
    #grad[0]=grad[0]-lG[0]
    #regGrad= grad + lG

    #return regCost[0] , regGrad
    return regCost[0],grad
    
def gradientDescent(X,y,theta,alpha,num_iters,l):
    m=len(y)
    J_history= np.zeros(shape=(num_iters,1))
    
    for i in range(num_iters):
        cost, grad = costFunction(theta,X,y,l)
        theta = theta - (alpha * grad)
        J_history[i]=cost
        #theta= theta - ((1/m) * ((X*theta)-y).transpose()*X).transpose() * alpha
        #J_history[i]=computeCost(X,y,theta,0)
        
        
    return theta,J_history
    
    
def oneVsAll(X,y,num_labels,l):
    m=X.shape[0]  # 5000 olmalı
    n=X.shape[1]  # 400 olmalı
    all_theta= np.zeros( shape=(num_labels,n+1)) # 10 satır 401 sütunlu 0 1 matrix
    #print('all_theta:',all_theta.shape[1])
    a = np.ones(shape=(m,1))
    X = np.hstack((a,X))  #[np.ones(m, 1) X]; # X matrixinin başına 1 lerden oluşan bir kolon ekle
    #print('X:',X[500,0])
    
        
    
    initial_theta = np.zeros(shape=(n + 1, 1));
    alpha = 0.01;
    num_iters=120;

    #all_theta = [] #np.zeros(shape=(num_labels,1))
    #all_J=[] 
    #all_J=np.zeros(shape=(num_labels,1))
    #all_theta = np.zeros()
    
    for c in range(num_labels):
        # if y==c : y1=0 else y1=1
        theta,J_history=gradientDescent(X,np.where(y==c,1,0),initial_theta,alpha,num_iters,l) # theta 401 sütunlu 1 satır matrix
        all_theta[c]=theta.ravel()
        #all_J[c]=J_history
        #all_theta.extend(theta)
        #all_J.extend(J_history)

    return all_theta #np.array(all_theta).reshape(num_labels,n+1), all_J

    #Result = op.fmin_cg(f= costFunction, 
    #                             x0 = initial_theta, 
    #                             #args = (X, np.where(y==5,1,0),l))#,
    #                             args=(X,y,l) ,disp=1)   
    #                             #method = 'TNC',
    #                             #jac = 'Gradient')
    #optimal_theta = Result.x;
    
    #op.fmin_cg(f=costFunction,x0=initial_theta,args=(X,y,l),
        
        #output = op.fmin_tnc(func = lrCostFunction, x0 = theta.flatten(), fprime = lrGradientDescent, \
        #                args = (X, y.flatten(), lmbda))
        #theta = output[0]

def predictOneVsAll(all_theta, X):
    m=X.shape[0]
    #num_labels=all_theta.shape[0]
    #oneNumber=np.array([X]).shape[0]
    
    
    p=np.zeros(shape=(m,1))
    a = np.ones(shape=(m,1))
    print('a s0',a.shape[0]) 
    print('a s1',a.shape[1]) 
    print('X s0',X.shape[0]) 
    print( 'X s1',X.shape[1])
    print('a:',len(a), ' X:',len(X))
    X = np.hstack((a,X))  #[np.ones(m, 1) X]; # add 1 numbers column in X (X matrixinin başına 1 lerden oluşan bir kolon ekle)


    predictions = sigmoid( np.dot(X , all_theta.transpose())); # 5000 rows, 10 columns matrix
    #print('prd1:',predictions[1])
    #print('rslt:',np.argmax( predictions[1], axis=0) )
    
    p = np.zeros(shape=(m, 1));    
    #aa=predictions.index(max(predictions))
    #print('aa',aa)
    for i in range(m):
     #   [val, index] = max(rslt(i,:));
     #  p(i) = index;
        p[i]=np.argmax( predictions[i], axis=0) ;
    #print('p:',p[1])
        
    #predictions = X @ all_theta.T
    #return np.argmax(predictions,axis=1)+1
    #a=np.argmax(predictions)
    #print('a:',a)
    #aa=np.amax(predictions,axis=1)
    #result = np.where(predictions == np.amax(predictions))
    #print('aa:',result)
    #result=np.argmax(predictions, axis=0)
    #print('')
    
    return p #np.argmax(predictions) # 5000 satırdaki her bir image için 10 kolondaki max degerin indexini dön
        


def DigitRecognation():

    mat = scipy.io.loadmat('ex3data1.mat')
#sorted(mat.keys())
#print(mat['X'][1][1])
#print(mat['y'][1])
#print(len(mat['X']))

    m=len(mat['X'])
#length_key = len(mat.keys()) 
#length_dict = {key: len(value) for key, value in mat.items()}
#print(length_dict)
#length_key = length_dict['key']  # length of the list stored at `'key'` ...
#img = cv2.imread('images/CloudyGoldenGate.jpg')


#plt.imshow(mat['X'][1].reshape(20,20))
#plt.show()

    #print(sigmoid(0))
    
    theta_t = np.array([[-2], [-1], [1], [2]])
    X_t= np.array([[1,0.1,0.6,1.1],[1,0.2,0.7,1.2],[1,0.3,0.8,1.3],[1,0.4,0.9,1.4],[1,0.5,1,1.5]])
    y_t = np.array([[1],[0],[1],[0],[1]] );
    lambda_t = 3;
    J, grad = costFunction(theta_t, X_t, y_t, lambda_t);

    #print('\nCost: %f\n', J);
    #print('Expected cost: 2.534819\n');
    #print('Gradients:\n');
    #print(' %f \n', grad);
    #print('Expected gradients:\n');
    #print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');
    
    #m , n = X.shape[0], X.shape[1]
    #X= np.append(np.ones((m,1)),X,axis=1)
    #y=y.reshape(m,1)
    #initial_theta = np.zeros((n+1,1))
    #cost, grad= costFunction(initial_theta,X,y)
    #print("Cost of initial theta is",cost)
    #print("Gradient at initial theta (zeros):",grad)

    #print(costFunction(0,))
    num_labels = 10
    l=0.1
    #print(mat['X'].shape[0])
    bestThetas=oneVsAll(mat['X'],mat['y'],num_labels,l)  # 10 rows, 401 columns matrix ; rows is 0-9 numbers; columns is pixels best theta of this number
    print('best theta',bestThetas.shape[0])
    
    size= 28,28
    img2 = Image.open('foto/IMG_6815.jpg').convert('LA')
    img2.thumbnail(size, Image.ANTIALIAS)
    
    plt.imshow(img2)
    plt.show()

    x2 = np.array(img2)#.reshape(20,20)) # 640x480x4 array
    print( 'x2:',  x2)

    #x21=np.zeros(shape=(400,1))
    #t=0
    #for i in range(20):
    #    for j in range(20):
    #        x21[t]=x2[i][j]
    #        t=t+1
    #print('x21:',x21)
    
    #im = imread("foto/IMG_6815.jpg")
    #im.reshape([1, 28, 28, 1])
    #print('im:',im.shape)
    #indices = np.dstack(np.indices(x2.shape[:2]))
    #data = np.concatenate((x2, indices), axis=-1)
    #print('data:',data)
    #print('im:',im)
    
    #img2.reshape([-1, 28, 28, 1])
    #xI=x2.shape[0]
    #yI=x2.shape[1]*x2.shape[2]
 
    #x2.resize((xI,yI)) # a 2D array
    #print('x2:',x2.reshape(20,20))
    
    #print('matX1', np.size(  mat['X'][1] ,1 ) )
    
    x3=np.array([ mat['X'][1]])
    #print('x3:',x3)
    pred = predictOneVsAll(bestThetas,x3)  #mat['X']);
    print('pred:',pred)
    #print('tahmin:',pred[2], ' gercek:',mat['y'][2])
    #print("Training Set Accuracy:",sum(pred[:,np.newaxis]==mat['y'])[0]/5000*100,"%")

    #print('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
    compare=np.zeros(shape=(m,2))

    correct=0
    #for i in range(m):
    #    if pred[i]==mat['y'][i]:
    #        correct = correct +1
    #    compare[i,0]=pred[i]
    #    compare[i,1]=mat['y'][i]
            
    #print('correct:',correct)
    #print('compare:',compare)

    
DigitRecognation()


