#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:54:35 2020

@author: aysegulsezen
"""
import scipy.io
import numpy as np
import math
import scipy.optimize as op

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoidGradient(z):
    g=np.zeros(z.shape[0])
    #print('z size:',z.shape[0])
    g= sigmoid(z) * (1- sigmoid(z))
    return g

def randInitializeWeights(L_in, L_out):
    W = np.zeros( shape=( L_out, 1 + L_in))
    epsilon_init=0.12;
    W= np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init
    #print('W:',W)
    #W= np.random.RandomState(L_out,1+L_in)*2*epsilon_init-epsilon_init;    
    return W



def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,l):
    Theta1=0
    Theta2=0
    
    Theta1=nn_params[0:(hidden_layer_size*  (input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    Theta2=nn_params[(hidden_layer_size*  (input_layer_size+1)): ].reshape(num_labels,(hidden_layer_size+1))

    m = X.shape[0]
    
    J = 0;
    Theta1_grad = np.zeros( shape=( Theta1.shape) );
    Theta2_grad = np.zeros( shape=( Theta2.shape) );
    
    ###### Part 1 Finding Cost     
    oneColumn = np.ones(shape=(m,1))
    X = np.hstack((oneColumn,X))  #[np.ones(m, 1) X]; # add 1 numbers column in X (X matrixinin başına 1 lerden oluşan bir kolon ekle)
    
    z2= np.dot(X,Theta1.transpose())
    a2=sigmoid(z2)
    a2=np.hstack((oneColumn,a2))
    z3=np.dot(a2,Theta2.transpose())
    hx=sigmoid(z3)
    
           
    #y matrix has 1-10 numbers, we should roll them our nn output.
    yMatrix= np.zeros(shape=(m,num_labels))    #   yMatrix = zeros(m,num_labels); # in octave
    outP= np.eye(num_labels)  # outP= eye(num_labels);
    
    # making output binary. Exp: X(54) (54. handwritten image) is equal number 6. y(54) is 6. We gain yMatrix(54) is 0000010000. 6. is 1 others 0 
    for i in range(1,m):
        #deger=0
        #if (y[i]==10):
        #    deger=0
        #else:
        deger= y[i]-1
        yMatrix[i] = outP[deger]
    
    #print('yMatrix 2301:',-yMatrix[2301])

    lamdaSum= l/(2*m) *  (sum(sum( np.square( Theta1[1:]))) + sum(sum( np.square( Theta2[1:])))  )
    J=  (1/m) * sum(sum(( -yMatrix * np.log(hx)-(1-yMatrix ) * np.log(1-hx))) ) + lamdaSum;
    print('cost J:',J)   # to watch how cost descrise after iterations

    ##### Part 2 Finding Grad
    delta3=hx-yMatrix
    delta2= np.dot(delta3 , Theta2[:,1:]) * sigmoidGradient(z2)
    Delta2= np.dot(delta3.transpose() , a2 )
    Delta1= np.dot(delta2.transpose() , X )
    Theta1_grad= 1/m * Delta1
    Theta2_grad= 1/m * Delta2
    

    ##### Part 3 regulization

    lambdaSumG1 = l/m * np.hstack( ( (np.zeros(shape=(Theta1.shape[0],1))) , Theta1[:,1:] ));
    lambdaSumG2 = l/m * np.hstack( ( (np.zeros(shape=(Theta2.shape[0],1))) , Theta2[:,1:] ));

    Theta1_grad = Theta1_grad + lambdaSumG1;
    Theta2_grad = Theta2_grad + lambdaSumG2;
    
    # Unroll gradients
    grad = np.concatenate([np.concatenate(Theta1_grad),np.concatenate(Theta2_grad)])  #np.array( [initial_Theta1 , initial_Theta2]);

    return J,grad



def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros( shape=(fan_out, 1 + fan_in));
    W = np.reshape( np.sin(range(1,((fan_out*(1+fan_in))+1)) ), W.shape , order='F') / 10; # numel(W) = fan_out * (1+fan_in) count of object
    return W

def computeNumericalGradient(J, theta , input_layer_size,hidden_layer_size,num_labels,X,y,lamda):
    Theta1= theta[0]  #auto shaped,in octave Theta1 = reshape( nn_params(1:hi ......
    Theta2= theta[1]  #auto shaped,in octave Theta2 = reshape(nn_params((1 + (hidden_layer_size * (inp ....
    countOfArrayMember= theta.shape[0] 

    numgrad = np.zeros( np.array([countOfArrayMember,1])); # theta 38 row 1, column 3 input 5 hidden 3 output.a0,h0 (4*5)+(6*3)
    perturb = np.zeros( np.array([countOfArrayMember,1]));
    e = math.e - 4;

    #print('theta:',theta)
    
    for p in range( 1 , countOfArrayMember):
    # Set perturbation vector
        perturb[p] = e
        thetaMinus =np.array([x - y for x, y in zip(theta, perturb)]) #np.subtract(theta , perturb)[0]
        thetaPlus = np.array([x + y for x, y in zip(theta, perturb)])  #np.dot( thetaC , perturb)
        loss1 = J(thetaMinus,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)[0];
        loss2 = J(thetaPlus,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)[0];
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;
    return numgrad


def checkNNGradients(lamda):
    if not lamda:
        lamda=0
    
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;
    
    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1);
    #y  = 1 + np.mod( range(1,m), num_labels).transpose();
    y = np.array([2,3,1,2,3])
    
    # Unroll parameters
    nn_params= np.concatenate([np.concatenate(Theta1),np.concatenate(Theta2)])    

    cost,grad= nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)

    #grad1= np.concatenate([grad[0],grad[1]])
    #cost1= nnCostFunction(grad1,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)[0]
    #print('cost1:',cost1)

    numgrad = computeNumericalGradient(nnCostFunction, nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
    
    print('numgrad:',numgrad[19], ' grad:', grad[0][19]) # to compare grad and numgrad, difference should be very small
    print('diff 0:',numgrad[1]-grad[0][1])

def predict(Theta1,Theta2,X):
    num_labels=Theta2.shape[0]
    m = X.shape[0] # X is an image/images that has 400 pixel. Our Neural Network has 400 input for every pixel.
    
    oneColumn = np.ones(shape=(m,1))
    X = np.hstack((oneColumn,X))  #[np.ones(m, 1) X]; # add 1 numbers column in X (X matrixinin başına 1 lerden oluşan bir kolon ekle)
    
    z2= np.dot(X,Theta1.transpose())
    a2=sigmoid(z2)
    a2=np.hstack((oneColumn,a2))
    z3=np.dot(a2,Theta2.transpose())
    hx=sigmoid(z3)    # our prediction; In our neural network; we have 10 output for every digit.
                      
    # hx is vector has 10 member. Index of max numbers of 10 members is the value we want to guess.
    #Example: hx = [[1.52834147e-05 2.89936597e-04 1.79826944e-04 9.64594577e-01 1.73522042e-03 5.90510137e-05 8.48302152e-04 5.08429881e-04
    #  3.94767885e-03 2.47331007e-05]]  4 member is max. Our guess is 4. first index is for number 1, last one for 0, etc. 
    #We need to find index of max number.
    p = np.zeros(shape=(m, 1));    
    for i in range(m):
        p[i]=np.argmax( hx[i], axis=0) ;

    return p+1 # index start 0  , sum 1 
    


def ex4_NN():  # main function
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    hidden_layer_size = 25;   # 25 hidden units
    num_labels = 10;          # 10 labels, from 1 to 10; we estimate digit on image. 
    
    mat = scipy.io.loadmat('ex4data1.mat')  # loading our training data.

    m = mat['X'].shape [0] # X 5000 rows  (images) 400 columns ( each images pixels (20x20))
    
    matW = scipy.io.loadmat('ex4weights.mat') # loading initial parameters (layers weight) 
    nn_params = np.array( [matW['Theta1'], matW['Theta2']]) #  [(25,401),(10,26)]
    # nn_params is our Thetas. Our Neural Network has 400 input unit (a1),25 hidden unit(a2), 10 output unit (a3)  

    #Unroll parameters, we make one vector all Thetas. It has data 25x401 + 10x26
    nn_params= np.concatenate([np.concatenate(nn_params[0]),np.concatenate(nn_params[1])]) 
    lmbda = 0
    
    J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels, mat['X'],mat['y'],lmbda)
    
    g=sigmoidGradient(np.array([0])) #np.array( [-1 ,-0.5, 0, 0.5, 1]))
    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    # Unroll parameters, make vector from arrays
    initial_nn_params =np.concatenate([np.concatenate(initial_Theta1),np.concatenate(initial_Theta2)])  #np.array( [initial_Theta1 , initial_Theta2]);
    #checkNNGradients(0)  # you can open this code. It is for making sure your grad is correct. comparing grad and numgrad.
    
    
    debug_J  = nnCostFunction(nn_params, input_layer_size, 
                          hidden_layer_size, num_labels, mat['X'],mat['y'],3);
    print('debugJ:',debug_J[0])
    
    #aaa= (initial_nn_params, input_layer_size, hidden_layer_size, num_labels, mat['X'],mat['y'], lmbda)
    def decoratedCost(Thetas):  # thanks stackexchange.
        return nnCostFunction(Thetas, input_layer_size, hidden_layer_size, num_labels, mat['X'],mat['y'], lmbda)[0]
    
    def decoratedGrad(Thetas):
        return nnCostFunction(Thetas, input_layer_size, hidden_layer_size, num_labels, mat['X'],mat['y'], lmbda)[1]

    # Finding the best Thetas for our NN. 
    #result=op.fmin_bfgs(decoratedCost,initial_nn_params,fprime=decoratedGrad,maxiter=10)
    result=op.fmin_cg(decoratedCost,initial_nn_params,fprime=decoratedGrad,maxiter=50)
    #print('result:',result.shape)
    
    bestTheta1=result[0:(hidden_layer_size*  (input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    bestTheta2=result[(hidden_layer_size*  (input_layer_size+1)): ].reshape(num_labels,(hidden_layer_size+1))
    imageNo=2010  # 1-5000 image. Write image no to want to guess
    x3=np.array([ mat['X'][imageNo]])   
    p=predict(bestTheta1,bestTheta2, x3) # mat['X'])
    print('p:',p)
    print('y:',mat['y'][imageNo])

    
ex4_NN()    

