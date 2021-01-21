#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:56:54 2021
Homework of Andrew Ng. Converted from Octave to Python. Movie
@author: aysegulsezen
"""
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from pandas import DataFrame
import scipy.optimize as op

def  cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambd):
    
    X = params[0:num_movies*num_features].reshape( num_movies, num_features);
    Theta = params[(num_movies*num_features):].reshape( num_users, num_features)#,order='F');
            
    J = 0;
    X_grad = np.zeros(shape=X.shape);
    Theta_grad = np.zeros( shape=Theta.shape);

    # ====================== YOUR CODE HERE ======================
    JRegT= lambd/2 *  sum ( sum ( pow(Theta , 2 )));
    JRegX= lambd/2 *  sum ( sum ( pow(X , 2 )));

    J= (1/2 * sum(sum ( R * (pow((np.dot( X,Theta.transpose()) - Y ) , 2) ) )) )  + JRegT + JRegX;
   
    X_grad     = (np.dot((R * ( np.dot(X,Theta.transpose()) - Y)) ,Theta)) + (lambd * X); 
    Theta_grad = ( np.dot( (R * (np.dot(X,Theta.transpose()) - Y)).transpose() , X )) + (lambd*Theta)   ; #    sum(R .* (X*Theta' - Y)) * X;%zeros(size(Theta));

    # =============================================================

    grad = np.array( [X_grad[:], Theta_grad[:]]);
    grad= np.concatenate([np.concatenate(grad[0]),np.concatenate(grad[1])]) 

    return J, grad

def computeNumericalGradient(J, theta,Y, R, num_users, num_movies, num_features, lambd):
    numgrad = np.zeros(theta.shape);
    perturb = np.zeros(theta.shape);
    e = 0.0001 #math.e-4;
    countOfArrayMember= theta.shape[0] 
    
    for p in range( 0,countOfArrayMember):
        # Set perturbation vector
        perturb[p] = e;
        thetaMinus =np.array([x - y for x, y in zip(theta, perturb)]) #np.subtract(theta , perturb)[0]
        thetaPlus = np.array([x + y for x, y in zip(theta, perturb)])  #np.dot( thetaC , perturb)

        loss1 = J(thetaMinus,Y, R, num_users, num_movies, num_features, lambd)[0];
        loss2 = J(thetaPlus,Y, R, num_users, num_movies, num_features, lambd)[0];
        
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;

    return numgrad 


def checkCostFunction(lambd):
    if not lambd:
        lambd = 0;


    ## Create small problem
    X_t = np.random.rand(4, 3);
    
    
    Theta_t = np.random.rand(5, 3);
    
    Y = np.dot(X_t , Theta_t.transpose());
    aa= np.random.rand(Y.shape[0],Y.shape[1])
    replace_mask = aa<0.5
    Y[replace_mask] = np.where(aa<0.5,0,1)[replace_mask]

    
    R = np.zeros(shape=Y.shape);
    R = np.where(Y==0,0,1)
    
    X = np.random.rand(X_t.shape[0],X_t.shape[1]);
    
    
    Theta = np.random.rand(Theta_t.shape[0],Theta_t.shape[1]);
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]
    nn_paramsJ= np.concatenate([np.concatenate(X),np.concatenate(Theta)]) 
    nn_paramsN= np.concatenate([np.concatenate(X.transpose()),np.concatenate(Theta.transpose())]) 
    
    cost,grad= cofiCostFunc(nn_paramsJ,Y, R, num_users, num_movies, num_features, lambd)
    
    numgrad = computeNumericalGradient(cofiCostFunc,nn_paramsJ, Y, R, num_users, num_movies, num_features, lambd)
    
    
    for i in range(0,len(numgrad)):
        print('num:',numgrad[i],'--grad:',grad[i])
    
    #disp([numgrad grad]);
    print('The above two columns you get should be very similar.' 
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)');

    diff=np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then ' 
         'the relative difference will be small (less than 1e-9). ' 
         'Relative Difference: ', diff);
    
    
def  normalizeRatings(Y, R):
    m = Y.shape[0] 
    n = Y.shape[1]
    Ymean = np.zeros( shape=(m, 1))
    Ynorm = np.zeros(shape=Y.shape);
    res_list = [i for i, value in enumerate(R[2,:]) if value == 1]
    for i in range( 1,m):
        idx=[j for j, value in enumerate(R[i,:]) if value == 1]#R[i,:].tolist().index(1)
        Ymean[i] = np.mean(Y[i, idx]);
        Ynorm[i, idx] = Y[i, idx] - Ymean[i];


    return Ynorm, Ymean

    

def Ex8_cofi():
    ################-1-
    print('Loading movie ratings dataset.');

    mat = scipy.io.loadmat('ex8_movies.mat') 
    Y= mat['Y'] # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on  943 users
    R= mat['R'] # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

    #  From the matrix, we can compute statistics like average rating.
    print('Average rating for movie 1 (Toy Story): / 5', np.mean(Y[0, R[0, :]]));
    
    plt.imshow(Y)
    plt.show()
    #ylabel('Movies');
    #xlabel('Users');
    
    ###############-2-
    mat = scipy.io.loadmat('ex8_movieParams.mat') 
    X= mat['X']
    Theta= mat['Theta']

    #  Reduce the data set size so that this runs faster
    num_users = 4; num_movies = 5; num_features = 3;
    X = X[0:num_movies, 0:num_features];
    Theta = Theta[0:num_users, 0:num_features];
    Y = Y[0:num_movies, 0:num_users];
    R = R[0:num_movies, 0:num_users];
    
    #  Evaluate cost function
    params=np.array([X[:] , Theta[:]])
    params=np.concatenate([np.concatenate(params[0]),np.concatenate(params[1])]) 
    J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0);
           
    print('Cost at loaded parameters: (this value should be about 22.22)', J[0]);
    
    ################-3-
    print('Checking Gradients (without regularization) ... ');

    #  Check gradients by running checkNNGradients
    checkCostFunction(0);
    
    ################-4-
    J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5);
           
    print('Cost at loaded parameters (lambda = 1.5):(this value should be about 31.34)', J[0]);
    
    ################-5-
    print('Checking Gradients (with regularization) ... ');

    #  Check gradients by running checkNNGradients
    checkCostFunction(1.5);
    
    ################-6-
    
    f=open('movie_ids.txt','r',encoding = "ISO-8859-1")
        
    movieList={ }
    for line in f.readlines():
        lineInfo=[]
        lineInfo=line.split(" ", 1)
        movieList[lineInfo[0]]=lineInfo[1]

    my_ratings = np.zeros( shape=(1682, 1));

    my_ratings[1] = 4;
    my_ratings[98] = 2;
    my_ratings[7] = 3;
    my_ratings[12]= 5;
    my_ratings[54] = 4;
    my_ratings[64]= 5;
    my_ratings[66]= 3;
    my_ratings[69] = 5;
    my_ratings[183] = 4;
    my_ratings[226] = 5;
    my_ratings[355]= 5;
    
    print('New user ratings:');
    for i in range( 1,len(my_ratings)):
    #for idx,movieName in movieList.items():
        if (my_ratings[i] > 0 ):
            print('Rated for ', my_ratings[i], movieList[str(i)])
            
    ###############-7-
    print('Training collaborative filtering...');

    mat = scipy.io.loadmat('ex8_movies.mat') 
    Y= mat['Y'] # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on  943 users
    R= mat['R'] # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i


    #  Add our own ratings to the data matrix
    Y = np.hstack((my_ratings, Y));
    R = np.hstack((np.where(my_ratings==0,0,1) , R));  
    Ynorm, Ymean = normalizeRatings(Y, R);
    
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10;
    
    X = np.random.randn(num_movies, num_features);
    Theta = np.random.randn(num_users, num_features);

    initial_parameters=np.array([X[:] , Theta[:]])
    initial_parameters=np.concatenate([np.concatenate(initial_parameters[0]),np.concatenate(initial_parameters[1])]) 


    #options = optimset('GradObj', 'on', 'MaxIter', 100);

    lambd = 10;
    def decoratedCost(Thetas):  # thanks stackexchange.
        return cofiCostFunc(Thetas,Y, R, num_users, num_movies, num_features, lambd)[0]
    
    def decoratedGrad(Thetas):
        return cofiCostFunc(Thetas,Y, R, num_users, num_movies, num_features, lambd)[1]

    #theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
     #                           num_features, lambda)), ...
      #          initial_parameters, options);
    result=op.fmin_cg(decoratedCost,initial_parameters,fprime=decoratedGrad,maxiter=50)
    
    X = result[0:num_movies*num_features].reshape(num_movies, num_features);
    Theta = result[(num_movies*num_features):].reshape(num_users, num_features);

    print('Recommender system learning completed.');
    
    ###################-8-
    p = np.dot(X , Theta.transpose())
    my_predictions = np.array([x + y for x, y in zip(p[:,0], Ymean)])  #p[:,0] + Ymean;

    #movieList = loadMovieList();
 
    r= sorted(my_predictions,reverse = True)   # sort(my_predictions, 'descend');
    ix=sorted(range(len(my_predictions)), key=lambda k: my_predictions[k],reverse = True)
    print('Top recommendations for you:');
    for i in range(1,10):
        j = ix[i];
        print('Predicting rating 1f for movie:', my_predictions[j],movieList[str(j)]);


    print('Original ratings provided:');
    for i in range( 1,len(my_ratings)):
        if my_ratings[i] > 0: 
            print('Rated for', my_ratings[i], movieList[str(i)]);

    
Ex8_cofi()


