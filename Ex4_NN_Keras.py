#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:17:49 2021

@author: aysegulsezen
"""
import scipy.io
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def Ex4_NN_Keras():
    #1- Loading training data
    print('Training data loading..')
    mat = scipy.io.loadmat('ex4data1.mat')  # loading our training data.
    X= mat['X'] # 5000 handwritten digit image(rows), 400 piksel of every image (columns)
    y= mat['y'] # 5000 image's answer of which number it is. 0-9
    
    #y matrix has 1-10 numbers, we should roll them our nn output.
    num_labels = 10; 
    m = X.shape[0]
    yMatrix= np.zeros(shape=(m,num_labels))    #   yMatrix = zeros(m,num_labels); # in octave
    outP= np.eye(num_labels)  # outP= eye(num_labels);
    
    # making output binary. Exp: X(54) (54. handwritten image) is equal number 6. y(54) is 6. We gain yMatrix(54) is 0000010000. 6. is 1 others 0 
    for i in range(1,m):
        deger= y[i]-1
        yMatrix[i] = outP[deger]
 

    #2- Define Keras Model
    model = Sequential()
    model.add(Dense(25, batch_input_shape=(None,400), activation='sigmoid')) # input layer: 400 pixel, hidden layer: 25 #relu
    #model.add(Dense(50, activation='sigmoid'))    
    model.add(Dense(10, activation='softmax')) # output layer (0-9 digit) #activation='sigmoid'
    
    #3- Compile and fit data
    print('Keras model compiling and fitting..')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, yMatrix, epochs=10, batch_size=128)
    
    #4-Evaulate Keras
    _, accuracy = model.evaluate(X, yMatrix)
    print('Accuracy of our model: %.2f' % (accuracy*100))
    
    #Prediction
    ImageNo=1 # 1-5000 image. Write image no to want to guess
    predictions = model.predict(X[ImageNo].reshape(1,-1))
    #print('Prediction Image1:',predictions[0])
    #Neural Network has 10 output.'predictions' has 10 values. Max value index is digit we want to guess.
    print('Prediction of Image 1 (It should be 0):', np.mod(np.argmax( predictions[0], axis=0)+1,10 )) # 10 is 0. we use mod 10.
    
    
Ex4_NN_Keras()