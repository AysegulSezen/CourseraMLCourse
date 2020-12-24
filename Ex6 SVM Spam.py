#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:49:26 2020
Homework of Andrew Ng. Converted from Octave to Python. Spam classification with SVM.
@author: aysegulsezen
"""
import re
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import svm
from sklearn import metrics
import scipy.io

def  getVocabList():
    #for  word in fid.split():   
    df=pd.read_table('vocab.txt',names=['id','word'])
    return df #vocabList

def  processEmail(email_contents):

    vocabList = getVocabList();

    word_indices = []

    # ========================== Preprocess Email ===========================
    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    email_contents =  email_contents.replace('<[^<>]+>',' ')    #regexprep(email_contents, '<[^<>]+>', ' ');
    #email_contents=re.compile(r'\W+', re.UNICODE).split(email_contents)

    # Handle Numbers
    email_contents = re.sub(r'\d+', 'number', email_contents) 
    
    # Handle URLS
    email_contents=re.sub('(http|https)://[^\s]*','httpaddr',email_contents)

    # Handle Email Addresses
    email_contents= re.sub('\S+@\S+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub(r"\$",'dollar',  email_contents)

    # ========================== Tokenize Email ===========================
    print('\n==== Processed Email ====\n\n');

    # Process file
    l = 0;

    for str in email_contents.split():   
        #%Remove any non alphanumeric characters
        str = re.sub(r'\W+', '', str)

        # Skip the word if it is too short
        if len(str) < 1:
            pass;

    # ====================== YOUR CODE HERE ======================
        for i in range(1, len(vocabList)):
            if (str == vocabList.loc[i]["word"]):
                word_indices.append( vocabList.loc[i]["id"])
    

    # =============================================================
    print('=========================');

    return word_indices 

def emailFeatures(word_indices):
    n = 1899;
    x = np.zeros( shape=(n, 1));
    #% ====================== YOUR CODE HERE ======================
    x[word_indices] = 1;
    #% =========================================================================
    
    return x


def Ex6_SVM_Spam():
    #############-1-
    file_contents = open("emailSample1.txt", "r")
    print('Preprocessing sample email (emailSample1.txt)')
    word_indices  = processEmail(file_contents.read());
    print(word_indices)
    
    #############-2-
    features=emailFeatures(word_indices)
    print('Length of feature vector:', len(features));
    print('Number of non-zero entries:', sum(features > 0));
    
    print('feature shape:',features.transpose().shape)
    
    #############-3-
    mat = scipy.io.loadmat('spamTrain.mat') 
    X= mat['X']  # 4000 email sample, 1899 feature (word of word_indicate); data is 1-0; word exist or not
    y= mat['y'][:,0] # 4000 answer of email sample is spam or not.

    print('Training Linear SVM (Spam Classification)')
    print('(this may take 1 to 2 minutes) ...')

    C = 0.1;
    model= svm.SVC(kernel='linear', C = C)
    model.fit(X,y)
    y_pred=model.predict(X)
    acc=metrics.accuracy_score(y, y_pred)


    print('Training Accuracy:', acc);
    
    #############-4-
    mat = scipy.io.loadmat('spamTest.mat') 
    Xtest= mat['Xtest']
    ytest= mat['ytest'][:,0]

    print('Evaluating the trained Linear SVM on a test set ...')

    y_pred=model.predict(Xtest)
    acc=metrics.accuracy_score(ytest, y_pred)


    print('Test Accuracy:', acc);
    
    ##############-5-
    # Making decision for which svm parameter is equal weight of words ( in octave homework model.w )
    #print('model.class_weight_:',model.class_weight_)
    #print('model.classes_',model.classes_)
    print('model.coef_',model.coef_)
    #print('model.intercept_',model.intercept_.shape)
    #print('model.support_',model.support_.shape)
    print('model.support_vectors_',model.support_vectors_.shape)
    #print('model.n_support_',model.n_support_)
    #print('model.shape_fit_',model.shape_fit_)
    
    vocabList = getVocabList();
    weight=model.coef_[0]  # is equal model.w ; weights of 1899 words.
    vocabList['weight']=weight  # adding weight value of every word after svm
    print('Top predictors of spam:',vocabList.sort_values(by=['weight'],ascending=False)  )

        
    #############-6-
    filename = 'spamSample1.txt';

    x=features
    p = model.predict(x.transpose());

    print('Processed Spam Classification: ', filename, p);
    print('(1 indicates spam, 0 indicates not spam)');
    
    #############-7- (Optional) spam eamil from my mailbox
    filename1 = 'spamSample2.txt';

    file_contents1 = open("spamSample2.txt", "r")
    word_indices1  = processEmail(file_contents1.read());
    features1=emailFeatures(word_indices1)
    x1=features1
    p1 = model.predict(x1.transpose());

    print('Processed Spam Classification: ', filename1, p1);
    print('(1 indicates spam, 0 indicates not spam)');
Ex6_SVM_Spam()


