#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:17:34 2020
Homework of Andrew Ng. Converted from Octave to Python. Kmeans algorithm.
Image compression
@author: aysegulsezen
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io 
from sklearn.cluster import KMeans


def findClosestCentroids(X, centroids):
    K = centroids.shape[0]

    idx = np.zeros(shape= X.shape);
    #====================== YOUR CODE HERE ======================

    for i in range(0,X.shape[0]):
        length= np.zeros( shape=( 1, centroids.shape[0]));   
        for k in range(0,centroids.shape[0]):
            x1=X[i,:]-centroids[k,:];
            length[0,k]= sum ( np.square(x1)) ;
        col= np.argmin(length[0,:]);
        idx[i]=col;
    # =============================================================

    return idx

def computeCentroids(X, idx, K):
    m = X.shape[0]
    n = X.shape[1]

    centroids = np.zeros( shape=( K, n))

    # ====================== YOUR CODE HERE ======================
    for i in range(0,K):
        value=[]
        for j in range(0,idx.shape[0]):
            if (idx[j,0]==i):
                value.append(X[j,:]);
        centroids[i,:]=  sum(value)/len(value)
                   
    # =============================================================
    return centroids

def  runkMeans(X, initial_centroids, max_iters, plot_progress):
    # Initialize values
    m = X.shape[0]
    n = X.shape[1]
    K = initial_centroids.shape[0]
    centroids = initial_centroids;
    previous_centroids = centroids;
    idx = np.zeros(shape=(m, 1));

    # Run K-Means
    for i in range(1,max_iters):
        # Output progress
        print('K-Means iteration', i, max_iters);
    
        #For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids);
    
        # Optionally, plot progress here
        if plot_progress:
            plt.scatter(X[:,0],X[:,1], c=idx[:,0],s=50, cmap='autumn')
            plt.scatter(centroids[:,0],centroids[:,1], marker='X', color='black')

            previous_centroids = centroids;
            print('Press enter to continue.');
                
        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K);
    
    if plot_progress:
        plt.legend()
        plt.show()

    return centroids, idx

def  kMeansInitCentroids(X, K):
    centroids = np.zeros( shape=( K, X.shape[1]));

    # ====================== YOUR CODE HERE ======================
    randidx = np.random.permutation(X.shape[0])  # randperm(X.shape[0])
    centroids = X[randidx[0:K], :]
    # =============================================================

    return centroids 


def ex7_Kmeans():  # main function
    #################-1-
    print('Finding closest centroids.');
    mat = scipy.io.loadmat('ex7data2.mat') 
    X= mat['X']
    
    K = 3; # 3 Centroids
    initial_centroids =np.array( [ [3, 3], [6, 2], [8 ,5]]);

    # Find the closest centroids for the examples using the initial_centroids
    idx = findClosestCentroids(X, initial_centroids);
    print('Closest centroids for the first 3 examples:', idx[0:3]);
    print('(the closest centroids should be 1, 3, 2 respectively)');
    
    #################-2-
    print('Computing centroids means.');

    #  Compute means based on the closest centroids found in the previous part.
    centroids = computeCentroids(X, idx, K);

    print('Centroids computed after initial finding of closest centroids:', centroids);
    print('(the centroids should be');
    print('   [ 2.428301 3.157924 ]\n');
    print('   [ 5.813503 2.633656 ]\n');
    print('   [ 7.119387 3.616684 ]\n\n');
    
    ################-3-
    print('Running K-Means clustering on example dataset.');

    K = 3;
    max_iters = 10;
    initial_centroids =np.array( [ [3, 3], [6, 2], [8 ,5]]);
    [centroids, idx] = runkMeans(X, initial_centroids, max_iters, True);
    
    print('K-Means Done.');
    
    ###############-4-
    A = io.imread('bird_small.png')
    A=A/255
    img_size = A.shape
    X = np.reshape(A, (img_size[0] * img_size[1], 3));
    K = 16; 
    max_iters = 10;
    initial_centroids = kMeansInitCentroids(X, K);
    [centroids, idx] = runkMeans(X, initial_centroids, max_iters,False);
    io.imshow(A)
    
    ##############-5-
    
    print('Applying K-Means to compress an image.');
    #print('centroids:',centroids)
    idx1 = findClosestCentroids(X, centroids);
    #print('idx1:',idx1[0:10])
    #print('idx1 sh:',idx1.shape)

    X_recovered=[]#np.zeros(shape=(idx1.shape[0]))  # 16384 e 3 olması lazım
    for i in range(0,idx1.shape[0]): # 16384
        X_recovered.append(centroids[int(idx1[i][0])])
   
    X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1],3));

    # Display the original image 
    io.imshow(A) #imagesc(A);
    #io.title("Original")
    io.show()

    # Display compressed image side by side
    #io.title('Compressed, with',K,'colors.') 
    io.imshow(X_recovered)#imagesc(X_recovered)   
    io.show()
    
    ###############-6- out of homework. Doing same job by python class
    print('Image compression by K-means (sklearn) python class..')
    kmeans = KMeans(
        init="random",
        n_clusters=K, 
        n_init=10,
        max_iter=max_iters,
        random_state=42
    )
    
    kmeans.fit(X) # X : pixels of image.
    
    print('kmeans.inertia_:',kmeans.inertia_)
    print('kmeans.cluster_centers_:',kmeans.cluster_centers_) # this is same with centroids variable on up.
    print('kmeans.n_iter_:',kmeans.n_iter_)
    print('kmeans.labels_[:5]',kmeans.labels_[:10])
    print('kmeans.labels_:',kmeans.labels_.shape)
    
    y_kmeans = kmeans.predict(X)   # y_kmeans is same with idx1 variable on up. 
    
    print('y_kmeans:',y_kmeans.shape)
    print('y_kmeans 0-10:',y_kmeans[0:10])
    
    X_recovered2=[]
    for i in range(0,y_kmeans.shape[0]): # 16384
        X_recovered2.append(kmeans.cluster_centers_[int(y_kmeans[i])])
   
    X_recovered2 = np.reshape(X_recovered2, (img_size[0], img_size[1],3));
    io.imshow(X_recovered2)  
    io.show()
        
    
ex7_Kmeans()
