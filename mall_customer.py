# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:10:29 2019

@author: jhunjhun
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

cmall = pd.read_csv('C:/Users/jhunjhun/Downloads/Mall_Customers.csv', index_col=0)

#x = cmall.iloc[:,1]
#y = cmall.iloc[:,2]
#plt.scatter(x,y)
#
#cmall = cmall.iloc[:,[0,1,2,3]]

X = cmall.iloc[:,[2,3]].values
 
f = plt.figure(1)
#kmeans = KMeans(n_clusters=3)
#kmeans.fit(iris)
#y_kmeans = kmeans.predict(iris)
#plt.scatter(x, y, c=y_kmeans, cmap='plasma')                       
#plt.xlabel('sepal length')
#plt.ylabel('sepal width')
#plt.title('Sepal Length V/S Sepal Width')
#centers = kmeans.cluster_centers_
#X = centers[:,0]
#Y = centers[:,1]
#plt.scatter(X, Y, c='black')
clf = KMeans(n_clusters=5, init='k-means++')
y_kmeans = clf.fit_predict(X)
fig = plt.figure(figsize=(10, 8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color='red', label='Cluster 1', edgecolors='black')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color='green', label='Cluster 2', edgecolors='black')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color='blue', label='Cluster 3', edgecolors='black')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color='yellow', label='Cluster 4', edgecolors='black')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color='orange', label='Cluster 5', edgecolors='black')
# cluster centres
plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1],marker='*', s=150, color='magenta', label='Centroid',edgecolors='black')
plt.legend()
plt.title('Clusters using KMeans')
plt.ylabel('Sepal Width')
plt.xlabel('Sepal Length')
f.show()

a = input('Enter first value: ')
b = input('Enter second value: ')

x = [[a,b]]
cluster = clf.fit_predict(x)
print(cluster)