''' Clustering example of iris dataset

'''




import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans

iris = pd.read_csv('C:/Users/jhunjhun/Downloads/iris.csv')
iris = iris.iloc[:,[0,1,2,3]]
#x = iris.iloc[:,0]
#y = iris.iloc[:,1]
X = iris.iloc[:,[0,1]].values
#print(iris)
#plt.scatter(x,y) 


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
clf = KMeans(n_clusters=3, init='k-means++')
y_kmeans = clf.fit_predict(X)
fig = plt.figure(figsize=(10, 8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color='red', label='Cluster 1', edgecolors='black')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color='green', label='Cluster 2', edgecolors='black')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color='blue', label='Cluster 3', edgecolors='black')
# cluster centres
plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], color='magenta', label='Centroid',edgecolors='black')
plt.legend()
plt.title('Clusters using KMeans')
plt.ylabel('Sepal Width')
plt.xlabel('Sepal Length')
f.show()


#g = plt.figure(2)
#x = iris.iloc[:,0]
#y = iris.iloc[:,2]
##print(iris)
##plt.scatter(x,y) 
#kmeans = KMeans(n_clusters=3)
#kmeans.fit(iris)
#y_kmeans = kmeans.predict(iris)
#plt.scatter(x, y, c=y_kmeans, cmap='plasma')
#plt.xlabel('sepal length')
#plt.ylabel('petal length')
#plt.title('Sepal Length V/S Petal Length')
#centers = kmeans.cluster_centers_
#X = centers[:,0]
#Y = centers[:,2]
#plt.scatter(X, Y, c='black')
#g.show()
#
#
#h = plt.figure(3)
#x = iris.iloc[:,0]
#y = iris.iloc[:,3]
##print(iris)
##plt.scatter(x,y) 
#kmeans = KMeans(n_clusters=3)
#kmeans.fit(iris)
#y_kmeans = kmeans.predict(iris)
#plt.scatter(x, y, c=y_kmeans, cmap='plasma')
#plt.xlabel('sepal length')
#plt.ylabel('petal width')
#plt.title('Sepal Length V/S Petal Width')
#centers = kmeans.cluster_centers_
#X = centers[:,0]
#Y = centers[:,3]
#plt.scatter(X, Y, c='black')
#h.show()
#
#
#i = plt.figure(4)
#x = iris.iloc[:,1]
#y = iris.iloc[:,2]
##print(iris)
##plt.scatter(x,y) 
#kmeans = KMeans(n_clusters=3)
#kmeans.fit(iris)
#y_kmeans = kmeans.predict(iris)
#plt.scatter(x, y, c=y_kmeans, cmap='plasma')
#plt.xlabel('sepal width')
#plt.ylabel('petal length')
#plt.title('Sepal Width V/S Petal Length')
#centers = kmeans.cluster_centers_
#X = centers[:,1]
#Y = centers[:,2]
#plt.scatter(X, Y, c='black')
#i.show()
#
#
#j = plt.figure(5)
#x = iris.iloc[:,1]
#y = iris.iloc[:,3]
##print(iris)
##plt.scatter(x,y) 
#kmeans = KMeans(n_clusters=3)
#kmeans.fit(iris)
#y_kmeans = kmeans.predict(iris)
#plt.scatter(x, y, c=y_kmeans, cmap='plasma')
#plt.xlabel('sepal width')
#plt.ylabel('petal width')
#plt.title('Sepal Width V/S Petal Width')
#centers = kmeans.cluster_centers_
#X = centers[:,1]
#Y = centers[:,3]
#plt.scatter(X, Y, c='black')
#j.show()
#
#
#k = plt.figure(6)
#x = iris.iloc[:,2]
#y = iris.iloc[:,3]
##print(iris)
##plt.scatter(x,y) 
#kmeans = KMeans(n_clusters=3)
#kmeans.fit(iris)
#y_kmeans = kmeans.predict(iris)
#plt.scatter(x, y, c=y_kmeans, cmap='plasma')
#plt.xlabel('petal length')
#plt.ylabel('petal width')
#plt.title('Petal Length V/S Petal Width')
#centers = kmeans.cluster_centers_
#X = centers[:,2]
#Y = centers[:,3]
#plt.scatter(X, Y, c='black')
#k.show()
