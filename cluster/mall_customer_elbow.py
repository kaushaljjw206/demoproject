
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.spatial.distance import cdist

cmall = pd.read_csv('C:/Users/jhunjhun/Downloads/Mall_Customers.csv', index_col=0)
cmall = cmall.iloc[:,[0,1,2,3]]

X = cmall.iloc[:,[2,3]]

sse = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    sse.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
#for k in K:
#    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(iris)
#    iris["clusters"] = kmeans.labels_
#    #print(data["clusters"])
#    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(K,sse,'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("Sum of Squared Error (SEE)")

print(K ,'\n' , sse)
kn = KneeLocator(list(K), sse, S=1.0, curve='convex', direction='decreasing')
#print(kn.knee)
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.title('Elbow method for mall customer dataset')
plt.show()
