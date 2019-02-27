import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import numpy as np
from scipy.spatial.distance import cdist

iris = pd.read_csv('C:/Users/jhunjhun/Downloads/iris.csv')
iris = iris.iloc[:,[2,3]]
sse = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(iris)
    kmeanModel.fit(iris)
    sse.append(sum(np.min(cdist(iris, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / iris.shape[0])
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
plt.show()
