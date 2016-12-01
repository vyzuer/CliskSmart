import time
import os
import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn import metrics

def kmeans(data, n_clusters=10, n_iter=5000):

    # Perform clustering with KMeans
    timer = time.time()
    print 'Performing kmeans...'
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=n_iter, tol=0.0, n_jobs=-1)
    kmeans.fit(data)

    print 'kmeans done. '
    print 'Total running time ', time.time() - timer

    silhouette_score = metrics.silhouette_score(data, kmeans.labels_)

    return kmeans, silhouette_score


def ap(data, norm=False, reduce_dim=False, n_dims=0):

    # Perform clustering
    timer = time.time()
    print 'Performing AP...'
    aprop = AffinityPropagation(damping=0.5, convergence_iter=50, max_iter=10000).fit(data)
    
    n_clusters = len(aprop.cluster_centers_indices_)
    print 'AP done. '
    print 'Total running time ', time.time() - timer

    print('Estimated number of clusters: %d' % n_clusters)
    print("Silhouette Coefficient: %0.6f"
          % metrics.silhouette_score(data, aprop.labels_, metric='sqeuclidean'))
    
    return aprop, n_clusters

