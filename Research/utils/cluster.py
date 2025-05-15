from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import  os


def _agglo_cluster(pca_arr, cluster_percentile = 20):
    
    pairwise_distances = np.linalg.norm(pca_arr[:, np.newaxis] - pca_arr, axis=2)
    upper_tri_indices = np.triu_indices_from(pairwise_distances, k=1)
    distances = pairwise_distances[upper_tri_indices]

    min_distance = distances.min()
    max_distance = distances.max()

    if min_distance < 0:
        min_distance = 0.0
    if max_distance < 0:
        max_distance = 1.0

    ### closest 20% points used for agglo cluster formation
    cluster_distance_threshold = np.percentile(distances, cluster_percentile)

    if not np.isfinite(cluster_distance_threshold) or cluster_distance_threshold < 0:
        cluster_distance_threshold = 1.0  # Default positive value        

    aggl_cluster = AgglomerativeClustering(
        linkage='average',
        n_clusters=None,
        distance_threshold=cluster_distance_threshold
    )

    cluster_labels = aggl_cluster.fit_predict(pca_arr)

    print('Agglom - nunmber of clusters generated: ', len(np.unique(cluster_labels)))

    return cluster_labels

def _dbscan_cluster(pca_arr, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(pca_arr)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print('DBSCAN - number of clusters (excluding noise):', n_clusters)
    return cluster_labels


def _kmeans_cluster(pca_arr, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_arr)
    print('KMeans - number of clusters:', len(np.unique(cluster_labels)))
    return cluster_labels

class Different_clustering_algorithm:

    def __init__(self):
        pass

    def call(self):
        pass

if __name__ == '__main__':

    print('Running __cluster.py__ in main ...')

