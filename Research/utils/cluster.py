from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import  os


def _agglo_cluster(pca_arr, cluster_distance_threshold):
    
    aggl_cluster = AgglomerativeClustering(
        linkage='average',
        n_clusters=None,
        distance_threshold=cluster_distance_threshold
    )

    cluster_labels = aggl_cluster.fit_predict(pca_arr)
    print('nunmber of clusters generated: ', len(np.unique(cluster_labels)))

    return cluster_labels

if __name__ == '__main__':

    print('Running __cluster.py__ in main ...')

