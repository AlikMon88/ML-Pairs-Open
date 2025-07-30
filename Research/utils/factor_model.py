'''
Factor/Feature-Modelling of Price-Data + Fundamental (or On-Chain data) --> PCA + Clustering 
'''

from .cluster import _agglo_cluster
from .methods import SelectCointegratedPairs
from .methods import SelectCointegratedPairs
from .methods import FilterHighCorrelationPairs  

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

class FactorModelPairs():
    def __init__(self, stocks, variance_perc, is_plot=False):
        self.is_plot = is_plot
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_perc)
        self.stocks = stocks
        
    def decomp(self, val_arr):
        scaled_arr = self.scaler.fit_transform(val_arr)
        pca_arr = self.pca.fit_transform(scaled_arr)
        print('Decomposed-Dim: ', pca_arr.shape[-1])
        
        return pca_arr
    
    def create_clusters(self, pca_arr):
        cluster_labels = _agglo_cluster(pca_arr)
        if self.is_plot:
            
            fig, axs = plt.subplots(1, 3, figsize=(14, 5))

            sns.scatterplot(x=pca_arr[:, 0], y=pca_arr[:, 1], hue=cluster_labels, palette="viridis", s=100, edgecolor="black", ax = axs[0])
            sns.scatterplot(x=pca_arr[:, 1], y=pca_arr[:, 2], hue=cluster_labels, palette="viridis", s=100, edgecolor="black", ax = axs[1])
            sns.scatterplot(x=pca_arr[:, 0], y=pca_arr[:, 2], hue=cluster_labels, palette="viridis", s=100, edgecolor="black", ax = axs[2])

            for i, stock in enumerate(self.stocks):
                axs[0].annotate(stock, (pca_arr[i, 0], pca_arr[i, 1]), fontsize=9, alpha=0.75)

            for i, stock in enumerate(self.stocks):
                axs[1].annotate(stock, (pca_arr[i, 1], pca_arr[i, 2]), fontsize=9, alpha=0.75)

            for i, stock in enumerate(self.stocks):
                axs[2].annotate(stock, (pca_arr[i, 0], pca_arr[i, 2]), fontsize=9, alpha=0.75)


            axs[0].set_title("Agglomerative Clustering of Stocks")
            axs[0].set_xlabel("PCA Component 1")
            axs[0].set_ylabel("PCA Component 2")

            axs[1].set_title("Agglomerative Clustering of Stocks")
            axs[1].set_xlabel("PCA Component 1")
            axs[1].set_ylabel("PCA Component 2")

            axs[2].set_title("Agglomerative Clustering of Stocks")
            axs[2].set_xlabel("PCA Component 1")
            axs[2].set_ylabel("PCA Component 2")

            axs[0].grid()
            axs[1].grid()
            axs[2].grid()

            plt.legend(title="Cluster")
            plt.tight_layout(pad=2)
            plt.show()    
        
        return cluster_labels
    
    def create_pairs(self, cluster_labels, df_uni):
        ## perform cointegration
        pairs = SelectCointegratedPairs(symbols=self.stocks, cluster_labels=self.cluster_labels, history=df_uni)
        print('Number of Pairs Found: ', len(self.pairs.keys()))

        return pairs
    
    def run(self, val_arr, df_uni):
        pca_arr = self.decomp(val_arr)
        cluster_labels = self.create_clusters(pca_arr)
        pairs = self.create_pairs(cluster_labels, df_uni)
        
        print('Factors Successfully Modeled.')
        
        return pairs