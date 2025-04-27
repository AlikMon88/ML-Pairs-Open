# Agglomerative Clustering Theory for Pairs Trading

## 1. Introduction
Agglomerative clustering is a bottom-up hierarchical clustering method where each observation starts in its own cluster, and pairs of clusters are merged iteratively based on a distance criterion until a single cluster remains or a stopping condition is met.

---

## 2. Key Concepts

- **Distance Metrics**:
  - **Euclidean Distance**:  
    `d(x, y) = sqrt(sum_i (xᵢ - yᵢ)²)`

- **Linkage Criteria**:
  - **Single Linkage**: Minimum distance between points.
  - **Complete Linkage**: Maximum distance between points.
  - **Average Linkage**: Average distance between all points.
  - **Ward's Method**: Minimize variance between clusters:  
    `D(A, B) = (|A| * |B|) / (|A| + |B|) * ||mean(x_A) - mean(x_B)||²`

- **Dendrogram**: A tree diagram showing the merging steps of the clusters.

---

## 3. Application in Pairs Trading

- **Objective**: Group stocks with similar behaviors to find pairs exhibiting cointegration and mean-reversion characteristics.
- **Steps**:
  1. Calculate pairwise distances (e.g., `1 - correlation_ij` as a distance metric).
  2. Perform agglomerative clustering based on a chosen linkage.
  3. Cut the dendrogram at a threshold to form clusters.
  4. Identify candidate stock pairs from clusters.

---

## 4. Mathematical Background

- **Algorithm**:
  1. Treat each observation as its own cluster.
  2. Merge two closest clusters based on the chosen linkage criterion.
  3. Update the distance matrix after each merge.
  4. Repeat until one cluster remains or a threshold is reached.

- **Updating Distances**:
  - **Single Linkage**:  
    `d(A ∪ B, C) = min(d(A,C), d(B,C))`
  
  - **Complete Linkage**:  
    `d(A ∪ B, C) = max(d(A,C), d(B,C))`
  
  - **Average Linkage**:  
    `d(A ∪ B, C) = (|A| * d(A,C) + |B| * d(B,C)) / (|A| + |B|)`

---

## 5. Advantages and Limitations

- **Advantages**:
  - No need to pre-specify the number of clusters.
  - Flexibility in choosing different distance metrics and linkage methods.

- **Limitations**:
  - Computational complexity is high: `O(n³)`.
  - Sensitive to noise and outliers, which may affect cluster stability.

---

## References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*.
- Murtagh, F. & Contreras, P. — *Methods of Hierarchical Clustering*.
