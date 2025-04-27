# Agglomerative Clustering Theory for Pairs Trading

## 1. Introduction
Agglomerative clustering is a bottom-up hierarchical clustering method where each observation starts in its own cluster, and pairs of clusters are merged iteratively based on a distance criterion until a single cluster remains or a stopping condition is met.

---

## 2. Key Concepts

- **Distance Metrics**:
  - Euclidean Distance:  
    $$
    d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
    $$

- **Linkage Criteria**:
  - **Single Linkage**: Minimum distance between points.
  - **Complete Linkage**: Maximum distance between points.
  - **Average Linkage**: Average distance between all points.
  - **Ward's Method**: Minimize variance between clusters:

    $$
    D(A, B) = \frac{|A||B|}{|A| + |B|} \| \bar{x}_A - \bar{x}_B \|^2
    $$

- **Dendrogram**: Tree diagram showing the merging steps.

---

## 3. Application in Pairs Trading

- **Objective**: Group stocks with similar behaviors for pairs trading (cointegration, mean-reversion).
- **Steps**:
  1. Calculate pairwise distances (e.g., $1 - \rho_{ij}$ correlation distance).
  2. Perform agglomerative clustering.
  3. Cut dendrogram at a threshold to form clusters.
  4. Identify candidate stock pairs.

---

## 4. Mathematical Background

- **Algorithm**:
  1. Treat each observation as its own cluster.
  2. Merge two closest clusters based on linkage.
  3. Update the distance matrix.
  4. Repeat until one cluster or threshold reached.

- **Updating Distances**:
  - Single Linkage:

    $$
    d(A \cup B, C) = \min(d(A,C), d(B,C))
    $$

  - Complete Linkage:

    $$
    d(A \cup B, C) = \max(d(A,C), d(B,C))
    $$

  - Average Linkage:

    $$
    d(A \cup B, C) = \frac{|A|d(A,C) + |B|d(B,C)}{|A|+|B|}
    $$

---

## 5. Advantages and Limitations

- **Advantages**:
  - No need to pre-specify number of clusters.
  - Different distance/linkage flexibility.

- **Limitations**:
  - Computationally expensive (O($n^3$)).
  - Sensitive to noise and outliers.

---

*References*:
- Hastie, Tibshirani, Friedman (The Elements of Statistical Learning)
- Murtagh, F. & Contreras, P. (Methods of Hierarchical Clustering)
