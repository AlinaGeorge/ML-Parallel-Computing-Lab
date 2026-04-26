# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import skfuzzy as fuzz

# Load Iris dataset
X = load_iris().data

# Standardize data
X = StandardScaler().fit_transform(X)

# -------- K-Means --------
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# -------- K-Medoids --------
kmedoids = KMedoids(n_clusters=3, random_state=42)
labels_kmedoids = kmedoids.fit_predict(X)

# -------- Fuzzy C-Means --------
_, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X.T, c=3, m=2, error=0.005, maxiter=1000
)
labels_fcm = np.argmax(u, axis=0)

# -------- Output --------
print("K-Means Labels:\n", labels_kmeans)
print("\nK-Medoids Labels:\n", labels_kmedoids)
print("\nFuzzy C-Means Labels:\n", labels_fcm)
