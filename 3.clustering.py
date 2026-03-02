# ===============================
# K-MEANS CUSTOMER CLUSTERING
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -------------------------------
# 1. LOAD DATASET
# -------------------------------

# Replace with your dataset path
data = pd.read_csv("supermarket_sales.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# -------------------------------
# 2. DATA PREPROCESSING
# -------------------------------

# Select relevant numerical features
features = [
    "Unit price",
    "Quantity",
    "Tax 5%",
    "Total",
    "cogs",
    "gross income",
    "Rating"
]

X = data[features]

# Handle missing values (if any)
X = X.fillna(X.mean())

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. ELBOW METHOD
# -------------------------------

inertia = []

K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# -------------------------------
# 4. APPLY K-MEANS
# -------------------------------

optimal_k = 4   # Change based on elbow plot

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

print("\nCluster Counts:")
print(data["Cluster"].value_counts())

# -------------------------------
# 5. ANALYZE CLUSTER CHARACTERISTICS
# -------------------------------

cluster_summary = data.groupby("Cluster")[features].mean()

print("\nCluster Characteristics (Mean Values):")
print(cluster_summary)

# -------------------------------
# 6. PCA FOR VISUALIZATION
# -------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data["PCA1"] = X_pca[:, 0]
data["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2",
    data=data,
    s=60
)
plt.title("Customer Clusters (PCA Projection)")
plt.show()

# -------------------------------
# 7. SILHOUETTE SCORE
# -------------------------------

score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", score)
