import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

# Load data
X, y = load_iris(return_X_y=True)

# Standardize
X_scaled = StandardScaler().fit_transform(X)

model = LogisticRegression(max_iter=200)

# PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)
pca_score = cross_val_score(model, X_pca, y, cv=5).mean()

# LDA (max components = classes-1 = 2)
X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X_scaled, y)
lda_score = cross_val_score(model, X_lda, y, cv=5).mean()

# SVD
X_svd = TruncatedSVD(n_components=2).fit_transform(X_scaled)
svd_score = cross_val_score(model, X_svd, y, cv=5).mean()

# t-SNE (no CV directly, use transformed data)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
tsne_score = cross_val_score(model, X_tsne, y, cv=5).mean()

pca_score, lda_score, svd_score, tsne_score
