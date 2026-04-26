# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Import required libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Classifier
clf = LogisticRegression(max_iter=200)

# Function to compute cross-validation score
def cv_score(X_new, y):
    scores = cross_val_score(clf, X_new, y, cv=5)
    return scores.mean()

print("\nCASE 1: Reduction to 2 features")

X_pca_2 = PCA(n_components=2).fit_transform(X)
X_lda_2 = LDA(n_components=2).fit_transform(X, y)   # Max possible = 2
X_tsne_2 = TSNE(n_components=2, random_state=42).fit_transform(X)
X_svd_2 = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)

print("PCA  (2D):", cv_score(X_pca_2, y))
print("LDA  (2D):", cv_score(X_lda_2, y))
print("t-SNE(2D):", cv_score(X_tsne_2, y))
print("SVD  (2D):", cv_score(X_svd_2, y))

print("\nCASE 2: Reduction to 3 features")

X_pca_3 = PCA(n_components=3).fit_transform(X)
X_tsne_3 = TSNE(n_components=3, random_state=42).fit_transform(X)
X_svd_3 = TruncatedSVD(n_components=3, random_state=42).fit_transform(X)

print("PCA  (3D):", cv_score(X_pca_3, y))
print("t-SNE(3D):", cv_score(X_tsne_3, y))
print("SVD  (3D):", cv_score(X_svd_3, y))

