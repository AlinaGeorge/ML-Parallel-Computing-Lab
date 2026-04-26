import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# a) Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# b) EDA
df = pd.DataFrame(X, columns=feature_names)
df["class"] = y

print(df.describe())

# ---- BASIC PAIR PLOT (EDA) ----
sns.pairplot(df, hue="class")
plt.show()
# -------------------------------

# Baseline model
model = LogisticRegression(max_iter=200)
baseline_acc = cross_val_score(model, X, y, cv=5).mean()

# c-a) Univariate Feature Selection
ufs = SelectKBest(chi2, k=2)
X_ufs = ufs.fit_transform(X, y)
ufs_acc = cross_val_score(model, X_ufs, y, cv=5).mean()

# c-b) Random Forest Feature Importance
rf = RandomForestClassifier()
rf.fit(X, y)
important_features = rf.feature_importances_.argsort()[-2:]
X_rf = X[:, important_features]
rf_acc = cross_val_score(model, X_rf, y, cv=5).mean()

# c-c) RFE using SVM
svm = SVC(kernel="linear")
rfe = RFE(svm, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
rfe_acc = cross_val_score(model, X_rfe, y, cv=5).mean()

# d & e) Results
print("\nAccuracy Comparison:")
print("Before Feature Selection:", round(baseline_acc, 3))
print("Univariate FS:", round(ufs_acc, 3))
print("Random Forest FS:", round(rf_acc, 3))
print("RFE (SVM):", round(rfe_acc, 3))
