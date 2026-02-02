import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# a. Load dataset and split X, y

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")
y_named = y.map(dict(enumerate(iris.target_names)))

print("Dataset shape:", X.shape)
print("\nClass distribution:\n", y_named.value_counts())


# b. Exploratory Data Analysis

print("\nSummary statistics:\n", X.describe())

df = X.copy()
df["species"] = y_named

sns.pairplot(df, hue="species", diag_kind="kde")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# c-a. Univariate Feature Selection

uni_selector = SelectKBest(score_func=f_classif, k=2)
X_uni = uni_selector.fit_transform(X, y)
uni_features = X.columns[uni_selector.get_support()]
print("\nUnivariate selected features:", list(uni_features))


# c-b. Random Forest Feature Importance

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nRandom Forest Feature Importance:\n", rf_importance.sort_values(ascending=False))

rf_importance.sort_values().plot(kind="barh", title="Random Forest Feature Importance")
plt.show()

top_rf_features = rf_importance.sort_values(ascending=False).index[:2]

# c-c. RFE using SVM

svm_linear = SVC(kernel="linear")
rfe = RFE(estimator=svm_linear, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
rfe_features = X.columns[rfe.support_]
print("\nRFE selected features:", list(rfe_features))


# d & e. Model Evaluation and Comparison

def evaluate_model(X_features, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.3, random_state=42, stratify=y
    )
    model = SVC(kernel="rbf", gamma="scale")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{label} Accuracy: {acc:.4f}")
    return acc

print("\nModel Performance Comparison:")
acc_all = evaluate_model(X, y, "All Features")
acc_uni = evaluate_model(X[uni_features], y, "Univariate Features")
acc_rf  = evaluate_model(X[top_rf_features], y, "Random Forest Features")
acc_rfe = evaluate_model(X[rfe_features], y, "RFE Features")


# Final Summary

summary = pd.DataFrame({
    "Feature Set": ["All Features", "Univariate", "Random Forest", "RFE"],
    "No. of Features": [4, 2, 2, 2],
    "Accuracy": [acc_all, acc_uni, acc_rf, acc_rfe]
})

print("\nFinal Accuracy Comparison:")
print(summary)
