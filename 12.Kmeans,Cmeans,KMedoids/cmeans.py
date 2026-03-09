import numpy as np

# Dataset
X = np.array([[2,3],[3,4],[5,8],[8,8],[1,2],[9,10]])

c = 2        # number of clusters
m = 2        # fuzziness parameter
n = len(X)

# Initialize membership matrix
U = np.random.dirichlet(np.ones(c), size=n)

for _ in range(10):

    # Compute cluster centers
    centers = []
    for j in range(c):
        numerator = np.sum((U[:, j]**m).reshape(-1,1) * X, axis=0)
        denominator = np.sum(U[:, j]**m)
        centers.append(numerator / denominator)

    centers = np.array(centers)

    # Update membership matrix
    for i in range(n):
        for j in range(c):
            dist = np.linalg.norm(X[i] - centers[j])
            denom = 0
            for k in range(c):
                denom += (dist / np.linalg.norm(X[i] - centers[k])) ** (2 / (m-1))
            U[i][j] = 1 / denom

print("Cluster Centers:")
print(centers)

print("\nMembership Matrix:")
print(U)
