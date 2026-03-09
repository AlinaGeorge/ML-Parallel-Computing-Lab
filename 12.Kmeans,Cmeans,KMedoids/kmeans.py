import numpy as np

# Sample dataset
X = np.array([[2,3],[3,4],[5,8],[8,8],[1,2],[9,10]])

k = 2
centroids = X[:k]   # initial centroids

while True:

    cluster1 = []
    cluster2 = []

    # Assign points to nearest centroid
    for i in X:
        dist1 = np.linalg.norm(i - centroids[0])
        dist2 = np.linalg.norm(i - centroids[1])

        if dist1 < dist2:
            cluster1.append(i)
        else:
            cluster2.append(i)

    # Calculate new centroids
    new_centroids = np.array([
        np.mean(cluster1, axis=0),
        np.mean(cluster2, axis=0)
    ])

    # Stop if centroids don't change
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

print("Final Centroids:")
print(centroids)
