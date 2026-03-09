import numpy as np

# Dataset
X = np.array([[2,3],[3,4],[5,8],[8,8],[1,2],[9,10]])

k = 2
medoids = X[:k]   # initial medoids

while True:
    clusters = [[] for _ in range(k)]

    # Assignment step
    for point in X:
        distances = [np.linalg.norm(point - m) for m in medoids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Update medoids
    new_medoids = []
    for cluster in clusters:
        cluster = np.array(cluster)
        total_distances = []

        for p in cluster:
            total = np.sum(np.linalg.norm(cluster - p, axis=1))
            total_distances.append(total)

        new_medoids.append(cluster[np.argmin(total_distances)])

    new_medoids = np.array(new_medoids)

    if np.all(medoids == new_medoids):
        break

    medoids = new_medoids

print("Final Medoids:")
print(medoids)
