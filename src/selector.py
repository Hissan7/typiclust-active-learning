import numpy as np
from typicality import compute_typicality_of_points


def select_most_typical_per_cluster(features: np.ndarray, cluster_labels: np.ndarray, k = 20):
    selected_indices = []

    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_features = features[cluster_indices]

        scores = compute_typicality_of_points(cluster_features, k=k)
        best_local_index = np.argmax(scores)
        best_global_index = cluster_indices[best_local_index]

        selected_indices.append(best_global_index)

    return selected_indices