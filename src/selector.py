import numpy as np
from typicality import compute_typicality_of_points


def select_most_typical_per_cluster(features: np.ndarray, cluster_labels: np.ndarray, budget: int, min_cluster_size: int = 5):
    selected_indices = []
    selected_set = set()

    unique_clusters = np.unique(cluster_labels)

    cluster_to_indices = {}
    for cluster_id in unique_clusters:
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) >= min_cluster_size:
            cluster_to_indices[cluster_id] = indices

    cluster_label_counts = {cluster_id: 0 for cluster_id in cluster_to_indices.keys()}

    while len(selected_indices) < budget and len(cluster_to_indices) > 0:
        min_count = min(cluster_label_counts.values())

        candidate_clusters = [
            cluster_id for cluster_id, count in cluster_label_counts.items()
            if count == min_count
        ]

        chosen_cluster = max(candidate_clusters, key=lambda cid: len(cluster_to_indices[cid]))
        cluster_indices = cluster_to_indices[chosen_cluster]

        available_indices = [idx for idx in cluster_indices if idx not in selected_set]

        if len(available_indices) < min_cluster_size:
            del cluster_to_indices[chosen_cluster]
            del cluster_label_counts[chosen_cluster]
            continue

        cluster_features = features[available_indices]
        k = min(20, len(available_indices) - 1)

        if k < 1:
            del cluster_to_indices[chosen_cluster]
            del cluster_label_counts[chosen_cluster]
            continue

        scores = compute_typicality_of_points(cluster_features, k=k)
        best_local_index = int(np.argmax(scores))
        best_global_index = available_indices[best_local_index]

        selected_indices.append(best_global_index)
        selected_set.add(best_global_index)
        cluster_label_counts[chosen_cluster] += 1

    return selected_indices