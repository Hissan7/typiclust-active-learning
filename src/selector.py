import numpy as np
from typicality import compute_typicality_of_points


def select_most_typical_per_cluster(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    budget: int,
    min_cluster_size: int = 5
):
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


def select_weighted_typical_samples(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    budget: int,
    min_cluster_size: int = 5
):
    """
    Modified TPCRP:
    allocate selections approximately proportional to cluster size,
    then pick the most typical points inside each cluster.
    """
    selected_indices = []

    unique_clusters = np.unique(cluster_labels)

    cluster_to_indices = {}
    cluster_sizes = {}
    for cluster_id in unique_clusters:
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) >= min_cluster_size:
            cluster_to_indices[cluster_id] = indices
            cluster_sizes[cluster_id] = len(indices)

    if not cluster_to_indices:
        return selected_indices

    total_points = sum(cluster_sizes.values())

    # Initial allocation proportional to cluster size
    allocation = {}
    remainders = []

    allocated_total = 0
    for cluster_id, size in cluster_sizes.items():
        exact_share = budget * (size / total_points)
        base_share = int(np.floor(exact_share))
        allocation[cluster_id] = base_share
        allocated_total += base_share
        remainders.append((cluster_id, exact_share - base_share))

    remaining_budget = budget - allocated_total
    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(remaining_budget):
        cluster_id = remainders[i % len(remainders)][0]
        allocation[cluster_id] += 1

    for cluster_id in allocation:
        if allocation[cluster_id] == 0 and len(cluster_to_indices[cluster_id]) >= min_cluster_size:
            allocation[cluster_id] = 1

    while sum(allocation.values()) > budget:
        removable = [cid for cid, count in allocation.items() if count > 1]
        if not removable:
            break
        smallest = min(removable, key=lambda cid: cluster_sizes[cid])
        allocation[smallest] -= 1

    # Select top typical points within each cluster
    for cluster_id, num_to_select in allocation.items():
        if num_to_select <= 0:
            continue

        cluster_indices = cluster_to_indices[cluster_id]
        cluster_features = features[cluster_indices]

        k = min(20, len(cluster_indices) - 1)
        if k < 1:
            continue

        scores = compute_typicality_of_points(cluster_features, k=k)
        ranked_local_indices = np.argsort(scores)[::-1]  # descending

        take = min(num_to_select, len(ranked_local_indices))
        chosen_local = ranked_local_indices[:take]
        chosen_global = cluster_indices[chosen_local]

        selected_indices.extend(chosen_global.tolist())

    selected_indices = selected_indices[:budget]

    return selected_indices

from typicality import compute_typicality_of_points, compute_centrality


def select_centrality_typical_samples(features, cluster_labels, budget, alpha=0.7):
    """
    Modified TPCRP:
    combine typicality + centrality score
    """

    selected_indices = []

    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:

        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_features = features[cluster_indices]

        typicality = compute_typicality_of_points(cluster_features)

        centrality = compute_centrality(cluster_features)

        # normalize scores
        typicality = (typicality - typicality.min()) / (typicality.max() - typicality.min() + 1e-8)
        centrality = (centrality - centrality.min()) / (centrality.max() - centrality.min() + 1e-8)

        score = alpha * typicality + (1 - alpha) * centrality

        best_index_local = np.argmax(score)

        best_index_global = cluster_indices[best_index_local]

        selected_indices.append(best_index_global)

    return selected_indices[:budget]