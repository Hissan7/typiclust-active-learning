import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality_of_points(cluster_features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Computes typicality for each point in a cluster
    typicality = 1 / average distance to k nearest neighbours
    """

    if len(cluster_features) == 1:
        return np.array([float("inf")])

    k = min(k, len(cluster_features) - 1)

    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(cluster_features)

    distances, _ = nbrs.kneighbors(cluster_features)

    neighbour_distances = distances[:, 1:]

    avg_distances = neighbour_distances.mean(axis=1)

    typicality = 1.0 / (avg_distances + 1e-8)

    return typicality


def compute_centrality(cluster_features: np.ndarray) -> np.ndarray:
    """
    Computes centrality = inverse distance to cluster centroid
    """

    centroid = np.mean(cluster_features, axis=0)

    distances = np.linalg.norm(cluster_features - centroid, axis=1)

    centrality = 1.0 / (distances + 1e-8)

    return centrality