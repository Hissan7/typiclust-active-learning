from sklearn.cluster import KMeans


def cluster_features(features, num_clusters: int):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels