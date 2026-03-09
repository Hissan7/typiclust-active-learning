from data_loader import get_the_cifar10_train_loader
from feature_extractor import FeatureExtractor
from clustering import cluster_features
from selector import select_most_typical_per_cluster
import numpy as np

def main():
    budget = 10

    print("1. Loading CIFAR-10...")
    dataset, loader = get_the_cifar10_train_loader(subset_size=5000)
    print("2. Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(loader).numpy()

    print(f"3. Extracted feature shape: {features.shape}")

    print("4. Clustering features...")
    cluster_labels = cluster_features(features, num_clusters=budget)

    print(f"5. Number of cluster assignments: {len(cluster_labels)}")
    print("6. First 20 cluster labels:", cluster_labels[:20])

    print("7. Selecting most typical sample per cluster...")
    selected_indices = select_most_typical_per_cluster(features, cluster_labels, k=20)

    print(f"8. Number of selected samples: {len(selected_indices)}")
    print("9. Selected indices:", selected_indices)

    print("10. Selected labels:")
    selected_labels = [dataset[i][1] for i in selected_indices]
    print(selected_labels)

    np.save("results/selected_indices.npy", np.array(selected_indices)) #indices
    np.save("results/selected_labels.npy", np.array(selected_labels)) #labels

if __name__ == "__main__":
    main()