import os
import numpy as np

from data_loader import get_the_cifar10_train_loader
from test_loader import get_cifar10_test_loader
from feature_extractor import FeatureExtractor
from clustering import cluster_features
from selector import select_most_typical_per_cluster
from train_classifier import create_selected_subset, train_model


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

    selected_labels = [dataset[i][1] for i in selected_indices]
    print("10. Selected labels:", selected_labels)

    os.makedirs("results", exist_ok=True)
    np.save("results/selected_indices.npy", np.array(selected_indices))
    np.save("results/selected_labels.npy", np.array(selected_labels))

    print("11. Loading CIFAR-10 test set...")
    _, test_loader = get_cifar10_test_loader()

    print("12. Creating labelled subset...")
    train_subset = create_selected_subset(dataset, selected_indices)

    print("13. Training classifier...")
    _, test_acc = train_model(train_subset, test_loader, epochs=5, batch_size=16)

    print(f"14. Final test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()