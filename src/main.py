from data_loader import get_the_cifar10_train_loader
from feature_extractor import FeatureExtractor
from clustering import cluster_features


def main():
    budget = 10

    print("1. Loading CIFAR-10...")
    dataset, loader = get_the_cifar10_train_loader()

    print("2. Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(loader).numpy()

    print(f"3. Extracted feature shape: {features.shape}")

    print("4. Clustering features...")
    cluster_labels = cluster_features(features, num_clusters=budget)

    print(f"5. Number of cluster assignments: {len(cluster_labels)}")
    print("6. First 20 cluster labels:", cluster_labels[:20])


if __name__ == "__main__":
    main()