import os
import numpy as np

from data_loader import get_the_cifar10_train_loader
from test_loader import get_cifar10_test_loader
from feature_extractor import FeatureExtractor
from clustering import cluster_features
from selector import select_most_typical_per_cluster
from random_selector import select_random_samples
from train_classifier import create_selected_subset, train_model

SUBSET_SIZE = 50000
BUDGET = 1000
EPOCHS = 10


def run_tpcrp(dataset, loader, test_loader):
    print("\n--- Running TPCRP ---")

    print("1. Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(loader).numpy()

    print(f"2. Extracted feature shape: {features.shape}")

    print("3. Clustering features...")
    cluster_labels = cluster_features(features, num_clusters=BUDGET)

    print("4. Selecting most typical sample per cluster...")
    selected_indices = select_most_typical_per_cluster(features, cluster_labels, k=20)

    print(f"5. Number of selected samples: {len(selected_indices)}")
    selected_labels = [dataset[i][1] for i in selected_indices]
    print("6. Selected labels:", selected_labels)

    train_subset = create_selected_subset(dataset, selected_indices)

    print("7. Training classifier on TPCRP samples...")
    _, test_acc = train_model(train_subset, test_loader, EPOCHS, batch_size=16)

    return selected_indices, test_acc


def run_random_baseline(dataset, test_loader):
    print("\n--- Running Random Baseline ---")

    selected_indices = select_random_samples(len(dataset), BUDGET, seed=42)

    print(f"1. Number of selected samples: {len(selected_indices)}")
    selected_labels = [dataset[i][1] for i in selected_indices]
    print("2. Selected labels:", selected_labels)

    train_subset = create_selected_subset(dataset, selected_indices)

    print("3. Training classifier on random samples...")
    _, test_acc = train_model(train_subset, test_loader, EPOCHS, batch_size=16)

    return selected_indices, test_acc


def main():
    print("Loading CIFAR-10 train set...")
    dataset, loader = get_the_cifar10_train_loader(subset_size=SUBSET_SIZE)

    print("Loading CIFAR-10 test set...")
    _, test_loader = get_cifar10_test_loader()

    os.makedirs("results", exist_ok=True)

    tpcrp_indices, tpcrp_acc = run_tpcrp(dataset, loader, test_loader)
    random_indices, random_acc = run_random_baseline(dataset, test_loader)

    np.save("results/tpcrp_indices.npy", np.array(tpcrp_indices))
    np.save("results/random_indices.npy", np.array(random_indices))

    print("\n--- Final Comparison ---")
    print(f"TPCRP Test Accuracy:  {tpcrp_acc:.2f}%")
    print(f"Random Test Accuracy: {random_acc:.2f}%")
    print(f"Budget: {BUDGET} | Subset size: {SUBSET_SIZE} | Epochs: {EPOCHS}")


if __name__ == "__main__":
    main()