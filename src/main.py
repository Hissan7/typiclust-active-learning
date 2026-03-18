import os
import numpy as np
from torch.utils.data import Subset

from test_loader import get_cifar10_test_loader
from clustering import cluster_features
from selector import select_most_typical_per_cluster, select_weighted_typical_samples, select_centrality_typical_samples
from random_selector import select_random_samples
from train_classifier import create_selected_subset, train_model

from simclr.train_simclr import train_simclr
from simclr.extract_embeddings import extract_embeddings

SUBSET_SIZE = 50000
BUDGET = 200
EPOCHS = 30
SIMCLR_EPOCHS = 30

MODEL_PATH = "models/simclr_resnet18.pth"


def run_tpcrp(dataset, features, test_loader):
    print("\n--- Running TPCRP ---")

    print("1. Clustering SimCLR embeddings...")
    cluster_labels = cluster_features(features, num_clusters=BUDGET)

    print("2. Selecting most typical sample per cluster...")
    selected_indices = select_most_typical_per_cluster(
        features,
        cluster_labels,
        budget=BUDGET
    )

    print(f"3. Number of selected samples: {len(selected_indices)}")

    selected_labels = [dataset[i][1] for i in selected_indices]
    print("4. Selected labels:", selected_labels)

    train_subset = create_selected_subset(dataset, selected_indices)

    print("5. Training classifier on TPCRP samples...")
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



def run_weighted_tpcrp(dataset, features, test_loader):
    print("\n--- Running Modified TPCRP (Cluster-Weighted) ---")

    print("1. Clustering SimCLR embeddings...")
    cluster_labels = cluster_features(features, num_clusters=BUDGET)

    print("2. Selecting weighted typical samples...")
    selected_indices = select_weighted_typical_samples(
        features,
        cluster_labels,
        budget=BUDGET
    )

    print(f"3. Number of selected samples: {len(selected_indices)}")

    selected_labels = [dataset[i][1] for i in selected_indices]
    print("4. Selected labels:", selected_labels)

    train_subset = create_selected_subset(dataset, selected_indices)

    print("5. Training classifier on modified TPCRP samples...")
    _, test_acc = train_model(train_subset, test_loader, EPOCHS, batch_size=16)

    return selected_indices, test_acc



def run_centrality_tpcrp(dataset, features, test_loader):

    print("\n--- Running Centrality-Aware TPCRP ---")

    print("1. Clustering embeddings...")
    cluster_labels = cluster_features(features, num_clusters=BUDGET)

    print("2. Selecting centrality-weighted typical samples...")
    selected_indices = select_centrality_typical_samples(
        features,
        cluster_labels,
        budget=BUDGET
    )

    print(f"3. Number of selected samples: {len(selected_indices)}")

    selected_labels = [dataset[i][1] for i in selected_indices]
    print("4. Selected labels:", selected_labels)

    train_subset = create_selected_subset(dataset, selected_indices)

    print("5. Training classifier...")
    _, test_acc = train_model(train_subset, test_loader, EPOCHS, batch_size=16)

    return selected_indices, test_acc



def main():
    print("Loading CIFAR-10 test set...")
    _, test_loader = get_cifar10_test_loader()

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Train SimCLR only if model doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("\n--- Training SimCLR ---")
        train_simclr(
            epochs=SIMCLR_EPOCHS,
            batch_size=128,
            lr=1e-3,
            temperature=0.5,
            save_path=MODEL_PATH
        )
    else:
        print("\n--- Using existing SimCLR model ---")

    print("\n--- Extracting SimCLR embeddings ---")
    dataset, features = extract_embeddings(MODEL_PATH, train=True)

    features = features[:SUBSET_SIZE]
    dataset = Subset(dataset, list(range(SUBSET_SIZE)))

    print(f"Embedding shape: {features.shape}")

    # Run TPCRP
    tpcrp_indices, tpcrp_acc = run_tpcrp(dataset, features, test_loader)

    # Run modified TPCRP (centrality-weighted-typicality)
    centrality_indices, centrality_acc = run_centrality_tpcrp(dataset, features, test_loader)

    # Run Random baseline
    random_indices, random_acc = run_random_baseline(dataset, test_loader)

    np.save("results/tpcrp_indices.npy", np.array(tpcrp_indices)) # Normal TPCRP
    np.save("results/weighted_tpcrp_indices.npy", np.array(centrality_indices)) # Modified TPCRP
    np.save("results/random_indices.npy", np.array(random_indices)) # Random baseline

    print("\n--- Final Comparison ---")
    print(f"TPCRP Test Accuracy:           {tpcrp_acc:.2f}%")
    print(f"Centrality TPCRP Accuracy (Modified implementation):     {centrality_acc:.2f}%")
    print(f"Random Test Accuracy:          {random_acc:.2f}%")
    print(
        f"Budget: {BUDGET} | Subset size: {SUBSET_SIZE} | "
        f"Epochs: {EPOCHS} | SimCLR epochs: {SIMCLR_EPOCHS}"
    )


if __name__ == "__main__":
        main()