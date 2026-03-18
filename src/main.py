import os
import numpy as np
from torch.utils.data import Subset
from clustering import cluster_features
from plot_results import plot_bar,plot_runs
from simclr.train_simclr import train_simclr
from test_loader import get_cifar10_test_loader
from random_selector import select_random_samples
from simclr.extract_embeddings import extract_embeddings
from train_classifier import create_selected_subset, train_model
from selector import select_most_typical_per_cluster, select_weighted_typical_samples, select_centrality_typical_samples


SUBSET_SIZE = 50000
BUDGET = 200
EPOCHS = 30
SIMCLR_EPOCHS = 30
NUM_RUNS = 10 #5

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

    tpcrp_results = [] # original tpcrp
    centrality_results = [] # modified verson
    random_results = [] #random

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

    for run in range(NUM_RUNS):

        print(f"\n===== RUN {run+1} =====")

        # Run TPCRP
        tpcrp_indices, tpcrp_acc = run_tpcrp(dataset, features, test_loader)

        # Run modified TPCRP (centrality-weighted-typicality)
        centrality_indices, centrality_acc = run_centrality_tpcrp(dataset, features, test_loader)

        # Run Random baseline
        random_indices, random_acc = run_random_baseline(dataset, test_loader)

        # store results
        tpcrp_results.append(tpcrp_acc)
        centrality_results.append(centrality_acc)
        random_results.append(random_acc)

    # save final selections (last run)
    np.save("results/tpcrp_indices.npy", np.array(tpcrp_indices)) # Normal TPCRP
    np.save("results/weighted_tpcrp_indices.npy", np.array(centrality_indices)) # Modified TPCRP
    np.save("results/random_indices.npy", np.array(random_indices)) # Random baseline

    # compute stats
    tpcrp_mean, tpcrp_std = np.mean(tpcrp_results), np.std(tpcrp_results)
    centrality_mean, centrality_std = np.mean(centrality_results), np.std(centrality_results)
    random_mean, random_std = np.mean(random_results), np.std(random_results)

    print("\n--- Final Comparison ---")
    print(f"TPCRP Runs: {tpcrp_results}")
    print(f"Centrality Runs: {centrality_results}")
    print(f"Random Runs: {random_results}")

    print("\n--- Mean ± Std ---")
    print(f"TPCRP: {tpcrp_mean:.2f} ± {tpcrp_std:.2f}")
    print(f"Centrality TPCRP: {centrality_mean:.2f} ± {centrality_std:.2f}")
    print(f"Random: {random_mean:.2f} ± {random_std:.2f}")

    print(
        f"\nBudget: {BUDGET} | Subset size: {SUBSET_SIZE} | "
        f"Epochs: {EPOCHS} | SimCLR epochs: {SIMCLR_EPOCHS}"
    )

    # plot metrics 
    plot_bar(tpcrp_results, centrality_results, random_results)
    plot_runs(tpcrp_results, centrality_results, random_results)
    print("\nSaved plots to results/bar_plot.png and results/line_plot.png")


if __name__ == "__main__":
    main()