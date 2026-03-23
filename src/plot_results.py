import os
import numpy as np
import matplotlib.pyplot as plt


def plot_bar(tpcrp_results, centrality_results, random_results):
    os.makedirs("results", exist_ok=True)

    means = [
        np.mean(tpcrp_results),
        np.mean(centrality_results),
        np.mean(random_results),
    ]

    stds = [
        np.std(tpcrp_results),
        np.std(centrality_results),
        np.std(random_results),
    ]

    labels = ["TPCRP", "Centrality TPCRP", "Random"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Mean Accuracy Across Runs")
    plt.tight_layout()
    plt.savefig("results/bar_plot.png")
    plt.close()


def plot_runs(tpcrp_results, centrality_results, random_results):
    os.makedirs("results", exist_ok=True)

    runs = np.arange(1, len(tpcrp_results) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(runs, tpcrp_results, marker="o", label="TPCRP")
    plt.plot(runs, centrality_results, marker="o", label="Centrality TPCRP")
    plt.plot(runs, random_results, marker="o", label="Random")
    plt.xlabel("Run")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy Across Runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/line_plot.png")
    plt.close()