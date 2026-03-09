import numpy as np


def select_random_samples(dataset_size: int, budget: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(dataset_size, size=budget, replace=False)
    return selected_indices.tolist()