"""
data/partition.py
IID and Non-IID Dirichlet partitioning for FL clients.
Follows Li et al. (2022) NIID-Bench methodology.
"""

import numpy as np
from typing import List, Tuple, Dict


def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Randomly shuffle and split data equally across clients.
    Each client gets an i.i.d. sample of the global distribution.

    Returns:
        List of (X_client, y_client) tuples, one per client.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    splits  = np.array_split(indices, num_clients)

    partitions = [(X[idx], y[idx]) for idx in splits]

    print(f"[IID] {num_clients} clients | "
          f"avg {len(X)//num_clients} samples each")
    return partitions


def partition_non_iid_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.1,
    seed: int = 42,
    min_samples: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Dirichlet Non-IID partitioning (Li et al. 2022).

    Each client receives a proportion of each class drawn from Dir(alpha).
    Lower alpha = higher heterogeneity (each client dominated by 1-2 classes).

    Args:
        X           : Feature array
        y           : Label array (int class indices)
        num_clients : Number of FL clients
        alpha       : Dirichlet concentration parameter
                      0.1 = high heterogeneity  (stress test)
                      0.5 = moderate            (realistic)
                      1.0 = mild                (near-IID)
        seed        : Random seed for reproducibility
        min_samples : Minimum samples per client (prevents empty clients)

    Returns:
        List of (X_client, y_client) tuples, one per client.
    """
    rng        = np.random.default_rng(seed)
    classes    = np.unique(y)
    num_classes = len(classes)

    # Index samples by class
    class_indices = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    # Draw proportions from Dirichlet for each class
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in classes:
        idxs = class_indices[c]
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # Ensure minimum samples
        proportions = np.maximum(proportions, min_samples / len(idxs))
        proportions /= proportions.sum()
        # Split class indices proportionally
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        chunks = np.split(idxs, splits)
        for i, chunk in enumerate(chunks):
            client_indices[i].extend(chunk.tolist())

    partitions = []
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        rng.shuffle(idx)
        partitions.append((X[idx], y[idx]))

    # Log distribution stats
    sizes = [len(p[0]) for p in partitions]
    print(f"[Non-IID α={alpha}] {num_clients} clients | "
          f"min={min(sizes)} max={max(sizes)} avg={int(np.mean(sizes))} samples")
    _log_label_distribution(partitions, classes)

    return partitions


def _log_label_distribution(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    classes: np.ndarray
):
    """Print per-client class distribution summary."""
    print(f"  {'Client':<8}", end="")
    for c in classes:
        print(f"  Class{c:<3}", end="")
    print()
    for i, (_, y_c) in enumerate(partitions):
        print(f"  {i:<8}", end="")
        for c in classes:
            pct = 100 * np.sum(y_c == c) / max(len(y_c), 1)
            print(f"  {pct:>6.1f}%", end="")
        print()


def get_partitions(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    strategy: str = "non_iid",
    alpha: float = 0.1,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Unified entry point for partitioning.

    Args:
        strategy : "iid" or "non_iid"
        alpha    : Dirichlet alpha (only used for non_iid)
    """
    if strategy == "iid":
        return partition_iid(X, y, num_clients, seed)
    elif strategy == "non_iid":
        return partition_non_iid_dirichlet(X, y, num_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'iid' or 'non_iid'.")
