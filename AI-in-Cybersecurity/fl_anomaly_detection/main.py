"""
main.py
CLI entry point — run experiments from command line or Colab.

Usage:
    python main.py --config configs/phase1_config.yaml
    python main.py --config configs/phase2_config.yaml --strategy adaptive --alpha 0.1
"""

import argparse
import yaml
import numpy as np
from pathlib import Path

from data.download import download_nsl_kdd, download_unsw_nb15
from data.preprocess import preprocess_nsl_kdd, preprocess_unsw_nb15, load_processed
from data.partition import get_partitions
from federation.simulation import FLSimulation
from utils.logger import get_logger

logger = get_logger("main")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(config: dict):
    """Run a single FL experiment from a config dict."""

    dataset   = config["data"]["dataset"]
    strategy  = config["data"]["partitioning"]["strategy"]
    alpha     = config["data"]["partitioning"].get("dirichlet_alpha", 0.1)
    n_clients = config["federation"]["num_clients"]
    agg       = config["aggregation"]["strategy"]

    logger.info("=" * 60)
    logger.info("Experiment: %s", config["experiment"]["name"])
    logger.info("Dataset: %s | Partition: %s (α=%s) | Aggregation: %s",
                dataset, strategy, alpha, agg)
    logger.info("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────
    proc_dir = Path("./data/processed") / dataset
    if not (proc_dir / "X_train.npy").exists():
        logger.info("Preprocessing %s...", dataset)
        if dataset == "nsl_kdd":
            preprocess_nsl_kdd()
        else:
            preprocess_unsw_nb15()

    X_train, X_test, _, y_test, y_bin_train, y_bin_test = load_processed(dataset)
    logger.info("Data loaded: train=%s test=%s", X_train.shape, X_test.shape)

    # ── Step 2: Partition ─────────────────────────────────────
    partitions = get_partitions(
        X=X_train, y=y_bin_train,
        num_clients=n_clients,
        strategy=strategy,
        alpha=alpha,
        seed=config["experiment"]["seed"]
    )

    # ── Step 3: Run FL ────────────────────────────────────────
    sim = FLSimulation(config)
    sim.setup(partitions, X_test, y_bin_test)
    results = sim.run()

    # ── Step 4: Print final metrics ───────────────────────────
    m = results["final_metrics"]
    logger.info("\n" + "=" * 40)
    logger.info("FINAL RESULTS")
    logger.info("  AUC-ROC   : %.4f", m["auc_roc"])
    logger.info("  F1 Score  : %.4f", m["f1"])
    logger.info("  DR        : %.4f", m["dr"])
    logger.info("  FAR       : %.4f", m["far"])
    logger.info("  Conv.Round: %d",   results.get("convergence_round", -1))
    logger.info("=" * 40)

    return results


def run_sweep(config_path: str):
    """
    Run full experiment sweep across all conditions:
    - 2 aggregation strategies (fedavg, adaptive)
    - 4 data conditions (iid, alpha=1.0, 0.5, 0.1)
    """
    base_config = load_config(config_path)
    strategies  = ["fedavg", "adaptive"]
    conditions  = [("iid", None), ("non_iid", 1.0), ("non_iid", 0.5), ("non_iid", 0.1)]

    all_results = {}

    for strategy in strategies:
        for (part_strategy, alpha) in conditions:
            cond_name = "IID" if part_strategy == "iid" else f"α={alpha}"
            exp_name  = f"{strategy}_{cond_name}"

            config = base_config.copy()
            config["experiment"]["name"]              = exp_name
            config["aggregation"]["strategy"]         = strategy
            config["data"]["partitioning"]["strategy"] = part_strategy
            if alpha:
                config["data"]["partitioning"]["dirichlet_alpha"] = alpha

            results = run_experiment(config)
            all_results[exp_name] = results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Anomaly Detection")
    parser.add_argument("--config",   type=str, default="configs/phase1_config.yaml")
    parser.add_argument("--sweep",    action="store_true", help="Run full condition sweep")
    parser.add_argument("--strategy", type=str, default=None, help="Override aggregation strategy")
    parser.add_argument("--alpha",    type=float, default=None, help="Override Dirichlet alpha")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.strategy:
        config["aggregation"]["strategy"] = args.strategy
    if args.alpha:
        config["data"]["partitioning"]["dirichlet_alpha"] = args.alpha
        config["data"]["partitioning"]["strategy"] = "non_iid"

    if args.sweep:
        run_sweep(args.config)
    else:
        run_experiment(config)
