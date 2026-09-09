"""federation/simulation.py — fully debugged"""
import numpy as np
import torch
import json
import time
from pathlib import Path
from copy import deepcopy
from typing import List, Dict

from models.autoencoder import build_autoencoder
from federation.client import FLClient
from aggregation.adaptive import get_aggregator, reset_scaffold_state
from evaluation.metrics import compute_metrics, compute_global_threshold
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def _defaults(config: dict) -> dict:
    config.setdefault("experiment", {})
    config["experiment"].setdefault("seed",   42)
    config["experiment"].setdefault("device", "cpu")
    config["experiment"].setdefault("name",   "experiment")
    config.setdefault("federation", {})
    config["federation"].setdefault("num_clients", 10)
    config["federation"].setdefault("num_rounds",  100)
    config.setdefault("local_training", {})
    config["local_training"].setdefault("epochs",                  5)
    config["local_training"].setdefault("batch_size",              64)
    config["local_training"].setdefault("learning_rate",           0.001)
    config["local_training"].setdefault("early_stopping_patience", 10)
    config.setdefault("autoencoder", {})
    config["autoencoder"].setdefault("input_dim",      41)
    config["autoencoder"].setdefault("hidden_dims",    [32, 16])
    config["autoencoder"].setdefault("bottleneck_dim", 8)
    config["autoencoder"].setdefault("activation",     "relu")
    config["autoencoder"].setdefault("dropout",        0.1)
    config.setdefault("aggregation", {})
    config["aggregation"].setdefault("strategy",    "fedavg")
    config["aggregation"].setdefault("fedprox_mu",  0.01)
    config["aggregation"].setdefault("temperature", 1.0)
    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("anomaly_percentile", 95)
    config.setdefault("results", {})
    config["results"].setdefault("log_dir",             "./results/logs")
    config["results"].setdefault("plot_dir",            "./results/plots")
    config["results"].setdefault("checkpoint_dir",      "./results/checkpoints")
    config["results"].setdefault("save_every_n_rounds", 10)
    return config


class FLSimulation:
    def __init__(self, config: dict):
        self.config  = _defaults(deepcopy(config))
        self.device  = self._get_device()
        self.results = {"rounds": [], "final_metrics": {}, "convergence_round": None}
        set_seed(self.config["experiment"]["seed"])
        if self.config["aggregation"]["strategy"] == "scaffold":
            reset_scaffold_state()
        for k in ["log_dir", "checkpoint_dir", "plot_dir"]:
            Path(self.config["results"][k]).mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> str:
        want_cuda = self.config["experiment"].get("device", "cpu") == "cuda"
        if want_cuda and torch.cuda.is_available():
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
            return "cuda"
        logger.info("Using CPU")
        return "cpu"

    def setup(self, partitions: List, X_test: np.ndarray, y_test: np.ndarray):
        self.X_test = X_test.astype(np.float32)
        self.y_test = y_test.astype(int)
        # Always sync input_dim from real data
        self.config["autoencoder"]["input_dim"] = X_test.shape[1]
        self.global_model = build_autoencoder(self.config).to(self.device)
        logger.info("Model params: %d | input_dim: %d",
                    sum(p.numel() for p in self.global_model.parameters()),
                    X_test.shape[1])
        self.clients = [
            FLClient(i, X_c.astype(np.float32), X_c.astype(np.float32),
                     y_c.astype(int), self.config, self.device)
            for i, (X_c, y_c) in enumerate(partitions)
        ]
        logger.info("Clients: %d", len(self.clients))
        strategy = self.config["aggregation"]["strategy"]
        tau      = float(self.config["aggregation"].get("temperature", 1.0))
        lr       = float(self.config["local_training"].get("learning_rate", 0.001))
        steps    = int(self.config["local_training"].get("epochs", 5))
        self.aggregate_fn = get_aggregator(strategy, temperature=tau,
                                           local_lr=lr, local_steps=steps)
        logger.info("Aggregation strategy: %s", strategy)

    def run(self) -> Dict:
        T          = int(self.config["federation"]["num_rounds"])
        strategy   = self.config["aggregation"]["strategy"]
        mu         = float(self.config["aggregation"].get("fedprox_mu", 0.01))
        save_every = int(self.config["results"].get("save_every_n_rounds", 10))
        patience   = int(self.config["local_training"].get("early_stopping_patience", 10))
        best_auc   = 0.0
        no_improve = 0
        t0         = time.time()

        logger.info("="*55)
        logger.info("FL start | rounds=%d clients=%d strategy=%s",
                    T, len(self.clients), strategy)
        logger.info("="*55)

        for rnd in range(1, T + 1):
            rr = self._run_round(rnd, strategy, mu)
            self.results["rounds"].append(rr)

            if rnd % 5 == 0:
                aw     = rr.get("alpha_weights")
                aw_str = "Uniform" if not aw else str(np.round(np.array(aw[:4]), 3)) + "..."
                logger.info("Round %3d/%d | Loss=%.4f | AUC=%.4f | alpha=%s",
                            rnd, T, rr["avg_train_loss"],
                            rr["metrics"]["auc_roc"], aw_str)

            if rnd % save_every == 0:
                self._ckpt(rnd)

            auc = rr["metrics"]["auc_roc"]
            if auc > best_auc + 1e-4:
                best_auc = auc; no_improve = 0
                self._ckpt(rnd, "best")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at round %d", rnd)
                    self.results["convergence_round"] = rnd - patience
                    break

        logger.info("Done in %.1fs", time.time() - t0)
        self.results["final_metrics"]     = self._eval()
        self.results["convergence_round"] = self.results.get("convergence_round") or T
        self._save()
        return self.results

    def _run_round(self, rnd: int, strategy: str, mu: float) -> Dict:
        # Current global weights as flat numpy array
        gw = np.concatenate([
            p.data.cpu().numpy().flatten()
            for p in self.global_model.parameters()
        ]).astype(np.float64)

        client_weights, client_sizes, losses = [], [], []
        for c in self.clients:
            lw, loss = c.train(self.global_model, gw, strategy, mu)
            client_weights.append(np.array(lw, dtype=np.float64))
            client_sizes.append(max(int(c.train_loader.dataset.tensors[0].shape[0]), 1))
            losses.append(float(loss))

        # Aggregate — returns (new_weights, alpha_or_None)
        agg_result  = self.aggregate_fn(gw, client_weights, client_sizes)
        new_weights = np.array(agg_result[0], dtype=np.float64)
        alpha       = agg_result[1] if (len(agg_result) > 1 and agg_result[1] is not None) else None

        # Load new weights back into global model
        offset = 0
        for p in self.global_model.parameters():
            n      = p.numel()
            chunk  = new_weights[offset: offset + n]
            p.data = torch.tensor(chunk.reshape(p.shape),
                                  dtype=p.dtype, device=self.device)
            offset += n

        return {
            "round":          rnd,
            "avg_train_loss": float(np.mean(losses)),
            "metrics":        self._eval(),
            "alpha_weights":  [float(a) for a in alpha] if alpha is not None else None
        }

    def _eval(self) -> Dict:
        self.global_model.eval()
        X_t = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            errors = torch.mean(
                (X_t - self.global_model(X_t)) ** 2, dim=1
            ).cpu().numpy()
        thresh = compute_global_threshold(
            errors, self.y_test,
            percentile=self.config["evaluation"]["anomaly_percentile"])
        return compute_metrics(errors, self.y_test, thresh)

    def _ckpt(self, rnd: int, tag: str = ""):
        name = f"round_{rnd}{'_' + tag if tag else ''}.pt"
        torch.save(self.global_model.state_dict(),
                   Path(self.config["results"]["checkpoint_dir"]) / name)

    def _save(self):
        path = Path(self.config["results"]["log_dir"]) / \
               f"{self.config['experiment']['name']}_results.json"
        with open(path, "w") as f:
            json.dump({
                "final_metrics":     self.results["final_metrics"],
                "convergence_round": self.results["convergence_round"],
                "rounds": [
                    {"round": r["round"],
                     "loss":  r["avg_train_loss"],
                     "auc":   r["metrics"]["auc_roc"],
                     "f1":    r["metrics"]["f1"]}
                    for r in self.results["rounds"]
                ]
            }, f, indent=2, default=str)
        logger.info("Saved → %s", path)
