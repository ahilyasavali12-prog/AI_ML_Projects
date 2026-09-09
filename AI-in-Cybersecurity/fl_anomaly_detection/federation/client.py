"""federation/client.py — fully debugged"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from copy import deepcopy


class FLClient:
    def __init__(self, client_id, X_train, X_eval, y_eval, config, device="cpu"):
        self.client_id = client_id
        self.device    = torch.device(device)
        self.config    = config

        # Train ONLY on normal traffic (unsupervised autoencoder)
        y_eval_arr = np.array(y_eval).astype(int)
        X_train_arr = np.array(X_train, dtype=np.float32)
        # Filter: use normal samples for training
        normal_mask = (y_eval_arr[:len(X_train_arr)] == 0)
        X_normal = X_train_arr[normal_mask] if normal_mask.sum() > 0 else X_train_arr

        batch = int(config["local_training"]["batch_size"])
        self.train_loader = self._loader(X_normal, batch, shuffle=True)
        self.eval_loader  = self._loader(np.array(X_eval, dtype=np.float32), batch, shuffle=False)
        self.y_eval       = y_eval_arr

    def _loader(self, X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        t = torch.tensor(X, dtype=torch.float32)
        return DataLoader(TensorDataset(t), batch_size=batch_size,
                          shuffle=shuffle, drop_last=False)

    def train(self, model: nn.Module, global_weights: Optional[np.ndarray] = None,
              strategy: str = "fedavg", mu: float = 0.01) -> Tuple[np.ndarray, float]:
        local_model = deepcopy(model).to(self.device)
        lr      = float(self.config["local_training"]["learning_rate"])
        epochs  = int(self.config["local_training"]["epochs"])
        opt     = torch.optim.Adam(local_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Snapshot global params for FedProx
        global_snap = None
        if strategy in ("fedprox", "scaffold") and global_weights is not None:
            global_snap = {n: p.data.clone()
                           for n, p in local_model.named_parameters()}

        total_loss = 0.0
        local_model.train()

        for _ in range(epochs):
            ep_loss = 0.0
            for (bx,) in self.train_loader:
                bx = bx.to(self.device)
                opt.zero_grad()
                loss = loss_fn(local_model(bx), bx)

                if strategy == "fedprox" and global_snap is not None:
                    prox = sum(
                        torch.sum((p - global_snap[n]) ** 2)
                        for n, p in local_model.named_parameters()
                    )
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                opt.step()
                ep_loss += loss.item()
            total_loss += ep_loss / max(len(self.train_loader), 1)

        avg_loss = total_loss / max(epochs, 1)
        weights  = np.concatenate([
            p.data.cpu().numpy().flatten()
            for p in local_model.parameters()
        ])
        return weights.astype(np.float64), float(avg_loss)
