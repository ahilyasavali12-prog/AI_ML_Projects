"""
models/autoencoder.py
Autoencoder-based anomaly detector.
Trained on normal traffic only — high reconstruction error = anomaly.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class Autoencoder(nn.Module):
    """
    Fully-connected symmetric autoencoder for anomaly detection.

    Architecture (NSL-KDD example):
        Input(41) → 32 → 16 → Bottleneck(8) → 16 → 32 → Output(41)

    Anomaly score = Mean Squared Reconstruction Error (MSRE)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [32, 16],
        bottleneck_dim: int = 8,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()

        self.input_dim      = input_dim
        self.hidden_dims    = hidden_dims
        self.bottleneck_dim = bottleneck_dim

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky": nn.LeakyReLU}[activation]

        # ── Encoder ───────────────────────────────────────────
        enc_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers += [nn.Linear(prev_dim, h_dim), act_fn(), nn.Dropout(dropout)]
            prev_dim = h_dim
        enc_layers += [nn.Linear(prev_dim, bottleneck_dim), act_fn()]
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ───────────────────────────────────────────
        dec_layers = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev_dim, h_dim), act_fn(), nn.Dropout(dropout)]
            prev_dim = h_dim
        dec_layers += [nn.Linear(prev_dim, input_dim)]   # no activation on output
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error."""
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)

    def get_flat_params(self) -> np.ndarray:
        """Return all parameters as a flat numpy array (for FL weight diff)."""
        return np.concatenate([
            p.data.cpu().numpy().flatten()
            for p in self.parameters()
        ])

    def set_flat_params(self, flat_params: np.ndarray):
        """Load flat numpy array back into model parameters."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = torch.tensor(
                flat_params[offset:offset + numel].reshape(p.shape),
                dtype=p.dtype
            )
            offset += numel


def build_autoencoder(config: dict) -> Autoencoder:
    """Construct autoencoder from config dict."""
    return Autoencoder(
        input_dim      = config["autoencoder"]["input_dim"],
        hidden_dims    = config["autoencoder"]["hidden_dims"],
        bottleneck_dim = config["autoencoder"]["bottleneck_dim"],
        dropout        = config["autoencoder"]["dropout"],
        activation     = config["autoencoder"]["activation"]
    )
