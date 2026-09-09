# FL Anomaly Detection — Adaptive Aggregation for Network Intrusion Detection

## Research Overview
Comparing **FedAvg vs. Adaptive Aggregation** under **IID and Non-IID** conditions
for network anomaly detection using autoencoder-based FL clients.

---

## Project Structure
```
fl_anomaly_detection/
│
├── data/                          # Data loading, partitioning, preprocessing
│   ├── __init__.py
│   ├── download.py                # Download NSL-KDD and UNSW-NB15
│   ├── preprocess.py              # Feature scaling, encoding
│   └── partition.py               # IID and Non-IID Dirichlet partitioning
│
├── models/                        # Local client models
│   ├── __init__.py
│   └── autoencoder.py             # Autoencoder anomaly detector
│
├── federation/                    # FL training loop
│   ├── __init__.py
│   ├── client.py                  # Flower FL client logic
│   ├── server.py                  # Flower FL server logic
│   └── simulation.py              # Run full FL simulation
│
├── aggregation/                   # Aggregation strategies
│   ├── __init__.py
│   ├── fedavg.py                  # Standard FedAvg
│   ├── fedprox.py                 # FedProx
│   ├── scaffold.py                # SCAFFOLD
│   └── adaptive.py                # Proposed gradient similarity method
│
├── evaluation/                    # Metrics and plotting
│   ├── __init__.py
│   ├── metrics.py                 # AUC-ROC, F1, DR, FAR
│   └── plots.py                   # All 6 research graphs
│
├── utils/                         # Shared utilities
│   ├── __init__.py
│   ├── logger.py                  # Logging setup
│   └── seed.py                    # Random seed control
│
├── configs/                       # Experiment configurations
│   ├── base_config.yaml           # Default hyperparameters
│   ├── phase1_config.yaml         # Week 2 — minimal experiment
│   ├── phase2_config.yaml         # Week 3 — core experiment
│   └── phase3_config.yaml         # Week 4 — full experiment
│
├── results/                       # Auto-generated outputs
│   ├── logs/                      # Training logs per experiment
│   ├── plots/                     # Saved figures
│   └── checkpoints/               # Saved model weights
│
├── notebooks/
│   └── FL_Anomaly_Detection.ipynb # Main Colab notebook (run everything here)
│
├── main.py                        # CLI entry point
├── requirements.txt               # All dependencies
└── README.md
```

---

## Quick Start (Google Colab)

```python
# 1. Clone or upload project
# 2. Open notebooks/FL_Anomaly_Detection.ipynb
# 3. Run Cell 1 — installs dependencies
# 4. Run Cell 2 — downloads datasets
# 5. Run Cell 3 — starts Phase 1 experiment
```

---

## Experiment Phases

| Phase | Week | Dataset | Clients | Conditions | Models |
|-------|------|---------|---------|------------|--------|
| 1 | 2 | NSL-KDD | 5 | IID + Non-IID α=0.1 | FedAvg vs Adaptive |
| 2 | 3 | NSL-KDD | 10 | IID + α=0.1,0.5,1.0 | + FedProx + SCAFFOLD |
| 3 | 4 | + UNSW-NB15 | 10 | All | + PerFedHypID |

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Clients (N) | 10 |
| Rounds (T) | 100 |
| Local epochs (E) | 5 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Bottleneck dim | 8 |
| Temperature τ | 1.0 |
| Dirichlet α | 0.1 / 0.5 / 1.0 |
