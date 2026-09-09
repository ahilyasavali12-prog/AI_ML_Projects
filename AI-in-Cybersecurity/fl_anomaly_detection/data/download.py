"""
data/download.py
Downloads NSL-KDD and UNSW-NB15 datasets.
Run once before any experiments.
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path

RAW_DIR = Path("./data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── NSL-KDD ───────────────────────────────────────────────────────────────────
NSL_KDD_URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
}

NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

def download_nsl_kdd():
    """Download NSL-KDD train and test sets."""
    nsl_dir = RAW_DIR / "nsl_kdd"
    nsl_dir.mkdir(exist_ok=True)

    for split, url in NSL_KDD_URLS.items():
        out_path = nsl_dir / f"KDD{split.capitalize()}+.txt"
        if out_path.exists():
            print(f"[NSL-KDD] {split} already exists, skipping.")
            continue
        print(f"[NSL-KDD] Downloading {split}...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print(f"[NSL-KDD] Saved to {out_path}")

    # Quick verify
    train_df = pd.read_csv(nsl_dir / "KDDTrain+.txt", header=None, names=NSL_KDD_COLUMNS)
    test_df  = pd.read_csv(nsl_dir / "KDDTest+.txt",  header=None, names=NSL_KDD_COLUMNS)
    print(f"[NSL-KDD] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df


# ── UNSW-NB15 ─────────────────────────────────────────────────────────────────
UNSW_URLS = {
    "train": "https://raw.githubusercontent.com/UNSW-NB15/UNSW-NB15-CSV/main/UNSW_NB15_training-set.csv",
    "test":  "https://raw.githubusercontent.com/UNSW-NB15/UNSW-NB15-CSV/main/UNSW_NB15_testing-set.csv"
}

def download_unsw_nb15():
    """
    Download UNSW-NB15. If direct URL fails (dataset gating),
    prints manual download instructions from the official source.
    """
    unsw_dir = RAW_DIR / "unsw_nb15"
    unsw_dir.mkdir(exist_ok=True)

    for split, url in UNSW_URLS.items():
        out_path = unsw_dir / f"UNSW_NB15_{split}.csv"
        if out_path.exists():
            print(f"[UNSW-NB15] {split} already exists, skipping.")
            continue
        print(f"[UNSW-NB15] Downloading {split}...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            print(f"[UNSW-NB15] Saved to {out_path}")
        except Exception as e:
            print(f"[UNSW-NB15] Auto-download failed: {e}")
            print("[UNSW-NB15] Manual download instructions:")
            print("  1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
            print("  2. Download UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv")
            print(f"  3. Place both files in: {unsw_dir}/")

    csv_files = list(unsw_dir.glob("*.csv"))
    if csv_files:
        train_df = pd.read_csv(unsw_dir / "UNSW_NB15_train.csv")
        print(f"[UNSW-NB15] Train: {len(train_df):,} rows")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Downloading datasets...")
    print("=" * 50)
    download_nsl_kdd()
    download_unsw_nb15()
    print("\nAll downloads complete. Data saved to ./data/raw/")
