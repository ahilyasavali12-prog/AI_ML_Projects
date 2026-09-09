"""
data/preprocess.py
Feature engineering, encoding, and normalisation
for NSL-KDD and UNSW-NB15.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

RAW_DIR  = Path("./data/raw")
PROC_DIR = Path("./data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── NSL-KDD attack label mapping ──────────────────────────────────────────────
NSL_KDD_ATTACK_MAP = {
    "normal": 0,
    # DoS
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1, "teardrop": 1,
    "apache2": 1, "udpstorm": 1, "processtable": 1, "worm": 1,
    # Probe
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    # R2L
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "sendmail": 3, "named": 3,
    "snmpgetattack": 3, "snmpguess": 3, "xlock": 3, "xsnoop": 3, "httptunnel": 3,
    # U2R
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "ps": 4, "sqlattack": 4, "xterm": 4,
}

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
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


def preprocess_nsl_kdd(save=True):
    """
    Load, encode, scale NSL-KDD.
    Returns:
        X_train, X_test : np.ndarray  (features, normalised)
        y_train, y_test : np.ndarray  (0=normal, 1-4=attack class)
        y_binary_train, y_binary_test : np.ndarray (0=normal, 1=any attack)
    """
    nsl_dir = RAW_DIR / "nsl_kdd"

    train_df = pd.read_csv(nsl_dir / "KDDTrain+.txt", header=None, names=NSL_KDD_COLUMNS)
    test_df  = pd.read_csv(nsl_dir / "KDDTest+.txt",  header=None, names=NSL_KDD_COLUMNS)

    # Drop difficulty column
    train_df.drop(columns=["difficulty"], inplace=True)
    test_df.drop(columns=["difficulty"], inplace=True)

    # Encode categorical features
    encoders = {}
    for col in NSL_KDD_CATEGORICAL:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        encoders[col] = le

    # Map attack labels
    train_df["label"] = train_df["label"].str.strip(".").map(
        lambda x: NSL_KDD_ATTACK_MAP.get(x, 1))
    test_df["label"] = test_df["label"].str.strip(".").map(
        lambda x: NSL_KDD_ATTACK_MAP.get(x, 1))

    # Split features / labels
    X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
    X_test  = test_df.drop(columns=["label"]).values.astype(np.float32)
    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # Binary labels (0=normal, 1=attack)
    y_bin_train = (y_train > 0).astype(int)
    y_bin_test  = (y_test  > 0).astype(int)

    # Normalise using train statistics only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if save:
        out = PROC_DIR / "nsl_kdd"
        out.mkdir(exist_ok=True)
        np.save(out / "X_train.npy", X_train)
        np.save(out / "X_test.npy",  X_test)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "y_test.npy",  y_test)
        np.save(out / "y_bin_train.npy", y_bin_train)
        np.save(out / "y_bin_test.npy",  y_bin_test)
        with open(out / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"[NSL-KDD] Preprocessed. X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, y_train, y_test, y_bin_train, y_bin_test


def preprocess_unsw_nb15(save=True):
    """
    Load, encode, scale UNSW-NB15.
    Returns same tuple as preprocess_nsl_kdd.
    """
    unsw_dir = RAW_DIR / "unsw_nb15"

    train_df = pd.read_csv(unsw_dir / "UNSW_NB15_train.csv")
    test_df  = pd.read_csv(unsw_dir / "UNSW_NB15_test.csv")

    # Drop non-feature columns
    drop_cols = ["id", "attack_cat"]
    train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], inplace=True)
    test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], inplace=True)

    label_col = "label"
    y_train = train_df[label_col].values
    y_test  = test_df[label_col].values
    train_df.drop(columns=[label_col], inplace=True)
    test_df.drop(columns=[label_col], inplace=True)

    # Encode remaining categoricals
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)
    y_bin_train = (y_train > 0).astype(int)
    y_bin_test  = (y_test  > 0).astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if save:
        out = PROC_DIR / "unsw_nb15"
        out.mkdir(exist_ok=True)
        np.save(out / "X_train.npy", X_train)
        np.save(out / "X_test.npy",  X_test)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "y_test.npy",  y_test)
        np.save(out / "y_bin_train.npy", y_bin_train)
        np.save(out / "y_bin_test.npy",  y_bin_test)
        with open(out / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"[UNSW-NB15] Preprocessed. X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, y_train, y_test, y_bin_train, y_bin_test


def load_processed(dataset="nsl_kdd"):
    """Load already-preprocessed arrays from disk."""
    out = PROC_DIR / dataset
    return (
        np.load(out / "X_train.npy"),
        np.load(out / "X_test.npy"),
        np.load(out / "y_train.npy"),
        np.load(out / "y_test.npy"),
        np.load(out / "y_bin_train.npy"),
        np.load(out / "y_bin_test.npy"),
    )


if __name__ == "__main__":
    preprocess_nsl_kdd()
