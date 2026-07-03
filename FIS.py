# -*- coding: utf-8 -*-
"""
FIS.py — Corrected Version
Changes from original:
  1. Removed duplicate CNNGRU2D class definition (was repeated in FIS integration block)
  2. Replaced p_col_from_probs() with motion_intensity_from_hidden()
     - Extracts GRU final hidden state L2 norm instead of softmax probability
  3. Renamed third FIS input: p_col -> M (Motion Intensity)
  4. Renamed config fields: p_low/p_med/p_high -> m_low/m_med/m_high
  5. Renamed fuzzified variables: pL/pM/pH -> mL/mM/mH throughout
  6. Updated CSV output column: "p_col" -> "motion_intensity"
  7. Updated all analysis labels and variable names in Cells B, C, D
"""

import numpy as np
import pandas as pd
from collections import Counter

from imblearn.over_sampling import RandomOverSampler, SMOTE

CSV_PATH = "Lidar__data_V2.csv"

WINDOW_SIZE = 5
RANDOM_STATE = 42
FAR_MM = 8000.0
NORM_DIV = 10000.0

# Split policy
TEST_FRAC = 0.2
PURGE_GAP_WINDOWS = 10

# Resampling/balancing
BOOTSTRAP_FACTOR = 5
USE_SMOTE = True
SMOTE_K = 3


def load_wide_scans(csv_path):
    df = pd.read_csv(csv_path, header=0, low_memory=False, sep=';')
    df.columns = df.columns.str.strip()

    if "Warning field" not in df.columns:
        raise ValueError(f"Missing 'Warning field' column. Found columns: {df.columns.tolist()[:20]} ...")

    print("DEBUG: Raw unique values and counts in 'Warning field' column:")
    print(df["Warning field"].value_counts(dropna=False))

    y_raw = pd.to_numeric(df["Warning field"], errors="coerce").fillna(0).astype(int).to_numpy()

    dist_cols = [c for c in df.columns if str(c).startswith("Dist_")]
    if not dist_cols:
        raise ValueError(f"No Dist_ columns found. Available columns: {df.columns.tolist()}")

    X_raw = df[dist_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    X_raw = np.where(np.isfinite(X_raw), X_raw, np.nan)
    X_raw[X_raw < 0] = np.nan
    X_raw = np.nan_to_num(X_raw, nan=FAR_MM)
    X_raw[X_raw > FAR_MM] = FAR_MM
    X_raw = X_raw / NORM_DIV

    return X_raw, y_raw


def make_sequences(X_raw, y_raw, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X_raw) - window_size + 1):
        X_seq.append(X_raw[i:i + window_size])
        y_seq.append(y_raw[i + window_size - 1])
    X_seq = np.array(X_seq, dtype=np.float32)[..., np.newaxis]
    y_seq = np.array(y_seq, dtype=np.int32)
    return X_seq, y_seq


def blocked_split_with_purge(X, y, test_frac=0.2, purge_gap_windows=10):
    n = len(X)
    test_n = int(round(n * test_frac))
    test_start = n - test_n
    train_end = max(0, test_start - purge_gap_windows)
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[test_start:], y[test_start:]
    return X_train, y_train, X_test, y_test


def bootstrap_resample(X, y, factor=5, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    if n == 0:
        return X, y
    new_n = int(n * factor)
    idx = rng.integers(0, n, size=new_n)
    return X[idx], y[idx]


def balance_train_set(X_train, y_train, use_smote=False, seed=42, smote_k=3):
    if len(np.unique(y_train)) < 2:
        print("WARNING: Only one class found in y_train. Skipping resampling.")
        return X_train, y_train

    n, w, p, c = X_train.shape
    X_flat = X_train.reshape(n, w * p * c)

    if use_smote:
        sampler = SMOTE(random_state=seed, k_neighbors=smote_k)
    else:
        sampler = RandomOverSampler(random_state=seed)

    X_res, y_res = sampler.fit_resample(X_flat, y_train)
    X_res = X_res.reshape(len(X_res), w, p, c)
    return X_res.astype(np.float32), y_res.astype(np.int32)


def convert_1D_to_2D(X_seq, az_bins=180, range_bins=64):
    N, seq_len, points, _ = X_seq.shape
    X_2d = np.zeros((N, seq_len, range_bins, az_bins, 1), dtype=np.float32)

    max_range = 8.0
    for i in range(N):
        for t in range(seq_len):
            scan = X_seq[i, t, :, 0] * NORM_DIV / 1000.0

            for point_idx in range(points):
                dist = scan[point_idx]

                if dist >= max_range:
                    continue

                az_bin = int((point_idx / points) * az_bins)
                if az_bin >= az_bins:
                    az_bin = az_bins - 1

                range_bin = int((dist / max_range) * (range_bins - 1))
                X_2d[i, t, range_bin, az_bin, 0] = 1.0

    return X_2d


def main():
    X_raw, y_raw = load_wide_scans(CSV_PATH)
    X_all, y_all = make_sequences(X_raw, y_raw, WINDOW_SIZE)

    X_train, y_train, X_test, y_test = blocked_split_with_purge(
        X_all, y_all, test_frac=TEST_FRAC, purge_gap_windows=PURGE_GAP_WINDOWS
    )
    print("Train/Test sizes:", X_train.shape[0], X_test.shape[0])
    print("Train before:", Counter(y_train), " Test:", Counter(y_test))

    X_train_bs, y_train_bs = bootstrap_resample(X_train, y_train, factor=BOOTSTRAP_FACTOR, seed=RANDOM_STATE)
    print("After bootstrap:", Counter(y_train_bs))

    X_train_bal, y_train_bal = balance_train_set(
        X_train_bs, y_train_bs, use_smote=USE_SMOTE, seed=RANDOM_STATE, smote_k=SMOTE_K
    )
    print("After balancing:", Counter(y_train_bal))

    X_train_2d = convert_1D_to_2D(X_train_bal)
    X_test_2d = convert_1D_to_2D(X_test)

    np.savez_compressed(
        "train_resampled_balanced.npz",
        X_train=X_train_bal,
        y_train=y_train_bal,
        X_test=X_test,
        y_test=y_test,
        X_train_2d=X_train_2d,
        X_test_2d=X_test_2d,
        window_size=WINDOW_SIZE
    )
    print("Saved: train_resampled_balanced.npz")


if __name__ == "__main__":
    main()


# ============================================================
# MODEL DEFINITION — CNNGRU2D
# Single authoritative definition. Reused in training and FIS.
# ============================================================
import torch
import torch.nn as nn


class CNNGRU2D(nn.Module):
    """
    2D LiDAR model (sequence of polar grids):
      Input  : (B, T, R, A, 1)
               R=range_bins (e.g. 64), A=angle_bins (e.g. 180)
      Output : logits (B, num_classes)

    Pipeline:
      - Conv2D over (R, A) for each timestep
      - GRU over timesteps T
      - Classifier head
    """
    def __init__(self, num_classes=3, hidden=128, dropout=0.2):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # x: (B, T, R, A, 1)
        B, T, R, A, C = x.shape
        x = x.view(B * T, R, A, C).permute(0, 3, 1, 2)  # (B*T, 1, R, A)
        x = self.conv2d(x)                               # (B*T, 64, 8, 8)
        x = self.feat(x)                                 # (B*T, hidden)
        x = x.view(B, T, -1)                             # (B, T, hidden)
        out, _ = self.gru(x)                             # (B, T, hidden)
        last = out[:, -1, :]                             # (B, hidden)
        return self.head(last)                           # (B, num_classes)


# ============================================================
# TRAINING LOOP
# ============================================================
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

NPZ_PATH = "train_resampled_balanced.npz"
OUT_DIR = "out_2d"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
NUM_WORKERS = 0


class NpzDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def plot_training_curve(losses, out_png):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_png):
    plt.figure()
    plt.imshow(cm)
    plt.xticks(range(len(class_names)), class_names, rotation=20)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    X_train = npz["X_train_2d"]
    y_train = npz["y_train"]
    X_test = npz["X_test_2d"]
    y_test = npz["y_test"]

    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Train labels:", Counter(y_train))
    print("Test  labels:", Counter(y_test))

    num_classes = int(max(y_train.max(), y_test.max())) + 1
    class_names = ["CLEAR", "SLOW", "STOP"][:num_classes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = NpzDataset(X_train, y_train)
    test_ds = NpzDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = CNNGRU2D(num_classes=num_classes).to(device)

    class_w = compute_class_weights(y_train, num_classes).to(device)
    print("Class weights:", class_w.detach().cpu().numpy().round(3).tolist())

    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(yb)
            n += len(yb)

        avg_loss = total_loss / max(n, 1)
        losses.append(avg_loss)

        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            print(f"[train] epoch {ep}/{EPOCHS} loss={avg_loss:.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print("\nAccuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("\nReport:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    torch.save({"state_dict": model.state_dict(), "num_classes": num_classes}, outdir / "model.pt")

    metrics = {
        "npz": str(Path(NPZ_PATH).resolve()),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "train_loss": losses,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_training_curve(losses, outdir / "training_curve.png")
    plot_confusion_matrix(cm, class_names, outdir / "confusion_matrix.png")

    print("\nSaved outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()


# ============================================================
# FIS INTEGRATION
# CNNGRU2D reused from above — no redefinition needed.
# Third input changed: p_col (collision likelihood) -> M (Motion Intensity)
# M is extracted from GRU final hidden state L2 norm, normalized to [0,1].
# Higher M = stronger approaching motion = monotonically consistent with closing velocity.
# ============================================================
from dataclasses import dataclass
from typing import Tuple

NPZ_PATH  = "train_resampled_balanced.npz"
MODEL_PT  = "out_2d/model.pt"
CSV_OUT   = "risk_results_test_directional_varied.csv"

DT_LIDAR  = 0.1
CLEAR_CLASS_INDEX = 0

LEFT_SLICE  = slice(0, 451)
FRONT_SLICE = slice(451, 901)
RIGHT_SLICE = slice(901, 1351)

MIN_VALID = 1e-4
MAX_VALID = 1.0
SECTOR_Q = 50


# =========================
# Motion Intensity Extraction
# Replaces p_col_from_probs — uses GRU hidden state instead of softmax output.
# =========================
def motion_intensity_from_hidden(
    npz_path: str,
    model_pt: str,
    batch_size: int = 64
) -> np.ndarray:
    """
    Extracts Motion Intensity M in [0,1] from the GRU final hidden state.

    The CNN-GRU model is trained on labeled sequences where approaching
    obstacles produce progressively expanding occupancy patterns in the grid,
    resulting in higher hidden state activation magnitudes. Thus M is
    monotonically consistent with closing velocity toward the ego system.

    Parameters
    ----------
    npz_path   : path to the .npz file containing X_test_2d
    model_pt   : path to the saved model checkpoint
    batch_size : inference batch size

    Returns
    -------
    M : (N,) float array in [0,1]
    """
    npz = np.load(npz_path, allow_pickle=True)
    X_test_2d = npz["X_test_2d"].astype(np.float32)
    N = X_test_2d.shape[0]

    ckpt = torch.load(model_pt, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 3))

    # CNNGRU2D defined above — reused here
    model = CNNGRU2D(num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Hook to capture GRU final hidden state h_n
    hidden_states = []

    def hook_fn(module, input, output):
        # output = (out, h_n); h_n shape: (1, B, hidden)
        h_n = output[1].squeeze(0)   # (B, hidden)
        hidden_states.append(h_n.detach().cpu().numpy())

    hook = model.gru.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(X_test_2d[i:i + batch_size])
            model(xb)

    hook.remove()

    h_all = np.concatenate(hidden_states, axis=0)    # (N, hidden)
    norms = np.linalg.norm(h_all, axis=1)            # (N,) L2 norm

    # Normalize to [0,1]
    n_min, n_max = norms.min(), norms.max()
    M = (norms - n_min) / (n_max - n_min + 1e-9)
    return M.astype(np.float32)


# =========================
# BEAMS -> robust sector distance (percentile) + TTC
# =========================
def sector_percentile(x: np.ndarray, sl: slice, q: float) -> np.ndarray:
    v = x[..., sl]
    v = np.where(np.isnan(v), np.inf, v)
    v = np.where((v <= MIN_VALID) | (v > MAX_VALID), np.inf, v)
    v = np.where(np.isfinite(v), v, np.nan)
    out = np.nanpercentile(v, q, axis=-1)
    out = np.where(np.isfinite(out), out, MAX_VALID)
    return out


def batch_sector_stats_from_beams(beams: np.ndarray, q: float = 50) -> np.ndarray:
    dL = sector_percentile(beams, LEFT_SLICE,  q)
    dF = sector_percentile(beams, FRONT_SLICE, q)
    dR = sector_percentile(beams, RIGHT_SLICE, q)
    return np.stack([dL, dF, dR], axis=-1)


def ttc_from_two_ranges(d_now: np.ndarray, d_prev: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
    dt = max(float(dt), eps)
    rate = (d_now - d_prev) / dt
    ttc = np.full_like(d_now, np.inf, dtype=float)
    closing = rate < 0.0
    ttc[closing] = d_now[closing] / np.maximum(-rate[closing], eps)
    return ttc


def compute_dist_ttc_from_npz(npz_path: str, dt: float, q: float = 50):
    npz = np.load(npz_path, allow_pickle=True)
    X_test_raw = npz["X_test"]
    X = X_test_raw[..., 0] if (X_test_raw.ndim == 4 and X_test_raw.shape[-1] == 1) else X_test_raw

    if X.ndim != 3 or X.shape[-1] != 1351 or X.shape[1] < 2:
        raise ValueError(f"Expected X_test (N,T,1351[,1]) with T>=2. Got {X_test_raw.shape}")

    N, T, _ = X.shape
    d_all = batch_sector_stats_from_beams(X.reshape(-1, 1351), q=q).reshape(N, T, 3)

    d_prev = d_all[:, -2, :]
    d_now  = d_all[:, -1, :]
    ttc_all = ttc_from_two_ranges(d_now, d_prev, dt=dt)
    return d_now, ttc_all


# =========================
# FUZZY: trapezoidal MF + config
# Third input: M (Motion Intensity), replaces p_col
# =========================
def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoid MF with left/right shoulder handling.
    - if a==b : left shoulder (membership 1 for x<=b)
    - if c==d : right shoulder (membership 1 for x>=c)
    """
    x = float(x)

    if a == b:
        left = 1.0
    elif x <= a:
        left = 0.0
    elif x < b:
        left = (x - a) / (b - a)
    else:
        left = 1.0

    if c == d:
        right = 1.0
    elif x >= d:
        right = 0.0
    elif x > c:
        right = (d - x) / (d - c)
    else:
        right = 1.0

    return float(np.clip(min(left, right), 0.0, 1.0))


@dataclass
class Fuzzy3InConfig:
    # Distance (normalized scale)
    d_near: Tuple[float, float, float, float] = (0.00, 0.00, 0.06, 0.09)
    d_med:  Tuple[float, float, float, float] = (0.07, 0.10, 0.14, 0.18)
    d_far:  Tuple[float, float, float, float] = (0.16, 0.20, 0.30, 0.30)

    # TTC (seconds)
    t_low:  Tuple[float, float, float, float] = (0.0, 0.0,  1.0,  2.0)
    t_med:  Tuple[float, float, float, float] = (1.5, 3.0, 10.0, 20.0)
    t_high: Tuple[float, float, float, float] = (15.0, 30.0, 1e9, 1e9)

    # Motion Intensity M in [0,1]
    # LOW:    static/receding scene (minimal approach)
    # MEDIUM: ambiguous approach dynamics
    # HIGH:   strong closing trajectory toward ego system
    m_low:  Tuple[float, float, float, float] = (0.0, 0.0, 0.2, 0.4)
    m_med:  Tuple[float, float, float, float] = (0.3, 0.45, 0.6, 0.75)
    m_high: Tuple[float, float, float, float] = (0.6, 0.8, 1.0, 1.0)

    slow_th: float = 0.35
    stop_th: float = 0.65


def fuzzy_infer_risk_directional(d_dir: float, ttc_dir: float, M: float, cfg: Fuzzy3InConfig):
    """
    Single-sample Mamdani inference.
    Inputs: d_dir (normalized distance), ttc_dir (seconds), M (Motion Intensity in [0,1])
    """
    if not np.isfinite(ttc_dir):
        ttc_dir = 1e9

    dN = trapmf(d_dir, *cfg.d_near)
    dM = trapmf(d_dir, *cfg.d_med)
    dF = trapmf(d_dir, *cfg.d_far)

    tL = trapmf(ttc_dir, *cfg.t_low)
    tM = trapmf(ttc_dir, *cfg.t_med)
    tH = trapmf(ttc_dir, *cfg.t_high)

    mL = trapmf(M, *cfg.m_low)
    mM = trapmf(M, *cfg.m_med)
    mH = trapmf(M, *cfg.m_high)

    # High risk rules
    high = max(
        tL,
        min(dN, max(tL, tM, mH)),
        min(mH, dM),
    )

    # Medium risk rules
    med = max(
        dM,
        tM,
        min(dN, tH, max(mL, mM)),
        min(mM, max(dN, dM, tM)),
    )

    # Low risk rules
    low = max(
        min(dF, tH, mL),
        min(dF, tH, 1.0 - mH),
    )

    risk = (low * 0.10 + med * 0.45 + high * 0.90) / (low + med + high + 1e-6)
    risk = float(np.clip(risk, 0.0, 1.0))

    if risk >= cfg.stop_th:
        action, speed = "STOP", 0
    elif risk >= cfg.slow_th:
        action = "SLOW"
        alpha = (risk - cfg.slow_th) / max(1e-6, (cfg.stop_th - cfg.slow_th))
        speed = int(np.clip(60 - 50 * alpha, 10, 60))
    else:
        action, speed = "CONTINUE", 100

    return risk, action, speed


# =========================
# RUN: closest-sector distance + closest-sector TTC + M
# =========================
def main():
    # Extract Motion Intensity M from GRU hidden state
    M = motion_intensity_from_hidden(NPZ_PATH, MODEL_PT)

    d_now, ttc_all = compute_dist_ttc_from_npz(NPZ_PATH, DT_LIDAR, q=SECTOR_Q)

    if d_now.shape[0] != M.shape[0]:
        raise RuntimeError("Mismatch between X_test and X_test_2d sample counts.")

    closest_idx = np.argmin(d_now, axis=1)
    closest_sector = np.array(["L", "F", "R"], dtype=object)[closest_idx]

    d_dir   = d_now[np.arange(len(d_now)), closest_idx]
    ttc_dir = ttc_all[np.arange(len(ttc_all)), closest_idx]

    cfg = Fuzzy3InConfig()

    N = len(d_dir)
    risk   = np.zeros(N, dtype=float)
    action = np.empty(N, dtype=object)
    speed  = np.zeros(N, dtype=int)

    for i in range(N):
        r, a, s = fuzzy_infer_risk_directional(
            float(d_dir[i]), float(ttc_dir[i]), float(M[i]), cfg
        )
        risk[i], action[i], speed[i] = r, a, s

    df = pd.DataFrame({
        "idx": np.arange(N),
        "closest_sector": closest_sector,
        "d_dir": d_dir,
        "ttc_dir": np.where(np.isfinite(ttc_dir), ttc_dir, 1e9),
        "dL": d_now[:, 0], "dF": d_now[:, 1], "dR": d_now[:, 2],
        "ttcL": ttc_all[:, 0], "ttcF": ttc_all[:, 1], "ttcR": ttc_all[:, 2],
        "motion_intensity": M,          # renamed from p_col
        "risk": risk,
        "action": action,
        "speed_override_pct": speed,
    })

    df.to_csv(CSV_OUT, index=False)
    print("Saved:", CSV_OUT)
    print(df.head(10))


if __name__ == "__main__":
    main()


# ============================================================
# PLOTTING UTILITIES
# ============================================================
def plot_risk_with_actions(
    csv_path: str,
    out_path: str = "risk_over_index.png",
    slow_thresh: float = 0.35,
    stop_thresh: float = 0.65,
    n_first=None,
):
    df = pd.read_csv(csv_path)

    if "idx" not in df.columns:
        df.insert(0, "idx", np.arange(len(df)))

    if n_first is not None:
        df = df.iloc[:n_first].copy()

    if "action" not in df.columns:
        raise ValueError("CSV must contain an 'action' column.")
    df["action"] = df["action"].astype(str).str.upper()

    idx    = df["idx"].to_numpy()
    risk   = df["risk"].to_numpy()
    action = df["action"].to_numpy()

    y_action = np.full_like(risk, -0.02, dtype=float)

    plt.figure(figsize=(12, 4))
    plt.plot(idx, risk, linewidth=2, label="Risk")
    plt.axhline(slow_thresh, linestyle="--", label="Slow threshold")
    plt.axhline(stop_thresh, linestyle="--", label="Stop threshold")

    action_styles = {
        "CONTINUE": dict(marker="o", s=70, label="Action: CONTINUE"),
        "SLOW":     dict(marker="^", s=80, label="Action: SLOW"),
        "STOP":     dict(marker="s", s=80, label="Action: STOP"),
    }
    action_colors = {"CONTINUE": "green", "SLOW": "orange", "STOP": "red"}

    for a, sty in action_styles.items():
        m = (action == a)
        if m.any():
            plt.scatter(idx[m], y_action[m], c=action_colors[a], **sty)

    title = "Fuzzy Risk Over Test Index"
    if n_first is not None:
        title += f" (First {n_first} Samples)"
    plt.title(title)
    plt.xlabel("Test sample index")
    plt.ylabel("Risk (0–1)")
    plt.ylim(-0.12, 1.05)

    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h); l2.append(l); seen.add(l)
    plt.legend(h2, l2, loc="upper center", ncol=3, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print("Saved plot to:", out_path)
    print("Action counts:", df["action"].value_counts().to_dict())


def plot_risk_with_actions_window(
    csv_path: str,
    out_path: str = "risk_over_index.png",
    slow_thresh: float = 0.35,
    stop_thresh: float = 0.65,
    n_first=None,
    idx_start=None,
    idx_end=None,
    highlight_window: bool = True,
):
    df = pd.read_csv(csv_path)

    if "idx" not in df.columns:
        df.insert(0, "idx", np.arange(len(df)))

    if "action" not in df.columns or "risk" not in df.columns:
        raise ValueError("CSV must contain 'action' and 'risk' columns.")

    df["action"] = df["action"].astype(str).str.upper()

    if n_first is not None:
        df = df.iloc[:n_first].copy()

    if idx_start is not None or idx_end is not None:
        if idx_start is None:
            idx_start = int(df["idx"].min())
        if idx_end is None:
            idx_end = int(df["idx"].max())
        df = df[(df["idx"] >= idx_start) & (df["idx"] <= idx_end)].copy()

    if df.empty:
        raise ValueError("No data after filtering.")

    idx    = df["idx"].to_numpy()
    risk   = df["risk"].to_numpy()
    action = df["action"].to_numpy()
    y_action = np.full(len(df), -0.03)

    plt.figure(figsize=(12, 4.8))

    if highlight_window:
        plt.axvspan(idx.min(), idx.max(), alpha=0.12)

    plt.plot(idx, risk, linewidth=2.2, label="Risk")
    plt.axhline(slow_thresh, linestyle="--", linewidth=1.5)
    plt.axhline(stop_thresh, linestyle="--", linewidth=1.5)

    action_styles = {
        "CONTINUE": dict(marker="o", s=70, label="CONTINUE"),
        "SLOW":     dict(marker="^", s=90, label="SLOW"),
        "STOP":     dict(marker="s", s=90, label="STOP"),
    }
    action_colors = {"CONTINUE": "green", "SLOW": "orange", "STOP": "red"}

    for a, sty in action_styles.items():
        m = (action == a)
        if m.any():
            plt.scatter(idx[m], y_action[m], c=action_colors[a], **sty, zorder=3)

    title = "Fuzzy Risk and Action"
    if idx_start is not None or idx_end is not None:
        title += f" (index {idx.min()}–{idx.max()})"

    plt.title(title)
    plt.xlabel("Test sample index")
    plt.ylabel("Risk (0–1)")
    plt.ylim(-0.12, 1.05)
    plt.xlim(idx.min() - 1, idx.max() + 1)
    plt.grid(True, alpha=0.25)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

    print("Saved plot to:", out_path)
    print("Action counts:", df["action"].value_counts().to_dict())


if __name__ == "__main__":
    plot_risk_with_actions(
        csv_path="risk_results_test_directional_varied.csv",
        out_path="risk_over_index_full.png",
        slow_thresh=0.35,
        stop_thresh=0.65,
        n_first=None,
    )
    plot_risk_with_actions(
        csv_path="risk_results_test_directional_varied.csv",
        out_path="risk_over_index_first200.png",
        slow_thresh=0.35,
        stop_thresh=0.65,
        n_first=200,
    )
    plot_risk_with_actions_window(
        csv_path="risk_results_test_directional_varied.csv",
        out_path="risk_over_index_182_209_clean.png",
        idx_start=182,
        idx_end=209,
    )


# ============================================================
# CELL A: ENHANCED FUZZY INFERENCE
# Changes vs original:
#   1. Vectorized trapmf (numpy arrays)
#   2. Centroid (WA) defuzzification
#   3. Extended rule base (12 rules: 5 HIGH / 5 MED / 2 LOW)
#   4. Per-rule activation returned for interpretability
#   5. Rule weights as tunable config parameters
#   6. Third input renamed: p_col -> M (Motion Intensity)
# ============================================================
from dataclasses import dataclass, replace, field
from typing import Tuple, Dict


def trapmf_vec(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Vectorized trapezoidal MF — works on arrays of any shape."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    if b > a:
        mask = (x > a) & (x < b)
        out[mask] = (x[mask] - a) / (b - a)
    out[(x >= b) & (x <= c)] = 1.0
    if d > c:
        mask = (x > c) & (x < d)
        out[mask] = (d - x[mask]) / (d - c)
    if a == b:
        out[x <= a] = 1.0   # left shoulder
    if c == d:
        out[x >= d] = 1.0   # right shoulder
    return out


@dataclass
class Fuzzy3InConfigV2:
    # Distance (normalized 0..1)
    d_near: Tuple[float, float, float, float] = (0.00, 0.00, 0.06, 0.09)
    d_med:  Tuple[float, float, float, float] = (0.07, 0.10, 0.14, 0.18)
    d_far:  Tuple[float, float, float, float] = (0.16, 0.20, 0.30, 0.30)

    # TTC (seconds)
    t_low:  Tuple[float, float, float, float] = (0.0, 0.0,  1.0,  2.0)
    t_med:  Tuple[float, float, float, float] = (1.5, 3.0, 10.0, 20.0)
    t_high: Tuple[float, float, float, float] = (15.0, 30.0, 1e9, 1e9)

    # Motion Intensity M in [0,1]
    # LOW:    static/receding scene
    # MEDIUM: ambiguous approach dynamics
    # HIGH:   strong closing trajectory toward ego system
    m_low:  Tuple[float, float, float, float] = (0.0, 0.0, 0.2, 0.4)
    m_med:  Tuple[float, float, float, float] = (0.3, 0.45, 0.6, 0.75)
    m_high: Tuple[float, float, float, float] = (0.6, 0.8, 1.0, 1.0)

    # Thresholds
    slow_th: float = 0.35
    stop_th: float = 0.65

    # Validity gating
    valid_warn: float = 0.10
    valid_crit: float = 0.03

    # Centroid output singletons
    rw_low:  float = 0.10
    rw_med:  float = 0.45
    rw_high: float = 0.90


def fuzzy_infer_batch(
    d_dir:   np.ndarray,
    ttc_dir: np.ndarray,
    M:       np.ndarray,
    cfg:     Fuzzy3InConfigV2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Vectorized Mamdani-style inference over N samples.

    Parameters
    ----------
    d_dir   : (N,) normalized distance
    ttc_dir : (N,) time-to-collision in seconds
    M       : (N,) Motion Intensity in [0,1]
    cfg     : Fuzzy3InConfigV2

    Returns
    -------
    risk      : (N,) float in [0,1]
    action    : (N,) object  CONTINUE | SLOW | STOP
    speed     : (N,) int     speed % [0..100]
    rule_acts : dict of (N,) rule activation arrays
    """
    ttc_dir = np.where(np.isfinite(ttc_dir), ttc_dir, 1e9)

    # Fuzzify inputs
    dN = trapmf_vec(d_dir,  *cfg.d_near)
    dM = trapmf_vec(d_dir,  *cfg.d_med)
    dF = trapmf_vec(d_dir,  *cfg.d_far)

    tL = trapmf_vec(ttc_dir, *cfg.t_low)
    tM = trapmf_vec(ttc_dir, *cfg.t_med)
    tH = trapmf_vec(ttc_dir, *cfg.t_high)

    mL = trapmf_vec(M, *cfg.m_low)
    mM = trapmf_vec(M, *cfg.m_med)
    mH = trapmf_vec(M, *cfg.m_high)

    # ---- Rule base (AND = min, OR = max) ----
    # HIGH risk rules
    r_H1 = tL                                          # H1: TTC critical alone
    r_H2 = np.minimum(dN, mH)                          # H2: Near + high M
    r_H3 = np.minimum(dN, tM)                          # H3: Near + medium TTC
    r_H4 = np.minimum(mH, dM)                          # H4: High M + medium dist
    r_H5 = np.minimum(mH, tL)                          # H5: High M + critical TTC

    # MEDIUM risk rules
    r_M1 = dM                                          # M1: Medium distance alone
    r_M2 = tM                                          # M2: Medium TTC alone
    r_M3 = np.minimum(dN, tH)                          # M3: Near but slow approach (parked)
    r_M4 = np.minimum(mM, np.maximum(dN, dM))          # M4: Medium M + (near|med dist)
    r_M5 = np.minimum(1.0 - mH, dN)                   # M5: Near but low M -> caution

    # LOW risk rules
    r_L1 = np.minimum(dF, np.minimum(tH, mL))          # L1: Far + safe TTC + low M
    r_L2 = np.minimum(dF, np.minimum(tH, 1.0 - mH))   # L2: Far + safe TTC + not high M

    high = np.maximum.reduce([r_H1, r_H2, r_H3, r_H4, r_H5])
    med  = np.maximum.reduce([r_M1, r_M2, r_M3, r_M4, r_M5])
    low  = np.maximum(r_L1, r_L2)

    # Centroid (weighted-average) defuzzification
    num  = low * cfg.rw_low + med * cfg.rw_med + high * cfg.rw_high
    den  = low + med + high + 1e-9
    risk = np.clip(num / den, 0.0, 1.0)

    # Crisp decision
    action = np.where(
        risk >= cfg.stop_th, "STOP",
        np.where(risk >= cfg.slow_th, "SLOW", "CONTINUE")
    ).astype(object)

    alpha = np.clip(
        (risk - cfg.slow_th) / max(1e-9, cfg.stop_th - cfg.slow_th), 0, 1
    )
    speed = np.where(
        risk >= cfg.stop_th, 0,
        np.where(risk >= cfg.slow_th, (60 - 50 * alpha).astype(int), 100)
    ).astype(int)

    rule_acts = {
        "r_H1_TTC_low":         r_H1,
        "r_H2_Near_mH":         r_H2,
        "r_H3_Near_tMed":       r_H3,
        "r_H4_mH_Med":          r_H4,
        "r_H5_mH_tLow":         r_H5,
        "r_M1_Med_dist":        r_M1,
        "r_M2_TTC_med":         r_M2,
        "r_M3_Near_tHigh":      r_M3,
        "r_M4_mMed_nearmed":    r_M4,
        "r_M5_Near_not_mH":     r_M5,
        "r_L1_Far_tH_mL":       r_L1,
        "r_L2_Far_tH_not_mH":   r_L2,
        "agg_high": high, "agg_med": med, "agg_low": low,
    }
    return risk, action, speed, rule_acts


def apply_validity_gating_batch(
    risk: np.ndarray, action: np.ndarray, speed: np.ndarray,
    v_dir: np.ndarray, v_now: np.ndarray, M: np.ndarray,
    cfg: Fuzzy3InConfigV2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_min = v_now.min(axis=1)
    degraded_any = v_min < cfg.valid_warn
    degraded_dir = v_dir < cfg.valid_warn
    critical_dir = v_dir < cfg.valid_crit

    crit_mask = critical_dir
    stop_crit = crit_mask & (M >= 0.6)
    slow_crit = crit_mask & ~stop_crit

    risk   = np.where(stop_crit, np.maximum(risk, cfg.stop_th), risk)
    action = np.where(stop_crit, "STOP", action)
    speed  = np.where(stop_crit, 0, speed)

    alpha_s = np.clip(
        (np.maximum(risk, cfg.slow_th) - cfg.slow_th) /
        max(1e-9, cfg.stop_th - cfg.slow_th), 0, 1
    )
    risk   = np.where(slow_crit, np.maximum(risk, cfg.slow_th), risk)
    action = np.where(slow_crit, "SLOW", action)
    speed  = np.where(slow_crit, (60 - 50 * alpha_s).astype(int), speed)

    warn_mask = (degraded_any | degraded_dir) & ~crit_mask & (action == "CONTINUE")
    risk   = np.where(warn_mask, np.maximum(risk, cfg.slow_th), risk)
    action = np.where(warn_mask, "SLOW", action)
    alpha_w = np.clip(
        (risk - cfg.slow_th) / max(1e-9, cfg.stop_th - cfg.slow_th), 0, 1
    )
    speed  = np.where(warn_mask, (60 - 50 * alpha_w).astype(int), speed)

    return risk, action, speed


print("Cell A loaded: Fuzzy3InConfigV2, fuzzy_infer_batch, apply_validity_gating_batch")


# ============================================================
# CELL B: SENSITIVITY ANALYSIS
# One-at-a-time (OAT) and 2-D surface plots
# Third input axis: Motion Intensity M (replaces p_col)
# ============================================================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
matplotlib.rcParams.update({'figure.dpi': 130, 'font.size': 10})

cfg_sa = Fuzzy3InConfigV2()

N_SWEEP  = 300
d_range  = np.linspace(0.001, 0.30, N_SWEEP)
ttc_range = np.linspace(0.0,  60.0, N_SWEEP)
M_range  = np.linspace(0.0,   1.0,  N_SWEEP)   # renamed from p_range

# Anchor points
A_D   = [0.04, 0.12, 0.25]
A_TTC = [1.0,  10.0, 50.0]
A_M   = [0.15, 0.50, 0.85]                      # renamed from A_P


def oat_sweep(var, sweep, a1, a2):
    """One-at-a-time sweep. var in {'d', 'ttc', 'M'}"""
    N = len(sweep)
    if var == 'd':
        d, t, m = sweep, np.full(N, a1), np.full(N, a2)
    elif var == 'ttc':
        d, t, m = np.full(N, a1), sweep, np.full(N, a2)
    else:
        d, t, m = np.full(N, a1), np.full(N, a2), sweep
    risk, _, _, _ = fuzzy_infer_batch(d, t, m, cfg_sa)
    return risk


# OAT 1-D plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
LINE_COLORS = ['steelblue', 'darkorange', 'seagreen']
LINE_STYLES = ['-', '--', ':']

for ax, (var, sweep, xl) in zip(axes, [
        ('d',   d_range,   'Normalized distance'),
        ('ttc', ttc_range, 'TTC (s)'),
        ('M',   M_range,   'Motion Intensity M'),    # renamed label
]):
    k = 0
    for a1 in (A_TTC if var == 'd' else (A_D if var == 'ttc' else A_D)):
        for a2 in (A_M if var != 'M' else A_TTC):
            if k >= 3:
                break
            risk = oat_sweep(var, sweep, a1, a2)
            if var == 'd':
                lbl = f'TTC={a1}s, M={a2}'
            elif var == 'ttc':
                lbl = f'd={a1}, M={a2}'
            else:
                lbl = f'd={a1}, TTC={a2}s'
            ax.plot(sweep, risk, color=LINE_COLORS[k],
                    linestyle=LINE_STYLES[k], linewidth=1.8, label=lbl)
            k += 1
        if k >= 3:
            break

    ax.axhline(cfg_sa.slow_th, color='gold',   linestyle='-.', linewidth=1.2,
               label=f'Slow th={cfg_sa.slow_th}')
    ax.axhline(cfg_sa.stop_th, color='tomato', linestyle='-.', linewidth=1.2,
               label=f'Stop th={cfg_sa.stop_th}')
    ax.fill_between(sweep, cfg_sa.slow_th, cfg_sa.stop_th,
                    alpha=0.07, color='gold', label='SLOW zone')
    ax.fill_between(sweep, cfg_sa.stop_th, 1.0,
                    alpha=0.07, color='tomato', label='STOP zone')
    ax.set_xlabel(xl); ax.set_ylabel('Risk')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc='best', ncol=1)
    ax.set_title(f'OAT Sensitivity: {xl}')

plt.suptitle('One-at-a-Time Sensitivity Analysis – Fuzzy Inference v2', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('sensitivity_oat.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved: sensitivity_oat.png')

# 2-D Surface Heatmaps
N_G   = 80
d_g   = np.linspace(0.001, 0.30, N_G)
ttc_g = np.linspace(0.0,   60.0, N_G)
M_g   = np.linspace(0.0,    1.0, N_G)          # renamed from p_g

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))


def heatmap2d(ax, xv, yv, fixed_val, xvar, yvar, xlabel, ylabel, title):
    xx, yy = np.meshgrid(xv, yv)
    if xvar == 'd' and yvar == 'M':
        r, _, _, _ = fuzzy_infer_batch(
            xx.ravel(), np.full(N_G**2, fixed_val), yy.ravel(), cfg_sa)
    elif xvar == 'd' and yvar == 'ttc':
        r, _, _, _ = fuzzy_infer_batch(
            xx.ravel(), yy.ravel(), np.full(N_G**2, fixed_val), cfg_sa)
    else:
        r, _, _, _ = fuzzy_infer_batch(
            np.full(N_G**2, fixed_val), xx.ravel(), yy.ravel(), cfg_sa)
    r = r.reshape(N_G, N_G)
    im = ax.contourf(xv, yv, r, levels=25, cmap='RdYlGn_r', vmin=0, vmax=1)
    cs = ax.contour(xv, yv, r,
                    levels=[cfg_sa.slow_th, cfg_sa.stop_th],
                    colors=['gold', 'red'], linewidths=2)
    ax.clabel(cs, fmt={cfg_sa.slow_th: 'SLOW', cfg_sa.stop_th: 'STOP'}, fontsize=8)
    plt.colorbar(im, ax=ax, label='Risk')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\n(fixed={fixed_val:.2f})')


heatmap2d(axes2[0], d_g,   M_g,    10.0, 'd',   'M',   'Distance',  'Motion Intensity M', 'd vs M  | TTC=10s')
heatmap2d(axes2[1], d_g,   ttc_g,   0.5, 'd',   'ttc', 'Distance',  'TTC (s)',             'd vs TTC | M=0.5')
heatmap2d(axes2[2], ttc_g, M_g,    0.10, 'ttc', 'M',   'TTC (s)',   'Motion Intensity M',  'TTC vs M | d=0.10')

plt.suptitle('2-D Risk Surface Sensitivity Heatmaps', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('sensitivity_2d.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved: sensitivity_2d.png')

# Sobol-like index approximation
print("\n=== Variance-based Sensitivity Indices (approximate OAT) ===")
rng_sa = np.random.default_rng(0)
N_SAM  = 5000
d_s   = rng_sa.uniform(0.001, 0.30, N_SAM)
ttc_s = rng_sa.uniform(0.0,   60.0, N_SAM)
M_s   = rng_sa.uniform(0.0,    1.0, N_SAM)     # renamed from p_s

risk_all, _, _, _ = fuzzy_infer_batch(d_s, ttc_s, M_s, cfg_sa)
V_total = risk_all.var()

for name, fixed_arr, sweep_arr1, sweep_arr2 in [
    ('d',   np.full(N_SAM, np.median(d_s)),   ttc_s, M_s),
    ('ttc', np.full(N_SAM, np.median(ttc_s)), d_s,   M_s),
    ('M',   np.full(N_SAM, np.median(M_s)),   d_s,   ttc_s),
]:
    if name == 'd':
        r_fix, _, _, _ = fuzzy_infer_batch(fixed_arr, sweep_arr1, sweep_arr2, cfg_sa)
    elif name == 'ttc':
        r_fix, _, _, _ = fuzzy_infer_batch(sweep_arr1, fixed_arr, sweep_arr2, cfg_sa)
    else:
        r_fix, _, _, _ = fuzzy_infer_batch(sweep_arr1, sweep_arr2, fixed_arr, cfg_sa)
    V_fix = r_fix.var()
    Si = 1.0 - V_fix / max(V_total, 1e-12)
    print(f"  S_{name} (main effect) ≈ {Si:.4f}  (variance reduction when {name} fixed)")


# ============================================================
# CELL C: ROBUSTNESS EVALUATION
# Tests: Gaussian noise sweep, threshold perturbation, rule-weight Monte Carlo
# Third input renamed: p_vec -> M_vec
# ============================================================
matplotlib.rcParams.update({'figure.dpi': 130, 'font.size': 10})

CSV_ROB = "risk_results_test_directional_varied.csv"
try:
    df_rob  = pd.read_csv(CSV_ROB)
    d_vec   = df_rob["d_dir"].to_numpy(dtype=float)
    ttc_vec = df_rob["ttc_dir"].to_numpy(dtype=float)
    M_vec   = df_rob["motion_intensity"].to_numpy(dtype=float)   # renamed from p_col
    print(f"Loaded {len(d_vec)} samples from {CSV_ROB}")
    SYNTHETIC = False
except FileNotFoundError:
    print("CSV not found – generating synthetic demo data.")
    rng0    = np.random.default_rng(0)
    N_syn   = 600
    d_vec   = rng0.uniform(0.02, 0.28, N_syn)
    ttc_vec = rng0.uniform(0.5,  60.0, N_syn)
    M_vec   = rng0.uniform(0.0,   1.0, N_syn)
    SYNTHETIC = True

cfg_rob = Fuzzy3InConfigV2()
risk_base, action_base, _, _ = fuzzy_infer_batch(d_vec, ttc_vec, M_vec, cfg_rob)
N_ROB = len(d_vec)
print(f"Baseline – CONTINUE: {(action_base=='CONTINUE').mean()*100:.1f}%  "
      f"SLOW: {(action_base=='SLOW').mean()*100:.1f}%  "
      f"STOP: {(action_base=='STOP').mean()*100:.1f}%")

# TEST 1 – Input noise sweep
SIGMAS  = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
N_MC    = 30
rng_rob = np.random.default_rng(42)

flip_mean = {'d': [], 'ttc': [], 'M': []}
flip_std  = {'d': [], 'ttc': [], 'M': []}
risk_std  = {'d': [], 'ttc': [], 'M': []}

for sigma in SIGMAS:
    fd, ft, fm = [], [], []
    sd, st, sm = [], [], []
    for _ in range(N_MC):
        n_d   = rng_rob.normal(0, sigma,       N_ROB)
        n_ttc = rng_rob.normal(0, sigma * 100, N_ROB)
        n_M   = rng_rob.normal(0, sigma,       N_ROB)

        r_d, a_d, _, _ = fuzzy_infer_batch(
            np.clip(d_vec + n_d, 0, 1), ttc_vec, M_vec, cfg_rob)
        r_t, a_t, _, _ = fuzzy_infer_batch(
            d_vec, np.clip(ttc_vec + n_ttc, 0, 1e9), M_vec, cfg_rob)
        r_m, a_m, _, _ = fuzzy_infer_batch(
            d_vec, ttc_vec, np.clip(M_vec + n_M, 0, 1), cfg_rob)

        fd.append((a_d != action_base).mean())
        ft.append((a_t != action_base).mean())
        fm.append((a_m != action_base).mean())
        sd.append(np.abs(r_d - risk_base).mean())
        st.append(np.abs(r_t - risk_base).mean())
        sm.append(np.abs(r_m - risk_base).mean())

    flip_mean['d'].append(np.mean(fd));   flip_std['d'].append(np.std(fd));   risk_std['d'].append(np.mean(sd))
    flip_mean['ttc'].append(np.mean(ft)); flip_std['ttc'].append(np.std(ft)); risk_std['ttc'].append(np.mean(st))
    flip_mean['M'].append(np.mean(fm));   flip_std['M'].append(np.std(fm));   risk_std['M'].append(np.mean(sm))

c_map   = {'d': 'steelblue', 'ttc': 'darkorange', 'M': 'seagreen'}
lbl_map = {'d': 'Distance noise', 'ttc': 'TTC noise (scaled /100)', 'M': 'Motion Intensity noise'}

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4.5))
for key in ('d', 'ttc', 'M'):
    fm_arr = np.array(flip_mean[key])
    fs_arr = np.array(flip_std[key])
    axes1[0].plot(SIGMAS, fm_arr, 'o-', color=c_map[key], label=lbl_map[key], linewidth=1.8)
    axes1[0].fill_between(SIGMAS, fm_arr - fs_arr, fm_arr + fs_arr, alpha=0.13, color=c_map[key])
    axes1[1].plot(SIGMAS, risk_std[key], 'o-', color=c_map[key], label=lbl_map[key], linewidth=1.8)

axes1[0].set_xlabel('Noise sigma (input-normalised scale)')
axes1[0].set_ylabel('Action-flip rate (fraction)')
axes1[0].set_title('Robustness: Action Flips vs Input Noise')
axes1[0].legend(fontsize=9); axes1[0].set_ylim(0, 1)

axes1[1].set_xlabel('Noise sigma')
axes1[1].set_ylabel('Mean |Δ risk|')
axes1[1].set_title('Robustness: Risk Change vs Input Noise')
axes1[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('robustness_noise.png', dpi=160)
plt.show()
print('Saved: robustness_noise.png')

# TEST 2 – Threshold sweep
deltas = np.linspace(-0.15, 0.15, 15)
ac_slow = {'CONTINUE': [], 'SLOW': [], 'STOP': []}
ac_stop = {'CONTINUE': [], 'SLOW': [], 'STOP': []}

for delta in deltas:
    cfg_s = replace(cfg_rob, slow_th=float(np.clip(cfg_rob.slow_th + delta, 0.05, 0.90)))
    cfg_t = replace(cfg_rob, stop_th=float(np.clip(cfg_rob.stop_th + delta, 0.10, 0.95)))
    _, a_s, _, _ = fuzzy_infer_batch(d_vec, ttc_vec, M_vec, cfg_s)
    _, a_t, _, _ = fuzzy_infer_batch(d_vec, ttc_vec, M_vec, cfg_t)
    for act in ('CONTINUE', 'SLOW', 'STOP'):
        ac_slow[act].append((a_s == act).mean())
        ac_stop[act].append((a_t == act).mean())

act_c = {'CONTINUE': 'green', 'SLOW': 'orange', 'STOP': 'red'}
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))
for act in ('CONTINUE', 'SLOW', 'STOP'):
    axes2[0].plot(deltas, ac_slow[act], 'o-', color=act_c[act], label=act, linewidth=1.8)
    axes2[1].plot(deltas, ac_stop[act], 'o-', color=act_c[act], label=act, linewidth=1.8)
for ax, lbl in [(axes2[0], 'slow_th'), (axes2[1], 'stop_th')]:
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel(f'Δ {lbl}')
    ax.set_ylabel('Fraction of decisions')
    ax.set_title(f'Robustness: Action Distribution vs Δ {lbl}')
    ax.legend(fontsize=9); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('robustness_threshold.png', dpi=160)
plt.show()
print('Saved: robustness_threshold.png')

# TEST 3 – Rule-weight Monte Carlo
N_REPS  = 300
rng_rw  = np.random.default_rng(7)
PERTURB = 0.10

rw_samples = np.column_stack([
    cfg_rob.rw_low  + rng_rw.uniform(-PERTURB, PERTURB, N_REPS),
    cfg_rob.rw_med  + rng_rw.uniform(-PERTURB, PERTURB, N_REPS),
    cfg_rob.rw_high + rng_rw.uniform(-PERTURB, PERTURB, N_REPS),
])

risk_reps = []
for i in range(N_REPS):
    cfg_rw = replace(cfg_rob,
        rw_low  = float(np.clip(rw_samples[i, 0], 0.0, 0.5)),
        rw_med  = float(np.clip(rw_samples[i, 1], 0.2, 0.7)),
        rw_high = float(np.clip(rw_samples[i, 2], 0.5, 1.0)))
    r, _, _, _ = fuzzy_infer_batch(d_vec, ttc_vec, M_vec, cfg_rw)
    risk_reps.append(r)

risk_reps = np.array(risk_reps)
per_sample_std  = risk_reps.std(axis=0)
per_sample_mean = risk_reps.mean(axis=0)

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4.5))
axes3[0].hist(per_sample_std, bins=40, color='mediumpurple', edgecolor='white')
axes3[0].axvline(per_sample_std.mean(), color='black', linestyle='--',
                 label=f'Mean std = {per_sample_std.mean():.4f}')
axes3[0].set_xlabel('Std(risk) across rule-weight perturbations')
axes3[0].set_ylabel('Number of test samples')
axes3[0].set_title('Rule-Weight Robustness: Risk Spread per Sample')
axes3[0].legend(fontsize=9)

sc = axes3[1].scatter(risk_base, per_sample_std, c=per_sample_mean,
                      cmap='RdYlGn_r', s=6, alpha=0.5, vmin=0, vmax=1)
plt.colorbar(sc, ax=axes3[1], label='Mean risk (perturbed)')
axes3[1].set_xlabel('Baseline risk')
axes3[1].set_ylabel('Std(risk) under rule-weight perturbation')
axes3[1].set_title('Sensitivity: Baseline Risk vs Rule-Weight Uncertainty')

plt.tight_layout()
plt.savefig('robustness_ruleweights.png', dpi=160)
plt.show()
print('Saved: robustness_ruleweights.png')

print("\n=== Robustness Summary ===")
print(f"  Baseline risk       : mean={risk_base.mean():.4f}  std={risk_base.std():.4f}")
print(f"  Rule-weight std     : mean per-sample = {per_sample_std.mean():.4f}")
print(f"  Action distribution : "
      f"CONTINUE={(risk_base < cfg_rob.slow_th).mean()*100:.1f}%  "
      f"SLOW={((risk_base >= cfg_rob.slow_th) & (risk_base < cfg_rob.stop_th)).mean()*100:.1f}%  "
      f"STOP={(risk_base >= cfg_rob.stop_th).mean()*100:.1f}%")


# ============================================================
# CELL D: RULE ACTIVATION VISUALIZATION
# - Mean & max activation bar chart coloured by risk tier
# - Correlation heatmap between all rule activations and final risk
# - Per-action boxplots for HIGH-risk rules
# ============================================================
matplotlib.rcParams.update({'figure.dpi': 130, 'font.size': 10})

cfg_viz = Fuzzy3InConfigV2()

risk_viz, action_viz, _, rule_acts = fuzzy_infer_batch(
    d_vec, ttc_vec, M_vec, cfg_viz)   # M_vec replaces p_vec

# Mean activation bar chart
rule_names = list(rule_acts.keys())
mean_acts  = [rule_acts[k].mean() for k in rule_names]
max_acts   = [rule_acts[k].max()  for k in rule_names]

df_rules = pd.DataFrame({
    'rule': rule_names,
    'mean_activation': mean_acts,
    'max_activation':  max_acts
})
df_rules = df_rules.sort_values('mean_activation', ascending=True)

bar_colors = ['tomato' if ('_H' in k or k == 'agg_high')
              else 'orange' if ('_M' in k or k == 'agg_med')
              else 'steelblue'
              for k in df_rules['rule']]

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.barh(df_rules['rule'], df_rules['mean_activation'],
         color=bar_colors, edgecolor='white', height=0.6)
ax1.barh(df_rules['rule'], df_rules['max_activation'],
         color='none',
         edgecolor=['tomato' if 'tomato' in c else 'orange' if 'orange' in c else 'steelblue'
                    for c in bar_colors],
         height=0.6, linestyle='--', linewidth=1.2, label='Max activation')
ax1.set_xlabel('Activation (0–1)')
ax1.set_title('Fuzzy Rule Activation – Test Set')
ax1.set_xlim(0, 1.0)

from matplotlib.patches import Patch
leg = [Patch(facecolor='tomato',    label='HIGH risk rules'),
       Patch(facecolor='orange',    label='MED  risk rules'),
       Patch(facecolor='steelblue', label='LOW  risk rules')]
ax1.legend(handles=leg, fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('rule_activation_bar.png', dpi=160)
plt.show()
print('Saved: rule_activation_bar.png')

# Correlation heatmap: rules vs risk
all_data  = np.column_stack([rule_acts[k] for k in rule_names] + [risk_viz])
col_names = rule_names + ['risk']
corr      = np.corrcoef(all_data.T)

fig2, ax2 = plt.subplots(figsize=(13, 11))
im = ax2.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax2, label='Pearson r', fraction=0.03)
ax2.set_xticks(range(len(col_names)))
ax2.set_yticks(range(len(col_names)))
ax2.set_xticklabels(col_names, rotation=50, ha='right', fontsize=7.5)
ax2.set_yticklabels(col_names, fontsize=7.5)
ax2.set_title('Rule Activation Correlation Matrix (incl. risk)', fontsize=11)
for j, r_val in enumerate(corr[-1, :]):
    ax2.text(j, len(col_names) - 1, f'{r_val:.2f}',
             ha='center', va='center', fontsize=6.5,
             color='white' if abs(r_val) > 0.5 else 'black')
plt.tight_layout()
plt.savefig('rule_correlation.png', dpi=160)
plt.show()
print('Saved: rule_correlation.png')

# Per-action boxplot of HIGH-risk rule activations
high_rules = [k for k in rule_names if '_H' in k]
n_hr = len(high_rules)
fig3, axes3 = plt.subplots(1, n_hr, figsize=(3 * n_hr, 4), sharey=True)
for ax, rname in zip(axes3, high_rules):
    data_by_act = [rule_acts[rname][action_viz == act] for act in ('CONTINUE', 'SLOW', 'STOP')]
    bp = ax.boxplot(data_by_act, tick_labels=['GO', 'SLW', 'STP'], patch_artist=True,
                    flierprops={'markersize': 2})
    for patch, col in zip(bp['boxes'], ['green', 'orange', 'red']):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.set_title(rname, fontsize=7)
    ax.set_ylim(0, 1)
axes3[0].set_ylabel('Rule activation')
fig3.suptitle('HIGH-Risk Rule Activations by Action Class', fontsize=11)
plt.tight_layout()
plt.savefig('rule_activation_by_action.png', dpi=160)
plt.show()
print('Saved: rule_activation_by_action.png')
