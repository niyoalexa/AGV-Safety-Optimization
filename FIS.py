import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

CSV_PATH = "Lidar_data_V2.csv"
WEIGHTS_PATH = "motion_regression_model.weights.h5"
CSV_OUT = "risk_results.csv"

WINDOW_SIZE = 50
STEP_SIZE = 22
DT = 0.1
MAX_RANGE_MM = 3500.0
DEADZONE_MM = 500.0

LEFT, FRONT, RIGHT = slice(0,451), slice(451,901), slice(901,1351)

SLOW_TH, STOP_TH, EPS = 0.35, 0.65, 1e-9

# Distance
D_NEAR = (0.00, 0.00, 0.06, 0.09)
D_MED  = (0.07, 0.10, 0.14, 0.18)
D_FAR  = (0.16, 0.20, 0.30, 0.30)

# TTC
T_LOW  = (0.0, 0.0, 1.0, 2.0)
T_MED  = (1.5, 3.0, 10.0, 20.0)
T_HIGH = (15.0, 30.0, 1e9, 1e9)

# CNN-GRU motion
M_LOW  = (0.00, 0.00, 0.20, 0.40)
M_MED  = (0.30, 0.45, 0.60, 0.75)
M_HIGH = (0.60, 0.80, 1.00, 1.00)

R_LOW, R_MED, R_HIGH = 0.10, 0.45, 0.90

df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()

dist_cols = sorted([c for c in df.columns if c.startswith("Dist_")],
                   key=lambda c: int(c.split("_")[1]))

if len(dist_cols) != 1351:
    raise ValueError(f"Expected 1351 LiDAR beams, found {len(dist_cols)}")

X_raw = df[dist_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
X_raw = np.where(np.isfinite(X_raw) & (X_raw > 0), X_raw, MAX_RANGE_MM)
X_raw = np.clip(X_raw, 0, MAX_RANGE_MM).astype(np.float32)

print("Raw LiDAR shape:", X_raw.shape)

starts = np.arange(0, len(X_raw) - WINDOW_SIZE + 1, STEP_SIZE)

X_seq_raw = np.array([X_raw[i:i+WINDOW_SIZE] for i in starts], dtype=np.float32)
X_model = X_seq_raw / MAX_RANGE_MM
X_model = X_model[..., np.newaxis]

print("Sequence shape:", X_model.shape)

def build_fuzzy_extractor(window_size=50, num_angles=1351):
    tf.keras.backend.clear_session()

    inputs = layers.Input(shape=(window_size, num_angles, 1))
    x = layers.GaussianNoise(0.001)(inputs)

    x = layers.Conv2D(64, (3,4), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1,4))(x)

    x = layers.Conv2D(128, (3,4), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1,4))(x)

    shape = x.shape
    x = layers.Reshape((window_size, shape[2] * shape[3]))(x)

    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    motion_out = layers.Dense(1, activation="sigmoid", name="motion")(x)

    return models.Model(inputs=inputs, outputs=motion_out)

motion_model = build_fuzzy_extractor(WINDOW_SIZE, len(dist_cols))
motion_model.load_weights(WEIGHTS_PATH)

print("Motion model weights loaded:", WEIGHTS_PATH)

motion = motion_model.predict(X_model, batch_size=16, verbose=0).flatten()
motion = np.clip(motion, 0.0, 1.0)

print(f"Predicted motion shape: {motion.shape}")
print(f"Predicted motion range: {motion.min():.4f} - {motion.max():.4f}")

def sector_distance(scan, sector):
    clean = np.where(scan[:, sector] < DEADZONE_MM, MAX_RANGE_MM, scan[:, sector])
    return np.percentile(clean, 50, axis=1) / MAX_RANGE_MM

def directional_distance(scan):
    return np.column_stack([sector_distance(scan, LEFT),
                            sector_distance(scan, FRONT),
                            sector_distance(scan, RIGHT)])

previous = X_seq_raw[:, -2, :]
current  = X_seq_raw[:, -1, :]

d_prev = directional_distance(previous)
d_now  = directional_distance(current)

closing_speed = (d_prev - d_now) / DT

ttc = np.full(d_now.shape, np.inf, dtype=np.float32)
approaching = closing_speed > 1e-6
ttc[approaching] = d_now[approaching] / closing_speed[approaching]

sector_idx = np.argmin(d_now, axis=1)
rows = np.arange(len(d_now))
sector_names = np.array(["LEFT", "FRONT", "RIGHT"])

sector = sector_names[sector_idx]
distance = d_now[rows, sector_idx]
TTC = ttc[rows, sector_idx]

def trapmf(x, a, b, c, d):
    if x <= a: return 1.0 if a == b else 0.0
    if x < b: return (x-a)/(b-a)
    if x <= c: return 1.0
    if x < d: return (d-x)/(d-c)
    return 1.0 if c == d else 0.0

def fuzzy_risk(distance, ttc_value, M):
    if not np.isfinite(ttc_value): ttc_value = 1e9

    # Fuzzification
    dN, dM, dF = trapmf(distance,*D_NEAR), trapmf(distance,*D_MED), trapmf(distance,*D_FAR)
    tL, tM, tH = trapmf(ttc_value,*T_LOW), trapmf(ttc_value,*T_MED), trapmf(ttc_value,*T_HIGH)
    mL, mM, mH = trapmf(M,*M_LOW), trapmf(M,*M_MED), trapmf(M,*M_HIGH)

    r_H1 = tL                          # H1: TTC critical alone
    r_H2 = min(dN,mH)                  # H2: Near + high motion
    r_H3 = min(dN,tM)                  # H3: Near + medium TTC
    r_H4 = min(mH,dM)                  # H4: High motion + medium distance
    r_H5 = min(mH,tL)                  # H5: High motion + critical TTC

    r_M1 = dM                          # M1: Medium distance alone
    r_M2 = tM                          # M2: Medium TTC alone
    r_M3 = min(dN,tH)                  # M3: Near + high TTC
    r_M4 = min(mM,max(dN,dM))          # M4: Medium motion + near/medium distance
    r_M5 = min(1.0-mH,dN)              # M5: Near but not high motion

    r_L1 = min(dF,min(tH,mL))          # L1: Far + safe TTC + low motion
    r_L2 = min(dF,min(tH,1.0-mH))      # L2: Far + safe TTC + not high motion

   
    mu_high = max(r_H1,r_H2,r_H3,r_H4,r_H5)
    mu_med  = max(r_M1,r_M2,r_M3,r_M4,r_M5)
    mu_low  = max(r_L1,r_L2)

    risk = (mu_low*R_LOW + mu_med*R_MED + mu_high*R_HIGH) / (mu_low + mu_med + mu_high + EPS)

    return float(np.clip(risk,0,1))


risk = np.array([fuzzy_risk(distance[i],TTC[i],motion[i]) for i in range(len(distance))])


action = np.where(risk >= STOP_TH,"STOP",np.where(risk >= SLOW_TH,"SLOW","CONTINUE"))

alpha = np.clip((risk-SLOW_TH)/max(EPS,STOP_TH-SLOW_TH),0,1)
speed = np.where(risk >= STOP_TH,0,np.where(risk >= SLOW_TH,(60-50*alpha).astype(int),100)).astype(int)


results = pd.DataFrame({
    "idx": rows, "sector": sector, "distance": distance,
    "TTC": np.where(np.isfinite(TTC),TTC,1e9),
    "motion_pred": motion, "risk": risk, "action": action, "speed_percent": speed
})

results.to_csv(CSV_OUT,index=False)

print("\n--- FUZZY INPUT / OUTPUT CHECK ---")
print(results[["distance","TTC","motion_pred","risk","action","speed_percent"]].head(10))

print("\n Action distribution:")
print(results["action"].value_counts())

print("\n Risk:")
print(results["risk"].describe())


START, END = 1,209
p = results[results["idx"].between(START,END)].copy()

if p.empty: raise ValueError(f"No samples between {START} and {END}")

x,y,a = p["idx"].to_numpy(),p["risk"].to_numpy(),p["action"].to_numpy()

plt.figure(figsize=(12,4.8))
plt.plot(x,y,linewidth=2,label="Risk")
plt.axhline(SLOW_TH,linestyle="--")
plt.axhline(STOP_TH,linestyle="--")

for name,marker,color in [("CONTINUE","o","green"),("SLOW","^","orange"),("STOP","s","red")]:
    mask = a == name
    if np.any(mask): plt.scatter(x[mask],np.full(mask.sum(),-0.03),marker=marker,s=70,color=color,label=name)

plt.xlabel("Test sample index"); plt.ylabel("Risk (0–1)")
plt.title(f"Fuzzy Risk and Action (index {START}–{END})")
plt.ylim(-0.10,1.05); plt.grid(alpha=0.25); plt.legend(loc="upper center",ncol=5)
plt.tight_layout(); plt.show()
