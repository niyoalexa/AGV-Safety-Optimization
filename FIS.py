# ============================================================
# FUZZY INFERENCE
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
        out[x < a] = 1.0   # left shoulder
    if c == d:
        out[x > d] = 1.0   # right shoulder
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
# Third input axis: Motion Intensity M 
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
# VALUATION
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
