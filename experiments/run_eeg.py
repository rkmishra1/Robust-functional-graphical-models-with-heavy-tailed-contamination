"""
EEG application: UCI "EEG Eye State" data (14 channels, 10-20 system,
128 Hz, single subject, 117 s; known blink artifacts in frontal channels).

Protocol (paper Section 'Planned data applications'):
  (i)   reference fit of the three estimators on the unperturbed epochs;
  (ii)  inject synthetic blink artifacts into a random eps fraction of
        epochs (frontal channels: AF3, F7, F3, F4, F8, AF4);
  (iii) network drift = Jaccard distance between the perturbed fit and the
        reference fit, per method; plus edge-count movement.

Run:  python3 experiments/run_eeg.py
"""
import os, sys, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fgm import (cosine_basis, curves_to_coeffs, block_glasso, fglasso_t_em,
                 fglasso_altt_em, _altt_weighted_cov, edge_set_from_omega,
                 jaccard_dist)

CFG = dict(K=6, T=64, nu=7.0,
           eps_grid=[0.0, 0.05, 0.10, 0.20], R=6,
           lam_grid=np.geomspace(0.2, 10.0, 9),
           frontal=[0, 1, 2, 11, 12, 13],     # AF3, F7, F3, F4, F8, AF4
           seed=20261101)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
CH_NAMES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
            "FC6", "F4", "F8", "AF4"]


def load_eeg():
    """Parse the UCI ARFF; return X raw (n_samples, 14)."""
    path = os.path.join(ROOT, "data", "EEG Eye State.arff")
    if not os.path.exists(path):
        raise FileNotFoundError("run: curl UCI dataset 264 into data/ first")
    rows, in_data = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if in_data:
                parts = [float(v) for v in line.split(",")]
                rows.append(parts[:-1])          # drop eye-state label
    return np.asarray(rows)


def make_epochs(X_raw, T):
    """Non-overlapping epochs of length T; robust per-(channel,coef) scaling
    happens in coefficient space later. Returns (n_epochs, 14, T)."""
    n_ep = X_raw.shape[0] // T
    return X_raw[:n_ep * T].reshape(n_ep, 14, T).transpose(0, 1, 2)


def inject_blinks(A3, eps, rng, K, frontal, amp_range=(5.0, 9.0)):
    """Blink-like bumps in frontal channels of eps*n epochs, in CURVE space.

    A3: (n, p, K) coefficients; we add a smooth bump to the curves and
    re-project.  Bumps: width 0.12-0.22, onset 0.2-0.5, frontal channels.
    """
    n, p, _ = A3.shape
    T = CFG["T"]
    t = (np.arange(T) + 0.5) / T
    m = max(1, int(round(eps * n))) if eps > 0 else 0
    idx = rng.choice(n, size=m, replace=False)
    Phi = cosine_basis(T, K)
    A_c = A3.copy()
    for i in idx:
        for j in frontal:
            amp = rng.uniform(*amp_range)
            t0 = rng.uniform(0.2, 0.5)
            w = rng.uniform(0.12, 0.22)
            bump = amp * np.exp(-0.5 * ((t - t0) / w) ** 2)
            # add bump in units of the channel's robust curve sd
            A_c[i, j] += (Phi.T @ bump) / T * 1.0
    return A_c, idx


def fit(method, A, lam, K, nu):
    if method == "fglasso":
        return block_glasso(A.T @ A / A.shape[0], lam, K)
    if method == "t_em":
        return fglasso_t_em(A, lam, K, nu=nu, max_iter=15)["Omega"]
    if method == "altt_em":
        return fglasso_altt_em(A, lam, K, nu=nu, max_iter=15)["Omega"]


def main():
    t0 = time.time()
    cfg = CFG
    K, T, nu = cfg["K"], cfg["T"], cfg["nu"]
    rng = np.random.default_rng(cfg["seed"])

    X_raw = load_eeg()
    print(f"EEG raw: {X_raw.shape[0]} samples x {X_raw.shape[1]} channels", flush=True)
    Ep = make_epochs(X_raw, T)                     # (n_ep, 14, T)
    n = Ep.shape[0]
    print(f"epochs: {n} x T={T}")

    # Common average reference (removes shared drift/compound activity),
    # then robust standardization per channel (median/MAD in curve space)
    Ep = Ep - Ep.mean(axis=1, keepdims=True)
    med = np.median(Ep, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(Ep - med), axis=0), 1e-6)
    Ep_s = (Ep - med) / mad

    Phi = cosine_basis(T, K)
    A = curves_to_coeffs(Ep_s, Phi)                # (n, pK)
    # Coefficient-wise robust standardization.  Sparsity-invariant: a
    # positive diagonal rescaling of the coefficients rescales Omega as
    # D Omega D and preserves the graph, but puts the penalty grid in the
    # meaningful range for partial correlations.
    a_med = np.median(A, axis=0)
    a_mad = np.maximum(1.4826 * np.median(np.abs(A - a_med), axis=0), 1e-8)
    A = (A - a_med) / a_mad
    # Winsorize extreme transients (the raw recording contains genuine
    # artifacts orders of magnitude beyond the bulk; per the paper's
    # protocol the reference fit is taken on artifact-rejected data).
    A = np.clip(A, -8.0, 8.0)
    print(f"  winsorized coefficients; frac clipped = "
          f"{np.mean(np.abs(A) >= 7.999):.4f}", flush=True)

    # Penalties calibrated to a common reference graph density.  Each
    # method's scatter lives at its own scale (the weighted covariances
    # of the robust estimators are not scale-comparable to the raw
    # sample covariance), so the grid is rescaled per method by the
    # median diagonal of its reference scatter.
    TARGET_EDGES = 45
    lams = {}
    for method in ["fglasso", "t_em", "altt_em"]:
        Om0 = fit(method, A, 1e-3 * 1.0, K, nu) if False else None
        # reference scatter scale for this method
        if method == "fglasso":
            S_ref = A.T @ A / A.shape[0]
        elif method == "t_em":
            r = fglasso_t_em(A, 1e-4, K, nu=nu, max_iter=15)
            D = A - r["mu"]
            S_ref = (D * r["weights"][:, None]).T @ D / r["weights"].sum()
        else:
            r = fglasso_altt_em(A, 1e-4, K, nu=nu, max_iter=15)
            Cc = (A.reshape(n, 14, K) - r["mu"])
            S_ref = _altt_weighted_cov(Cc, r["weights"], K)
        scale = float(np.median(np.diag(S_ref)))
        grid = np.geomspace(0.003, 6.0, 13) * scale
        best_lam, best_gap = None, 1e9
        for lam in grid:
            E = edge_set_from_omega(fit(method, A, float(lam), K, nu), K)
            gap = abs(len(E) - TARGET_EDGES)
            if gap < best_gap:
                best_gap, best_lam = gap, float(lam)
            if len(E) < TARGET_EDGES * 0.4:
                break
        lams[method] = best_lam
        print(f"  lambda[{method}] = {best_lam:.3f} (scale {scale:.2f}, "
              f"t={time.time()-t0:.0f}s)", flush=True)

    # reference fits on unperturbed epochs
    ref = {}
    for method in ["fglasso", "t_em", "altt_em"]:
        ref[method] = edge_set_from_omega(fit(method, A, lams[method], K, nu), K)
        print(f"  reference graph [{method}]: {len(ref[method])} edges "
              f"(t={time.time()-t0:.0f}s)", flush=True)

    # drift under injected blinks
    rows = []
    for rep in range(cfg["R"]):
        for i_eps, eps in enumerate(cfg["eps_grid"]):
            rng_eps = np.random.default_rng(cfg["seed"] + 500 + 100 * rep + i_eps)
            A3 = A.reshape(n, 14, K)
            if eps == 0.0:
                for method in ["fglasso", "t_em", "altt_em"]:
                    rows.append(dict(rep=rep, eps=eps, method=method,
                                     drift=0.0, n_edges=len(ref[method]),
                                     n_edges_ref=len(ref[method])))
                continue
            A_pert, _ = inject_blinks(A3, eps, rng_eps, K, cfg["frontal"])
            A_pert = A_pert.reshape(n, 14 * K)
            for method in ["fglasso", "t_em", "altt_em"]:
                E = edge_set_from_omega(fit(method, A_pert, lams[method], K, nu), K)
                rows.append(dict(rep=rep, eps=eps, method=method,
                                 drift=jaccard_dist(ref[method], E),
                                 n_edges=len(E),
                                 n_edges_ref=len(ref[method])))
        print(f"  rep {rep+1}/{cfg['R']} (t={time.time()-t0:.0f}s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "eeg_drift.csv"), index=False)
    summ = df.groupby(["method", "eps"])[["drift", "n_edges"]].agg(
        ["mean", "std"]).round(3)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(os.path.join(RESULTS, "summary_eeg.csv"),
                              index=False)
    print(summ.to_string())

    # ----------------------------- figures --------------------------------
    # Fig 5a: raw EEG window with a real blink artifact (frontal channel)
    # find a high-amplitude frontal excursion in raw data
    af3 = (X_raw[:, 0] - np.median(X_raw[:, 0])) / mad[0, 0]
    # a TYPICAL blink (90th pct excursion), not the extreme transient
    thr = np.quantile(np.abs(af3), 0.999)
    cands = np.where(np.abs(af3) > 0.4 * thr)[0]
    i_blink = int(cands[len(cands) // 2])
    t_ax = np.arange(0, 400) / 128.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axes[0, 0]
    lo = max(0, i_blink - 100)
    seg = X_raw[lo:lo + 400]
    for j in [0, 1, 2, 11, 12, 13]:
        z = (seg[:, j] - np.median(X_raw[:, j])) / mad[0, j]
        ax.plot(t_ax, np.clip(z, -30, 30), lw=0.7, label=CH_NAMES[j])
    ax.axvline((i_blink - lo) / 128.0, color="crimson", ls="--", lw=1)
    ax.set_ylim(-30, 30)
    ax.set_title("(a) raw EEG, frontal channels (real blinks)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("robust z")
    ax.legend(fontsize=6, ncol=3)

    # Fig 5b: adjacency comparison at eps=0.10 vs reference (one method pair)
    ax = axes[0, 1]
    # choose fglasso reference vs perturbed graph at eps=0.10, rep 0
    rng_eps = np.random.default_rng(cfg["seed"] + 500 + 1)
    A_pert, _ = inject_blinks(A.reshape(n, 14, K), 0.10, rng_eps, K,
                              cfg["frontal"])
    A_pert = A_pert.reshape(n, 14 * K)
    E_ref = ref["fglasso"]
    E_pert = edge_set_from_omega(
        fit("fglasso", A_pert, lams["fglasso"], K, nu), K)
    M = np.zeros((14, 14, 3))
    for (a, b) in E_ref: M[a, b, 0] = M[b, a, 0] = 1.0
    for (a, b) in E_pert: M[a, b, 1] = M[b, a, 1] = 1.0
    both = E_ref & E_pert
    for (a, b) in both: M[a, b, 2] = M[b, a, 2] = 1.0
    img = np.zeros((14, 14))
    img[M[:, :, 0] == 1] = 1.0
    img[M[:, :, 1] == 1] = np.where(img[M[:, :, 1] == 1] == 1, 1.0, 2.0)
    img[M[:, :, 2] == 1] = 3.0
    im = ax.imshow(img, cmap="viridis", vmin=0, vmax=3)
    ax.set_xticks(range(14), CH_NAMES, rotation=90, fontsize=6)
    ax.set_yticks(range(14), CH_NAMES, fontsize=6)
    ax.set_title("(b) fGLasso graph at $\\epsilon=0.10$: reference (dark), "
                 "perturbed (light), both (yellow)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Fig 5c: drift vs eps
    ax = axes[1, 0]
    for method, mk in [("fglasso", "o"), ("t_em", "s"), ("altt_em", "^")]:
        g = df[df.method == method].groupby("eps")["drift"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                    marker=mk, capsize=3, label=method)
    ax.set_xlabel("injected blink fraction $\\epsilon$")
    ax.set_ylabel("network drift (Jaccard dist. to reference)")
    ax.set_title("(c) network drift under injected blinks")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Fig 5d: edge counts
    ax = axes[1, 1]
    for method, mk in [("fglasso", "o"), ("t_em", "s"), ("altt_em", "^")]:
        g = df[df.method == method].groupby("eps")["n_edges"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                    marker=mk, capsize=3, label=method)
    ax.set_xlabel("injected blink fraction $\\epsilon$")
    ax.set_ylabel("number of selected edges")
    ax.set_title("(d) edge counts")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("EEG Eye State (UCI): 14 channels, "
                 f"{n} epochs of {T/128:.2f} s")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_eeg.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig5_eeg.pdf"))
    print("done in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
