"""
Stressed spike-artifact study: mechanism (iii) at strengths where per-channel
contamination actually damages the graph.

Methods: fGLasso (Gaussian baseline), f t-EM (model (a), whole-curve scale),
f alt-t EM (model (b), per-channel scale).

Design as in the pilot (p=12, K=4, T=64, n=200, edge_prob=0.20) but spikes:
blink-shaped (width ~0.15), amplitude 6-10 channel sd, in 4 of 12 channels,
eps in {0, 0.05, 0.10, 0.20}, R = 12 paired replicates.

Run:  python3 experiments/run_spike_stress.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fgm import (cosine_basis, true_precision, random_graph, sample_data,
                 curves_to_coeffs, block_glasso, fglasso_t_em,
                 fglasso_altt_em, edge_set_from_omega, true_edges,
                 gaussian_heldout_loglik, tune_lambda_cv, jaccard_dist, prf)

CFG = dict(p=12, K=4, T=64, n=200, edge_prob=0.20,
           eps_grid=[0.0, 0.05, 0.10, 0.20], R=12, nu=7.0,
           lam_grid=np.geomspace(0.05, 0.8, 10),
           n_chan=4, amp_range=(6.0, 10.0), width_range=(0.10, 0.20),
           n_tune=3, seed=20261010)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")


def inject_blinks(X, eps, rng, chans, amp_range, width_range):
    """Blink-like bumps in a fixed set of channels for eps*n observations."""
    n, p, T = X.shape
    m = max(1, int(round(eps * n))) if eps > 0 else 0
    idx = rng.choice(n, size=m, replace=False)
    t = (np.arange(T) + 0.5) / T
    Xc = X.copy()
    for i in idx:
        for j in rng.choice(chans, size=len(chans), replace=False):
            sd = X[:, j, :].std()
            amp = rng.uniform(*amp_range) * sd
            t0 = rng.uniform(0.2, 0.5)
            w = rng.uniform(*width_range)
            Xc[i, j] += amp * np.exp(-0.5 * ((t - t0) / w) ** 2)
    return Xc, idx


def fit(method, A, lam, K, nu):
    if method == "fglasso":
        return block_glasso(A.T @ A / A.shape[0], lam, K)
    if method == "t_em":
        return fglasso_t_em(A, lam, K, nu=nu, max_iter=30)["Omega"]
    if method == "altt_em":
        return fglasso_altt_em(A, lam, K, nu=nu, max_iter=25)["Omega"]
    raise ValueError(method)


def main():
    t0 = time.time()
    cfg = CFG
    p, K, T, n = cfg["p"], cfg["K"], cfg["T"], cfg["n"]
    rng_master = np.random.default_rng(cfg["seed"])
    G = random_graph(p, cfg["edge_prob"], rng_master)
    E_true = true_edges(G)
    Omega_true, _ = true_precision(G, K, rng_master)
    Phi = cosine_basis(T, K)
    chans = list(range(cfg["n_chan"]))       # first n_chan channels corrupted
    print(f"graph edges={len(E_true)}; corrupted channels={chans}")

    # lambda tuning per method on clean data (same criterion for all)
    tune_A = []
    for r in range(cfg["n_tune"]):
        rng = np.random.default_rng(cfg["seed"] + 1000 + r)
        tune_A.append(curves_to_coeffs(sample_data(Omega_true, Phi, n, rng), Phi))
    lams = {}
    for method in ["fglasso", "t_em", "altt_em"]:
        def _f(A_tr, lam, m=method):
            Om = fit(m, A_tr, lam, K, cfg["nu"])
            mu = 0.0 if m == "fglasso" else None
            return Om, mu
        def _ll(Om, aux, Av, m=method):
            if m != "fglasso":                # center by fitted mean
                # approximate centering: subtract column medians of Av
                Av = Av - np.median(Av, axis=0)
            return gaussian_heldout_loglik(Om, Av)
        lam, _ = tune_lambda_cv(tune_A, cfg["lam_grid"], K, _f, _ll, n_folds=3)
        lams[method] = lam
        print(f"  lambda[{method}] = {lam:.3f}  (t={time.time()-t0:.0f}s)")

    rows, store = [], {}
    for rep in range(cfg["R"]):
        rng_data = np.random.default_rng(cfg["seed"] + 10_000 + rep)
        X = sample_data(Omega_true, Phi, n, rng_data)
        for i_eps, eps in enumerate(cfg["eps_grid"]):
            rng_eps = np.random.default_rng(cfg["seed"] + 20_000 + 100 * rep + i_eps)
            Xc, _ = inject_blinks(X, eps, rng_eps, chans,
                                  cfg["amp_range"], cfg["width_range"])
            A_c = curves_to_coeffs(Xc, Phi)
            for method in ["fglasso", "t_em", "altt_em"]:
                Om = fit(method, A_c, lams[method], K, cfg["nu"])
                E = edge_set_from_omega(Om, K)
                store[(rep, eps, method)] = E
                m = prf(E, E_true)
                rows.append(dict(rep=rep, eps=eps, method=method,
                                 op_err=np.linalg.norm(Om - Omega_true) / np.linalg.norm(Omega_true),
                                 **m))
        print(f"  rep {rep+1}/{cfg['R']} (t={time.time()-t0:.0f}s)")

    for r in rows:
        E0 = store[(r["rep"], 0.0, r["method"])]
        r["jaccard"] = 0.0 if r["eps"] == 0.0 else jaccard_dist(E0, store[(r["rep"], r["eps"], r["method"])])

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sweep_spike_stress.csv"), index=False)
    cols = ["f1", "precision", "recall", "op_err", "jaccard", "n_edges_est"]
    summ = df.groupby(["method", "eps"])[cols].agg(["mean", "std"]).round(3)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(os.path.join(RESULTS, "summary_spike_stress.csv"),
                              index=False)
    print(summ.to_string())

    # figure: 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    styles = {"fglasso": ("o", "fGLasso (Gaussian)"),
              "t_em": ("s", "f t-EM (model a)"),
              "altt_em": ("^", "f alt-t EM (model b)")}
    for ax, (col, lab) in zip(axes.ravel(),
                              [("f1", "edge F1"),
                               ("jaccard", "selection instability"),
                               ("op_err", "relative operator-norm error"),
                               ("n_edges_est", "selected edges")]):
        for method, (mk, lab2) in styles.items():
            g = df[df.method == method].groupby("eps")[col]
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                        marker=mk, capsize=3, label=lab2)
        ax.set_xlabel("contamination level $\\epsilon$"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Blink artifacts, 4 of 12 channels, amplitude 6-10 sd "
                 "(n=200, p=12, K=4)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_spike_stress.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig6_spike_stress.pdf"))
    print("done in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
