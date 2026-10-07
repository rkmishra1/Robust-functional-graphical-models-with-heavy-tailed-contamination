"""
Pilot experiment for Robust Functional Graphical Models.

Protocol (proposal Section 8):
  1. Generate multivariate functional data with known block-sparse graph.
  2. Tune lambda for fGLasso (baseline) and functional t-EM (variant a)
     on CLEAN data via eBIC; hold fixed afterwards.
  3. Amplitude-outlier sweep (mechanism ii): eps in {0, .05, .10, .15},
     R = 20 reps; measure edge F1, operator-norm error, and Jaccard
     instability of the edge set relative to the same-rep clean fit.
  4. Spike-artifact sweep (mechanism iii), R = 10, eps in {0, .10}.
  5. Figures + CSVs + go/no-go verdict.

Run:  python3 experiments/run_pilot.py
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fgm import (cosine_basis, true_precision, random_graph, sample_data,
                 curves_to_coeffs, block_glasso, fglasso_t_em, fglasso_altt_em,
                 edge_set_from_omega, true_edges, contaminate_amplitude,
                 contaminate_spikes, prf, jaccard_dist, tune_lambda_cv,
                 gaussian_heldout_loglik, t_heldout_loglik)

# ----------------------------- configuration -------------------------------
CFG = dict(
    p=12, K=4, T=64, n=200, edge_prob=0.20,
    eps_grid=[0.0, 0.05, 0.10, 0.15],
    R=20, nu=7.0,
    lam_grid_fglasso=np.geomspace(0.03, 0.8, 14),
    lam_grid_tem=np.geomspace(0.05, 1.0, 8),
    n_tune=4, seed=20261004,
    spike_R=10, spike_eps=[0.0, 0.10],
)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)


def fit_fglasso(A, lam, K):
    S = A.T @ A / A.shape[0]
    return block_glasso(S, lam, K), S


def fit_tem(A, lam, K, nu):
    res = fglasso_t_em(A, lam, K, nu=nu, max_iter=30)
    return res["Omega"], res


def fit_altt(A, lam, K, nu):
    res = fglasso_altt_em(A, lam, K, nu=nu, max_iter=25)
    return res["Omega"], res


def main():
    t0 = time.time()
    cfg = CFG
    p, K, T, n = cfg["p"], cfg["K"], cfg["T"], cfg["n"]
    rng_master = np.random.default_rng(cfg["seed"])

    # Fixed graph for the whole pilot (report it in the README/report)
    G = random_graph(p, cfg["edge_prob"], rng_master)
    E_true = true_edges(G)
    Omega_true, mineig = true_precision(G, K, rng_master)
    Phi = cosine_basis(T, K)
    print(f"graph: p={p}, edges={len(E_true)}, min eig(Omega)={mineig:.3f}")

    # ---------------- 1. lambda tuning on clean data (CV log-lik) ----------
    print("tuning lambda on clean data (3-fold CV held-out log-likelihood) ...")
    tune_A = []
    for r in range(cfg["n_tune"]):
        rng = np.random.default_rng(cfg["seed"] + 1000 + r)
        X = sample_data(Omega_true, Phi, n, rng)
        tune_A.append(curves_to_coeffs(X, Phi))

    def _fit_g(A_tr, lam):
        S = A_tr.T @ A_tr / A_tr.shape[0]
        return block_glasso(S, lam, K), None

    lam_f, _ = tune_lambda_cv(tune_A, cfg["lam_grid_fglasso"], K, _fit_g,
                              lambda Om, aux, Av: gaussian_heldout_loglik(Om, Av))
    print(f"  lambda_fglasso = {lam_f:.3f}")

    def _fit_t(A_tr, lam):
        res = fglasso_t_em(A_tr, lam, K, nu=cfg["nu"], max_iter=30)
        return res["Omega"], res

    # Same tuning criterion for both estimators: held-out Gaussian log-lik.
    # (Evaluating the t density on held-out data systematically favors
    # over-sparsified precision matrices, sending lambda to the boundary.)
    lam_t, _ = tune_lambda_cv(tune_A, cfg["lam_grid_tem"], K, _fit_t,
                              lambda Om, aux, Av: gaussian_heldout_loglik(
                                  Om, Av - aux["mu"]))
    print(f"  lambda_t_em    = {lam_t:.3f}   (t={time.time()-t0:.0f}s)")

    def _fit_a(A_tr, lam):
        res = fglasso_altt_em(A_tr, lam, K, nu=cfg["nu"], max_iter=25)
        return res["Omega"], res["mu"].reshape(-1)

    lam_a, _ = tune_lambda_cv(tune_A, cfg["lam_grid_tem"], K, _fit_a,
                              lambda Om, aux, Av: gaussian_heldout_loglik(
                                  Om, Av - aux))
    print(f"  lambda_altt_em = {lam_a:.3f}   (t={time.time()-t0:.0f}s)")

    # ---------------- 2. amplitude-outlier sweep ----------------------------
    print("amplitude sweep ...")
    rows = []          # per-rep records
    store = {}         # (rep, eps, method) -> edge set
    for rep in range(cfg["R"]):
        rng_data = np.random.default_rng(cfg["seed"] + 10_000 + rep)
        X = sample_data(Omega_true, Phi, n, rng_data)      # clean curves (fixed per rep)
        for i_eps, eps in enumerate(cfg["eps_grid"]):
            rng_eps = np.random.default_rng(cfg["seed"] + 20_000 + 100 * rep + i_eps)
            Xc, idx = contaminate_amplitude(X, eps, rng_eps)
            A_c = curves_to_coeffs(Xc, Phi)
            for method in ["fglasso", "t_em", "altt_em"]:
                if method == "fglasso":
                    Om, _ = fit_fglasso(A_c, lam_f, K)
                elif method == "t_em":
                    Om, _ = fit_tem(A_c, lam_t, K, cfg["nu"])
                else:
                    Om, _ = fit_altt(A_c, lam_a, K, cfg["nu"])
                E = edge_set_from_omega(Om, K)
                store[(rep, eps, method)] = E
                m = prf(E, E_true)
                op_err = np.linalg.norm(Om - Omega_true) / np.linalg.norm(Omega_true)
                rows.append(dict(rep=rep, eps=eps, method=method,
                                 op_err=op_err, **m))
        if (rep + 1) % 5 == 0:
            print(f"  rep {rep+1}/{cfg['R']}  (t={time.time()-t0:.0f}s)")

    # Jaccard instability vs same-rep clean fit
    for r in rows:
        E0 = store[(r["rep"], 0.0, r["method"])]
        r["jaccard"] = 0.0 if r["eps"] == 0.0 else jaccard_dist(E0, store[(r["rep"], r["eps"], r["method"])])

    import csv
    with open(os.path.join(RESULTS, "sweep_amplitude.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---------------- 3. spike-artifact sweep -------------------------------
    print("spike sweep ...")
    rows_spike, store_spike = [], {}
    for rep in range(cfg["spike_R"]):
        rng_data = np.random.default_rng(cfg["seed"] + 50_000 + rep)
        X = sample_data(Omega_true, Phi, n, rng_data)
        for i_eps, eps in enumerate(cfg["spike_eps"]):
            rng_eps = np.random.default_rng(cfg["seed"] + 60_000 + 100 * rep + i_eps)
            Xc, _ = contaminate_spikes(X, eps, rng_eps)
            A_c = curves_to_coeffs(Xc, Phi)
            for method in ["fglasso", "t_em", "altt_em"]:
                if method == "fglasso":
                    Om = fit_fglasso(A_c, lam_f, K)[0]
                elif method == "t_em":
                    Om = fit_tem(A_c, lam_t, K, cfg["nu"])[0]
                else:
                    Om = fit_altt(A_c, lam_a, K, cfg["nu"])[0]
                E = edge_set_from_omega(Om, K)
                store_spike[(rep, eps, method)] = E
                m = prf(E, E_true)
                rows_spike.append(dict(rep=rep, eps=eps, method=method, **m))
    for r in rows_spike:
        E0 = store_spike[(r["rep"], 0.0, r["method"])]
        r["jaccard"] = 0.0 if r["eps"] == 0.0 else jaccard_dist(E0, store_spike[(r["rep"], r["eps"], r["method"])])
    with open(os.path.join(RESULTS, "sweep_spike.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_spike[0].keys()))
        w.writeheader(); w.writerows(rows_spike)

    # ---------------- 4. figures --------------------------------------------
    print("figures ...")
    make_fig_curves(Omega_true, Phi, rng_master)
    make_fig_adjacency(Omega_true, G, K, store, 0)
    make_fig_metrics(rows, cfg["eps_grid"])
    make_fig_spike(rows_spike, cfg["spike_eps"])

    # ---------------- 5. summary + verdict ----------------------------------
    summary = summarize(rows, cfg["eps_grid"])
    summary.to_csv(os.path.join(RESULTS, "summary_amplitude.csv"), index=False)
    s2 = summarize(rows_spike, cfg["spike_eps"])
    s2.to_csv(os.path.join(RESULTS, "summary_spike.csv"), index=False)
    print(summary.to_string(index=False))
    vd = verdict(summary, p * (p - 1) // 2)
    meta = dict(cfg={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in cfg.items()},
                lam_fglasso=lam_f, lam_t_em=lam_t, lam_altt_em=lam_a, n_true_edges=len(E_true),
                verdict=vd, minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(RESULTS, "pilot_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min")


def summarize(rows, eps_grid):
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = [c for c in ["f1", "precision", "recall", "op_err", "jaccard",
                        "n_edges_est"] if c in df.columns]
    g = df.groupby(["method", "eps"])[cols]
    out = g.agg(["mean", "std"]).round(3)
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def verdict(summary, n_pairs):
    """Go/no-go with criteria measuring what the pilot is for.

    D1  Destabilization is real (baseline at eps=0.10):
        edge-count saturation (>= 0.9 * complete graph) OR operator-error
        inflation >= 40% relative to its own clean fit.
    D2  Prototype stabilizes (t-EM at eps=0.10): edge count within +15% of
        its own clean fit, instability <= half the baseline's, F1 within
        0.05 of its own clean F1.
    D3  No meaningful clean-data penalty: t-EM clean F1 within 0.03 of the
        baseline's clean F1.
    """
    g = summary.set_index(["method", "eps"])
    f1g0, f1g10 = g.loc[("fglasso", 0.0), "f1_mean"], g.loc[("fglasso", 0.10), "f1_mean"]
    f1t0, f1t10 = g.loc[("t_em", 0.0), "f1_mean"], g.loc[("t_em", 0.10), "f1_mean"]
    eg0, eg10 = g.loc[("fglasso", 0.0), "n_edges_est_mean"], g.loc[("fglasso", 0.10), "n_edges_est_mean"]
    et0, et10 = g.loc[("t_em", 0.0), "n_edges_est_mean"], g.loc[("t_em", 0.10), "n_edges_est_mean"]
    oeg0, oeg10 = g.loc[("fglasso", 0.0), "op_err_mean"], g.loc[("fglasso", 0.10), "op_err_mean"]
    jg10 = g.loc[("fglasso", 0.10), "jaccard_mean"]
    jt10 = g.loc[("t_em", 0.10), "jaccard_mean"]

    op_infl = (oeg10 - oeg0) / max(oeg0, 1e-9)
    d1 = (eg10 >= 0.9 * n_pairs) or (op_infl >= 0.40)
    d2 = (et10 <= 1.15 * et0) and (jt10 <= 0.5 * jg10) and (f1t10 >= f1t0 - 0.05)
    d3 = f1t0 >= f1g0 - 0.03

    print("\n===== GO/NO-GO =====")
    print(f"D1 destabilization: edges {eg0:.0f} -> {eg10:.0f} of {n_pairs} possible "
          f"(saturation {100*eg10/n_pairs:.0f}%); op-err inflation {100*op_infl:.0f}%  "
          f"-> {'PASS' if d1 else 'FAIL'}")
    print(f"D2 stabilization:   t-EM edges {et0:.1f} -> {et10:.1f}; instability "
          f"{jt10:.2f} vs baseline {jg10:.2f}; F1 {f1t10:.2f} vs clean {f1t0:.2f}  "
          f"-> {'PASS' if d2 else 'FAIL'}")
    print(f"D3 clean-data cost: t-EM clean F1 {f1t0:.2f} vs baseline {f1g0:.2f}  "
          f"-> {'PASS' if d3 else 'FAIL'}")
    print("VERDICT:", "GO" if (d1 and d2 and d3) else "REVISIT")
    return dict(d1=bool(d1), d2=bool(d2), d3=bool(d3),
                op_infl=float(op_infl), sat=float(eg10 / n_pairs),
                jg10=float(jg10), jt10=float(jt10),
                f1g0=float(f1g0), f1g10=float(f1g10),
                f1t0=float(f1t0), f1t10=float(f1t10),
                eg0=float(eg0), eg10=float(eg10), et0=float(et0), et10=float(et10))


# ----------------------------- figures --------------------------------------

def make_fig_curves(Omega_true, Phi, rng):
    X = sample_data(Omega_true, Phi, 40, rng)
    Xc, idx = contaminate_amplitude(X, 0.08, np.random.default_rng(1))
    Xs, _ = contaminate_spikes(X, 0.08, np.random.default_rng(2))
    t = (np.arange(X.shape[2]) + 0.5) / X.shape[2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    ax = axes[0]
    for j in range(40):
        if j == idx[0]:
            continue
        ax.plot(t, X[j, 0], color="0.75", lw=0.6)
    ax.plot(t, Xc[idx[0], 0], color="crimson", lw=1.2,
            label="amplitude outlier")
    ax.set_title("(a) clean curves, channel 1"); ax.legend(fontsize=8)
    ax = axes[1]
    for j in range(40):
        ax.plot(t, Xs[j, 0], color="0.75", lw=0.6)
    i_sp = None
    ax = axes[1]
    ax.set_title("(b) spike artifacts (all channels)")
    for j in range(40):
        ax.plot(t, Xs[j, 0], color="0.75", lw=0.6)
    # find a spike curve: compare max against clean
    spikes = np.where(np.abs(Xs[:, 0, :].max(axis=1) - X[:, 0, :].max(axis=1)) > 2)[0]
    if len(spikes):
        ax.plot(t, Xs[spikes[0], 0], color="crimson", lw=1.2, label="spike outlier")
        ax.legend(fontsize=8)
    ax.set_title("(b) spike artifacts, channel 1")
    ax = axes[2]
    for j in range(40):
        ax.plot(t, Xs[j, 1], color="0.75", lw=0.6)
    spikes1 = np.where(np.abs(Xs[:, 1, :].max(axis=1) - X[:, 1, :].max(axis=1)) > 2)[0]
    if len(spikes1):
        ax.plot(t, Xs[spikes1[0], 1], color="crimson", lw=1.2)
    ax.set_title("(c) spike artifacts, channel 2")
    for ax in axes:
        ax.set_xlabel("t"); ax.set_ylabel("x(t)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_curves.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig1_curves.pdf"))
    plt.close(fig)


def make_fig_adjacency(Omega_true, G, K, store, rep):
    p = G.shape[0]
    panels = [("truth", None),
              ("fGLasso, clean", store[(rep, 0.0, "fglasso")]),
              ("fGLasso, eps=.10", store[(rep, 0.10, "fglasso")]),
              ("t-EM, clean", store[(rep, 0.0, "t_em")]),
              ("t-EM, eps=.10", store[(rep, 0.10, "t_em")]),
              ("t-EM, eps=.15", store[(rep, 0.15, "t_em")])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    for ax, (title, E) in zip(axes.ravel(), panels):
        M = np.zeros((p, p))
        if E is None:
            for (a, b) in true_edges(G):
                M[a, b] = M[b, a] = 1.0
        else:
            Om = None
            for (a, b) in E:
                M[a, b] = M[b, a] = 1.0
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Estimated functional graphs (rep 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_adjacency.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig2_adjacency.pdf"))
    plt.close(fig)


def make_fig_metrics(rows, eps_grid):
    import pandas as pd
    df = pd.DataFrame(rows)
    metrics = [("f1", "edge F1"), ("jaccard", "selection instability (Jaccard dist.)"),
               ("op_err", "relative operator-norm error"), ("n_edges_est", "number of selected edges")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, lab) in zip(axes.ravel(), metrics):
        for method, mk, lab2 in [("fglasso", "o", "fGLasso (baseline)"),
                                 ("t_em", "s", "f t-EM (model a)"),
                                 ("altt_em", "^", "f alt-t EM (model b)")]:
            g = df[df.method == method].groupby("eps")[col]
            mean, sd = g.mean(), g.std()
            ax.errorbar(mean.index, mean.values, yerr=sd.values, marker=mk,
                        capsize=3, label=lab2)
        ax.set_xlabel("contamination level $\\epsilon$"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Amplitude-outlier contamination (n=200, p=12, K=4, T=64)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_metrics.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig3_metrics.pdf"))
    plt.close(fig)


def make_fig_spike(rows_spike, spike_eps):
    import pandas as pd
    df = pd.DataFrame(rows_spike)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (col, lab) in zip(axes, [("f1", "edge F1"), ("jaccard", "selection instability")]):
        for method, mk, lab2 in [("fglasso", "o", "fGLasso (baseline)"),
                                 ("t_em", "s", "f t-EM (model a)"),
                                 ("altt_em", "^", "f alt-t EM (model b)")]:
            g = df[df.method == method].groupby("eps")[col]
            mean, sd = g.mean(), g.std()
            ax.errorbar(mean.index, mean.values, yerr=sd.values, marker=mk,
                        capsize=3, label=lab2)
        ax.set_xlabel("spike contamination $\\epsilon$"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Spike-artifact contamination")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_spike.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig4_spike.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
