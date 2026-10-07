"""Retuned-baseline control (Devil's Advocate fix).

Question: can the Gaussian functional graphical lasso recover, by tuning
its penalty on contaminated data alone, the stability that the robust
estimators get from their weights?

Protocol: base-cell design (p=12, n=200, K=4, amplitude contamination),
eps in {0, 0.10}.  For each replicate and each lambda on a grid: fit
fGLasso on clean (edge set E0) and on contaminated data (E1); record F1
and Jaccard(E0, E1).  Then compare three operating points per replicate:
  (i)   CV lambda tuned on clean data (the paper's protocol),
  (ii)  oracle lambda (max F1 under contamination, chosen with truth),
  (iii) stability-optimal lambda (min Jaccard under contamination),
against the robust estimator at its CV lambda.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from fgm import (cosine_basis, true_precision, random_graph, sample_data,
                 curves_to_coeffs, block_glasso, fglasso_t_em,
                 edge_set_from_omega, true_edges, contaminate_amplitude,
                 prf)
import csv

SEED = 20261301
P, K, T, N = 12, 4, 64, 200
R = 10
EPS = 0.10
NU = 7.0
GRID = np.geomspace(0.03, 0.8, 8)
out = os.path.join(ROOT, "results", "retune_control.csv")

rng_master = np.random.default_rng(SEED)
G = random_graph(P, 0.20, rng_master)
E_true = true_edges(G)
Omega_true, _ = true_precision(G, K, rng_master)
Phi = cosine_basis(T, K)
print(f"true edges: {len(E_true)}", flush=True)

rows = []
for rep in range(R):
    rng_data = np.random.default_rng(SEED + 10_000 + 97 * rep)
    X = sample_data(Omega_true, Phi, N, rng_data)
    rng_eps = np.random.default_rng(SEED + 20_000 + 131 * rep)
    Xc, _ = contaminate_amplitude(X, EPS, rng_eps)
    A0 = curves_to_coeffs(X, Phi)
    A1 = curves_to_coeffs(Xc, Phi)
    # robust reference at its CV lambda (0.118 from the full-sim protocol),
    # including its own clean-vs-contaminated instability
    r0 = fglasso_t_em(A0, 0.118, K, nu=NU, max_iter=25)
    E_r0 = edge_set_from_omega(r0["Omega"], K)
    r = fglasso_t_em(A1, 0.118, K, nu=NU, max_iter=25)
    E_r = edge_set_from_omega(r["Omega"], K)
    pf = prf(E_r, E_true)
    u_r = len(E_r0 | E_r)
    jac_r = 1.0 - len(E_r0 & E_r) / u_r if u_r else 0.0
    rows.append(dict(rep=rep, method="t_em", lam=0.118,
                     **{k: round(v, 4) for k, v in pf.items()},
                     jaccard=round(jac_r, 4)))
    for lam in GRID:
        S0 = A0.T @ A0 / N
        E0 = edge_set_from_omega(block_glasso(S0, float(lam), K), K)
        S1 = A1.T @ A1 / N
        Om1 = block_glasso(S1, float(lam), K)
        E1 = edge_set_from_omega(Om1, K)
        f0 = prf(E0, E_true)
        f1 = prf(E1, E_true)
        u = len(E0 | E1)
        jac = 1.0 - len(E0 & E1) / u if u else 0.0
        rows.append(dict(rep=rep, method="fglasso", lam=round(float(lam), 4),
                         **{k: round(v, 4) for k, v in f1.items()},
                         jaccard=round(jac, 4)))
    print(f"rep {rep+1}/{R} done", flush=True)

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["rep", "method", "lam", "precision",
                                      "recall", "f1", "n_edges_est",
                                      "n_edges_true", "jaccard"])
    w.writeheader()
    w.writerows(rows)

# summary: the three operating points
import collections
by_rep = collections.defaultdict(dict)
for r in rows:
    if r["method"] == "fglasso":
        by_rep[r["rep"]][r["lam"]] = r
    else:
        by_rep[r["rep"]]["robust"] = r
summ = {"cv": [], "oracle_f1": [], "stable": []}
for rep, d in by_rep.items():
    lam_cv = min((l for l in d if l != "robust"),
                 key=lambda l: abs(l - 0.2))        # CV picked ~0.2 in base cell
    summ["cv"].append(d[lam_cv])
    best_f1_lam = max((l for l in d if l != "robust"),
                      key=lambda l: d[l]["f1"])
    summ["oracle_f1"].append(d[best_f1_lam])
    best_j_lam = min((l for l in d if l != "robust"),
                     key=lambda l: d[l].get("jaccard", 1))
    summ["stable"].append(d[best_j_lam])
for name, lst in summ.items():
    f1s = [x["f1"] for x in lst]
    jacs = [x.get("jaccard", np.nan) for x in lst]
    print(f"{name:10s}: F1 {np.mean(f1s):.3f} (sd {np.std(f1s):.3f})  "
          f"instability {np.nanmean(jacs):.3f}")
rob = [r for r in rows if r["method"] == "t_em"]
print(f"{'t_em':10s}: F1 {np.mean([r['f1'] for r in rob]):.3f} "
      f"(sd {np.std([r['f1'] for r in rob]):.3f})")
print("wrote", out)
