"""
Full simulation study (manuscript Section 'Full simulation study').

Design: one-at-a-time around the base cell (p=12, n=200, K=4, T=64,
edge_prob=0.20, amplitude contamination), plus a mechanism comparison and
a degrees-of-freedom table.

Cell groups (run concurrently as separate processes):
  --cells main1   : base, n=100, n=400, K=3, K=6      (p=12)
  --cells main2   : p=20, p=50
  --cells extra   : mechanism comparison (amplitude vs blink) + nu table

Methods: fglasso (Gaussian baseline), t_em (model a), altt_em (model b),
wglasso (discretize-then-winsorize + glasso), and in the nu table
t_em with profiled nu.  Penalties: k-fold CV held-out Gaussian
log-likelihood on the clean data of each cell.

Every cell appends its rows to results/full_sim_<group>.csv immediately,
so partial runs remain usable.

Run:  python3 -u experiments/run_full_sim.py --cells main1 &
      python3 -u experiments/run_full_sim.py --cells main2 &
      python3 -u experiments/run_full_sim.py --cells extra &
"""
import os, sys, csv, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fgm import (cosine_basis, true_precision, random_graph, sample_data,
                 curves_to_coeffs, block_glasso, fglasso_t_em,
                 fglasso_altt_em, edge_set_from_omega, true_edges,
                 gaussian_heldout_loglik, tune_lambda_cv,
                 contaminate_amplitude)

RESULTS = os.path.join(ROOT, "results")
SEED = 20261201

METHODS = ["fglasso", "t_em", "altt_em", "wglasso"]


# ------------------------------- fitting -----------------------------------

def fit(method, A, lam, K, nu):
    """Fit one estimator.  Returns (Omega, fitted_nu_or_None)."""
    if method == "fglasso":
        return block_glasso(A.T @ A / A.shape[0], lam, K), None
    if method == "t_em":
        r = fglasso_t_em(A, lam, K, nu=nu, max_iter=25)
        return r["Omega"], None
    if method == "altt_em":
        r = fglasso_altt_em(A, lam, K, nu=nu, max_iter=20)
        return r["Omega"], None
    if method == "wglasso":
        # discretize-then-robust: winsorize standardized coefficients at 3
        # robust sd, then plain glasso
        med = np.median(A, axis=0)
        mad = np.maximum(1.4826 * np.median(np.abs(A - med), axis=0), 1e-8)
        Aw = np.clip((A - med) / mad, -3.0, 3.0)
        return block_glasso(Aw.T @ Aw / Aw.shape[0], lam, K), None
    if method == "t_em_prof":
        r = fglasso_t_em(A, lam, K, nu=nu, max_iter=25, profile_nu=True)
        return r["Omega"], r["nu"]
    raise ValueError(method)


def lambda_for(method, tune_A, K, nu, n_lambda=5, n_folds=2):
    grid = np.geomspace(0.05, 1.0, n_lambda)
    if method in ("fglasso", "wglasso"):
        grid = np.geomspace(0.05, 0.8, n_lambda)

    def _f(A_tr, lam, m=method):
        Om, _ = fit(m, A_tr, lam, K, nu)
        aux = None
        if m in ("t_em", "altt_em", "t_em_prof"):
            # center held-out data by a robust location for scoring
            aux = np.median(A_tr, axis=0)
        return Om, aux

    def _ll(Om, aux, Av):
        if aux is not None:
            Av = Av - np.median(Av, axis=0)
        return gaussian_heldout_loglik(Om, Av)

    lam, _ = tune_lambda_cv(tune_A, grid, K, _f, _ll,
                            n_folds=n_folds, seed=SEED)
    return lam


def make_contamination(X, eps, mechanism, rng):
    if mechanism == "amplitude":
        return contaminate_amplitude(X, eps, rng)[0]
    if mechanism == "blink":
        from run_spike_stress import inject_blinks
        return inject_blinks(X, eps, rng, list(range(4)),
                             (6.0, 10.0), (0.10, 0.20))[0]
    raise ValueError(mechanism)


def _append_csv(out_csv, rows):
    if out_csv and rows:
        exists = os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(rows)


def run_cell(tag, p, n, K, eps_grid, mechanism, R, nu, methods,
             edge_prob=0.20, out_csv=None, log=print, n_folds=2,
             density_target=None, fixed_lam=None, graph_seed=None,
             edge_scale=None):
    # NOTE: the legacy seed is hash(tag)-based and therefore NOT stable
    # across processes (randomized str hashing).  New cells should pass an
    # explicit integer graph_seed.
    t0 = time.time()
    seed_base = graph_seed if graph_seed is not None else SEED + hash(tag) % 100000
    rng_master = np.random.default_rng(seed_base)
    G = random_graph(p, edge_prob, rng_master)
    E_true = true_edges(G)
    if edge_scale is not None:
        Omega_true, _ = true_precision(G, K, rng_master, edge_scale=edge_scale)
    else:
        Omega_true, _ = true_precision(G, K, rng_master)
    Phi = cosine_basis(64, K)

    tune_A = []
    for r in range(2):
        rng = np.random.default_rng(SEED + 777 + r)
        tune_A.append(curves_to_coeffs(
            sample_data(Omega_true, Phi, n, rng), Phi))
    lams = {}
    if density_target:
        # benchmark operating point: lambda whose clean fit has edge count
        # closest to the true edge count (robust to CV over-shrinkage at
        # large d)
        for m in methods:
            best_lam, best_gap = None, 1e18
            for lam in np.geomspace(0.02, 6.0, 10):
                Om, _ = fit(m, tune_A[0], float(lam), K, nu)
                ne = len(edge_set_from_omega(Om, K))
                gap = abs(ne - density_target)
                if gap < best_gap:
                    best_gap, best_lam = gap, float(lam)
                if ne < 0.3 * density_target:
                    break
            lams[m] = best_lam
            log(f"  [{tag}] lambda[{m}]={best_lam:.3f} (density cal, "
                f"{time.time()-t0:.0f}s)")
    elif fixed_lam:
        for m in methods:
            lams[m] = fixed_lam
            log(f"  [{tag}] lambda[{m}]={lams[m]:.3f} (fixed, "
                f"{time.time()-t0:.0f}s)")
    else:
        for m in methods:
            lams[m] = lambda_for(m, tune_A, K, nu, n_folds=n_folds)
            log(f"  [{tag}] lambda[{m}]={lams[m]:.3f} ({time.time()-t0:.0f}s)")

    # resume: skip (rep, eps, method) triples already checkpointed
    done_keys = set()
    if out_csv and os.path.exists(out_csv):
        import pandas as pd
        prev = pd.read_csv(out_csv)
        prev = prev[prev["cell"] == tag]
        done_keys = {(int(r.rep), float(r.eps), r.method)
                     for r in prev.itertuples()}
        if done_keys:
            log(f"  [{tag}] resuming: {len(done_keys)} fits already done")

    store = {}
    pending = []
    for rep in range(R):
        rng_data = np.random.default_rng(SEED + 10_000 + 97 * rep
                                         + hash(tag) % 500)
        X = sample_data(Omega_true, Phi, n, rng_data)
        for i_eps, eps in enumerate(eps_grid):
            # eps = 0.0 always precedes contaminated levels, so the clean
            # fit for this rep is available for the Jaccard computation
            if all((rep, eps, m) in done_keys for m in methods):
                if eps == 0.0:
                    # still need the clean edge sets for later Jaccard
                    A0 = curves_to_coeffs(X, Phi)
                    for m in methods:
                        if (rep, 0.0, m) not in done_keys:
                            continue
                        # recover the clean fit cheaply at the same lambda
                        store[(rep, 0.0, m)] = None
                continue
            rng_eps = np.random.default_rng(SEED + 20_000 + 131 * rep
                                            + 17 * i_eps + hash(tag) % 500)
            Xc = make_contamination(X, eps, mechanism, rng_eps)
            A_c = curves_to_coeffs(Xc, Phi)
            for m in methods:
                if (rep, eps, m) in done_keys:
                    continue
                ft0 = time.time()
                Om, nu_hat = fit(m, A_c, lams[m], K, nu)
                fit_s = time.time() - ft0
                E = edge_set_from_omega(Om, K)
                store[(rep, eps, m)] = E
                if eps == 0.0:
                    store.setdefault((rep, 0.0, m), E)
                tp = len(E & E_true)
                prec = tp / len(E) if E else 0.0
                rec = tp / len(E_true)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                E0 = store.get((rep, 0.0, m))
                if eps == 0.0 or E0 is None:
                    jac = 0.0 if eps == 0.0 else None
                else:
                    u = len(E0 | E)
                    jac = 0.0 if u == 0 else round(1 - len(E0 & E) / u, 4)
                pending.append(dict(
                    cell=tag, p=p, n=n, K=K, eps=eps, method=m, rep=rep,
                    f1=round(f1, 4), precision=round(prec, 4),
                    recall=round(rec, 4),
                    op_err=round(float(np.linalg.norm(Om - Omega_true)
                                       / np.linalg.norm(Omega_true)), 4),
                    n_edges=len(E), fit_s=round(fit_s, 3),
                    lam=round(lams[m], 4), nu=nu,
                    nu_hat=round(nu_hat, 2) if nu_hat else None,
                    jaccard=jac))
        _append_csv(out_csv, pending)
        pending = []
        log(f"  [{tag}] rep {rep+1}/{R} ({time.time()-t0:.0f}s)")

    log(f"  [{tag}] cell done in {(time.time()-t0)/60:.1f} min")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True,
                    choices=["main1", "main2", "extra"])
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"full_sim_{args.cells}.csv")
    print(f"== group {args.cells} -> {out} ==", flush=True)

    done = set()
    if os.path.exists(out):
        import pandas as pd
        done = set(pd.read_csv(out)["cell"].unique())
        print(f"skipping already-done cells: {sorted(done)}", flush=True)

    if args.cells == "main1":
        # base, sample-size axis, basis axis (p=12; amplitude; eps {0,.1,.2})
        for tag, p, n, K, R, eps_grid, folds in [
            ("base", 12, 200, 4, 10, [0.0, 0.10, 0.20], 3),
            ("n100", 12, 100, 4, 10, [0.0, 0.10, 0.20], 5),
            ("n400", 12, 400, 4, 10, [0.0, 0.10, 0.20], 3),
            ("K3", 12, 200, 3, 10, [0.0, 0.10], 3),
            ("K6", 12, 200, 6, 10, [0.0, 0.10], 3),
        ]:
            if tag in done:
                continue
            run_cell(tag, p, n, K, eps_grid, "amplitude", R, 7.0, METHODS,
                     out_csv=out, log=print, n_folds=folds)

    elif args.cells == "main2":
        done = set()
        if os.path.exists(out):
            import pandas as pd
            done = set(pd.read_csv(out)["cell"].unique())
            print(f"skipping already-done cells: {sorted(done)}", flush=True)
        for tag, p, n, K, R, eps_grid, dens in [
            ("p20", 20, 200, 4, 8, [0.0, 0.10, 0.20], None),
            ("p50", 50, 400, 4, 6, [0.0, 0.10, 0.20], 245),
        ]:
            if tag in done:
                continue
            run_cell(tag, p, n, K, eps_grid, "amplitude", R, 7.0, METHODS,
                     out_csv=out, log=print, density_target=dens)

    elif args.cells == "extra":
        # mechanism comparison at the base cell
        for mech in ["amplitude", "blink"]:
            run_cell(f"mech_{mech}", 12, 200, 4, [0.0, 0.10, 0.20],
                     mech, 8, 7.0, METHODS, out_csv=out, log=print)
        # degrees-of-freedom table (amplitude, eps {0, .10})
        for nu in [3.0, 5.0, 7.0, 10.0, 15.0]:
            run_cell(f"nu{int(nu)}", 12, 200, 4, [0.0, 0.10], "amplitude",
                     8, nu, ["t_em"], out_csv=out, log=print)
        run_cell("nu_prof", 12, 200, 4, [0.0, 0.10], "amplitude",
                 8, 5.0, ["t_em_prof"], out_csv=out, log=print)
        run_cell("nu_altt", 12, 200, 4, [0.0, 0.10], "amplitude",
                 8, 7.0, ["altt_em"], out_csv=out, log=print)


if __name__ == "__main__":
    main()
