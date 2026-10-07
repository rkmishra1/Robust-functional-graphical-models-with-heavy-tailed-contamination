"""
fgm.py -- Core module for the Robust Functional Graphical Models pilot.

Components
----------
1. Simulation of multivariate functional data whose conditional-independence
   graph (among the p component functions) is known exactly, generated through
   a tensor-product basis expansion with a block-sparse precision matrix
   (Qiao, Guo & James 2019 setup).

2. Block graphical lasso ("functional graphical lasso") via ADMM:
       min_{Omega > 0}  tr(Omega S) - logdet Omega
                        + lam * sum_{a<b} ||Omega_{ab}||_F
   with block size K.  Used both as the clean-data baseline estimator and as
   the penalized M-step of the robust EM.

3. Classical functional t-process EM (proposal variant (a)): stacked basis
   coefficients a_i ~ t_{pK}(mu, Sigma, nu); penalized EM with Gamma weights
   (Finegold & Drton 2009 machinery applied in the functional basis domain).

4. Contamination mechanisms:
   (ii) curve-level amplitude outliers (whole-curve rescaling);
   (iii) sporadic single-channel spike artifacts (blink-like bumps).

5. Metrics: block-level edge precision/recall/F1, operator-norm error,
   Jaccard instability of edge sets between paired runs.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Basis
# ----------------------------------------------------------------------------

def cosine_basis(T: int, K: int) -> np.ndarray:
    """Orthonormal (w.r.t. uniform measure on [0,1]) cosine basis on a T-grid.

    Returns Phi of shape (T, K) with Phi^T Phi / T = I (up to round-off).
    Columns: 1, sqrt(2) cos(pi t), ..., sqrt(2) cos(pi (K-1) t), evaluated at
    grid midpoints t_l = (l + 0.5)/T.
    """
    t = (np.arange(T) + 0.5) / T
    Phi = np.empty((T, K))
    Phi[:, 0] = 1.0
    for k in range(1, K):
        Phi[:, k] = np.sqrt(2.0) * np.cos(np.pi * k * t)
    # Discrete orthonormalization (removes midpoint-rule error exactly)
    G = Phi.T @ Phi / T
    w, V = np.linalg.eigh(G)
    Phi = Phi @ (V @ np.diag(w ** -0.5))
    return Phi


def curves_to_coeffs(X: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Project discretized curves onto the orthonormal basis.

    X: (n, p, T) or (p, T).  Returns A: (n, p*K) with columns ordered as
    (j, k) -> j*K + k  (channel-major, matching block structure of Omega).
    """
    T = Phi.shape[0]
    K = Phi.shape[1]
    single = X.ndim == 2
    if single:
        X = X[None]
    n, p, _ = X.shape
    # a_{jk} = (1/T) * sum_l X_j(t_l) phi_k(t_l)
    A = np.einsum("npt,tk->npk", X, Phi) / T
    return A.reshape(n, p * K) if not single else A.reshape(p * K)


# ----------------------------------------------------------------------------
# Simulation: block-sparse functional precision model
# ----------------------------------------------------------------------------

def random_graph(p: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Erdos-Renyi adjacency on p nodes, no self loops. Returns boolean (p,p)."""
    G = rng.random((p, p)) < edge_prob
    G = np.triu(G, 1)
    return G | G.T


def true_precision(G: np.ndarray, K: int, rng: np.random.Generator,
                   edge_scale=(0.3, 0.8), diag_buffer=0.3) -> tuple:
    """Build a block-sparse SPD precision matrix Omega (pK x pK).

    Block (a,b) is identically zero iff (a,b) is a non-edge; edge blocks are
    random symmetric K x K matrices.  Diagonal blocks are inflated so that the
    block-Gershgorin condition guarantees positive definiteness with minimum
    eigenvalue >= diag_buffer.

    Returns (Omega (pK,pK), block_omega list of (a,b,K,K) edge blocks).
    """
    p = G.shape[0]
    d = p * K
    off = np.zeros((d, d))
    # Random orthogonal edge-block "shapes": Q = R via QR of iid normals
    for a in range(p):
        for b in range(a + 1, p):
            if G[a, b]:
                M = rng.standard_normal((K, K))
                Q, _ = np.linalg.qr(M)
                s = rng.uniform(*edge_scale)
                B = s * Q @ np.diag(rng.choice([-1.0, 1.0], size=K)) @ Q.T
                B = 0.5 * (B + B.T)
                off[a * K:(a + 1) * K, b * K:(b + 1) * K] = B
                off[b * K:(b + 1) * K, a * K:(a + 1) * K] = B.T
    # Gershgorin-type diagonal inflation
    r = np.array([sum(np.linalg.norm(off[a*K:(a+1)*K, b*K:(b+1)*K], 2)
                      for b in range(p) if b != a) for a in range(p)])
    diag = np.zeros((d, d))
    for a in range(p):
        M = rng.standard_normal((K, K))
        Q, _ = np.linalg.qr(M)
        Dq = Q @ np.diag(rng.uniform(0.5, 1.0, size=K)) @ Q.T
        Dq = 0.5 * (Dq + Dq.T)
        block = (r[a] + diag_buffer) * np.eye(K) + 0.3 * Dq
        diag[a * K:(a + 1) * K, a * K:(a + 1) * K] = block
    Omega = diag + off
    # SPD sanity
    mineig = np.linalg.eigvalsh(Omega).min()
    if mineig <= 0.05 * diag_buffer:
        raise RuntimeError(f"Omega not well-conditioned (mineig={mineig:.3f})")
    return Omega, mineig


def sample_data(Omega: np.ndarray, Phi: np.ndarray, n: int,
                rng: np.random.Generator, mu: np.ndarray | None = None):
    """Sample n iid curves X_i ~ N(mu, Sigma) in function space via basis.

    Omega: (pK,pK) precision of the basis coefficients. Returns X (n,p,T).
    """
    p = Omega.shape[0] // Phi.shape[1]
    K = Phi.shape[1]
    Sigma = np.linalg.inv(Omega)
    A = rng.multivariate_normal(np.zeros(p * K) if mu is None else mu,
                                Sigma, size=n)
    return coeffs_to_curves(A, Phi, p)


def coeffs_to_curves(A: np.ndarray, Phi: np.ndarray, p: int) -> np.ndarray:
    """A: (n, pK) channel-major coefficients -> X: (n, p, T)."""
    n, d = A.shape
    K = d // p
    return np.einsum("npk,tk->npt", A.reshape(n, p, K), Phi)


# ----------------------------------------------------------------------------
# Contamination mechanisms
# ----------------------------------------------------------------------------

def contaminate_amplitude(X: np.ndarray, eps: float, rng: np.random.Generator,
                          scale_range=(3.0, 6.0)) -> tuple:
    """Mechanism (ii): rescale whole curves for a fraction eps of observations.

    Returns (X_cont, outlier_index).
    """
    n = X.shape[0]
    m = max(1, int(round(eps * n))) if eps > 0 else 0
    idx = rng.choice(n, size=m, replace=False)
    Xc = X.copy()
    Xc[idx] *= rng.uniform(*scale_range, size=(m, 1, 1))
    return Xc, idx


def contaminate_spikes(X: np.ndarray, eps: float, rng: np.random.Generator,
                       n_chan=2, amp_range=(4.0, 8.0)) -> tuple:
    """Mechanism (iii): add blink-like bumps in a few channels for eps*n obs.

    Bump: A_j * exp(-(t - t0)^2 / (2 * width^2)) with width ~ 0.03, t0 in
    [0.05, 0.30] (early, blink-like), A_j ~ Unif(amp_range) * channel sd.
    Returns (X_cont, dict with outlier_index, channels).
    """
    n, p, T = X.shape
    m = max(1, int(round(eps * n))) if eps > 0 else 0
    idx = rng.choice(n, size=m, replace=False)
    t = (np.arange(T) + 0.5) / T
    Xc = X.copy()
    for i in idx:
        chans = rng.choice(p, size=n_chan, replace=False)
        for j in chans:
            sd = X[:, j, :].std()
            amp = rng.uniform(*amp_range) * sd
            t0 = rng.uniform(0.05, 0.30)
            w = rng.uniform(0.02, 0.04)
            Xc[i, j] += amp * np.exp(-0.5 * ((t - t0) / w) ** 2)
    return Xc, {"index": idx, "channels": chans if m else None}


# ----------------------------------------------------------------------------
# Block graphical lasso via ADMM  (baseline fGLasso and robust M-step)
# ----------------------------------------------------------------------------

def _group_soft(B: np.ndarray, thr: float) -> np.ndarray:
    nrm = np.linalg.norm(B)
    if nrm <= thr:
        return np.zeros_like(B)
    return (1.0 - thr / nrm) * B


def _group_soft_blocks(W: np.ndarray, thr: float, K: int) -> np.ndarray:
    """Vectorized group soft-threshold of every off-diagonal K x K block."""
    d = W.shape[0]
    p = d // K
    B = W.reshape(p, K, p, K)
    nrm = np.sqrt((B * B).sum(axis=(1, 3)))          # (p,p) block Frobenius norms
    fac = np.clip(1.0 - thr / np.maximum(nrm, 1e-300), 0.0, 1.0)
    np.fill_diagonal(fac, 1.0)                        # diagonal blocks untouched
    out = B * fac[:, None, :, None]
    return out.reshape(d, d)


def block_glasso(S: np.ndarray, lam: float, K: int, rho: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-7) -> np.ndarray:
    """ADMM for the group-penalized precision estimation problem.

        min_{Omega > 0}  tr(Omega S) - logdet Omega + lam * sum_{a<b} ||Omega_ab||_F

    S: (d,d) sample covariance with d = p*K; blocks are K x K.
    Returns Omega (d,d).
    """
    d = S.shape[0]
    Omega = np.linalg.inv(S + 1e-6 * np.eye(d))
    Z = Omega.copy()
    U = np.zeros_like(Omega)
    for it in range(max_iter):
        # Omega-update: solve  min tr(S Om) - logdet Om + (rho/2)||Om - Z + U||^2
        M = S - rho * (Z - U)
        mu_eig, V = np.linalg.eigh(M)
        theta = (-mu_eig + np.sqrt(mu_eig ** 2 + 4.0 * rho)) / (2.0 * rho)
        Omega = (V * theta) @ V.T
        Omega = 0.5 * (Omega + Omega.T)
        # Z-update: group soft-threshold off-diagonal blocks of (Omega + U)
        Znew = _group_soft_blocks(Omega + U, lam / rho, K)
        # U-update and residuals
        r = np.linalg.norm(Omega - Znew)          # primal residual
        s = np.linalg.norm(rho * (Znew - Z))      # dual residual
        U += Omega - Znew
        Z = Znew
        if r < tol * max(1.0, np.linalg.norm(Omega)) and \
           s < tol * max(1.0, np.linalg.norm(rho * U)):
            break
    return Omega


def edge_set_from_omega(Omega: np.ndarray, K: int, tol: float = 1e-8) -> set:
    """Edges = pairs (a,b), a<b, whose K x K precision block is nonzero."""
    p = Omega.shape[0] // K
    edges = set()
    for a in range(p):
        for b in range(a + 1, p):
            if np.linalg.norm(Omega[a*K:(a+1)*K, b*K:(b+1)*K]) > tol:
                edges.add((a, b))
    return edges


def true_edges(G: np.ndarray) -> set:
    p = G.shape[0]
    return {(a, b) for a in range(p) for b in range(a + 1, p) if G[a, b]}


# ----------------------------------------------------------------------------
# Robust EM: classical functional t-process (variant (a))
# ----------------------------------------------------------------------------

def _ecme_nu_update(w: np.ndarray, nu: float, nu_bounds=(1.0, 1000.0),
                    n_newton: int = 30) -> float:
    """ECME update of the degrees of freedom (Peel & McLachlan 2000).

    With E[S_i] = 1/w_i and E[log S_i] ~= -log w_i, the profile
    complete-data likelihood in nu is stationary at
        log(nu/2) - digamma(nu/2) = mean_i( 1/w_i + log w_i ) - 1,
    solved by Newton iterations on nu (the left side decreases from +inf
    to 0 as nu grows, so the update tends to large nu when all weights
    are near 1 -- the Gaussian limit -- and to small nu under
    contamination).
    """
    from scipy.special import digamma, polygamma
    rhs = np.mean(1.0 / w + np.log(w)) - 1.0
    nu_cur = nu
    for _ in range(n_newton):
        h = nu_cur / 2.0
        g = np.log(h) - digamma(h) - rhs
        gp = 1.0 / nu_cur - polygamma(1, h) / 2.0
        step = g / gp
        nu_new = nu_cur - step
        if not (nu_bounds[0] < nu_new < nu_bounds[1]):
            nu_new = float(np.clip(nu_cur - np.sign(step) *
                                   0.5 * nu_cur, *nu_bounds))
        if abs(nu_new - nu_cur) < 1e-6:
            nu_cur = nu_new
            break
        nu_cur = nu_new
    return float(nu_cur)


def fglasso_t_em(A: np.ndarray, lam: float, K: int, nu: float = 4.0,
                 max_iter: int = 60, tol: float = 1e-6, verbose: bool = False,
                 profile_nu: bool = False):
    """Penalized EM for a_i ~ t_{pK}(mu, Sigma, nu), Sigma block-sparse.

    E-step: w_i = (nu + d) / (nu + delta_i),  delta_i = (a_i-mu)' Om (a_i-mu)
    M-step: mu <- weighted mean;  Omega <- block_glasso(weighted cov, lam)
    With profile_nu=True, nu is updated each iteration by the ECME
    degrees-of-freedom update (Peel & McLachlan 2000).

    A: (n, pK) basis coefficients. Returns dict with Omega, mu, weights,
    n_iter, nu (fitted if profiled).

    Initialization is robust: weights are seeded from a diagonal
    median/MAD Mahalanobis distance so that the first M-step already
    down-weights contaminated curves (avoids the outlier-inflated
    covariance local minimum).
    """
    n, d = A.shape
    med = np.median(A, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(A - med), axis=0), 1e-8)
    delta0 = (((A - med) / mad) ** 2).sum(axis=1)
    w = (nu + d) / (nu + delta0)
    mu = (w @ A) / w.sum()
    diff = A - mu
    Sw = (diff * w[:, None]).T @ diff / w.sum()
    Omega = block_glasso(Sw, lam, K)
    nu_fitted = nu
    for it in range(max_iter):
        Si = np.linalg.inv(Omega)
        diff = A - mu
        delta = np.einsum("ij,jk,ik->i", diff, Si, diff)
        w = (nu_fitted + d) / (nu_fitted + delta)
        if profile_nu:
            nu_fitted = _ecme_nu_update(w, nu_fitted)
        mu = (w @ A) / w.sum()
        diff = A - mu
        Sw = (diff * w[:, None]).T @ diff / w.sum()
        Omega_new = block_glasso(Sw, lam, K)
        rel = np.linalg.norm(Omega_new - Omega) / max(1e-12, np.linalg.norm(Omega))
        Omega = Omega_new
        if rel < tol:
            break
    return {"Omega": Omega, "mu": mu, "weights": w, "n_iter": it + 1,
            "nu": nu_fitted}


# ----------------------------------------------------------------------------
# Robust EM: functional alternative t-process (variant (b))
# ----------------------------------------------------------------------------

def _altt_inner_e_step(C: np.ndarray, Omega: np.ndarray, w: np.ndarray,
                       nu: float, K: int, inner: int = 4, damp: float = 0.5):
    """Blockwise variational E-step for model (b).

    The Gamma posterior rate of S_ij is driven by the within-channel
    quadratic form  delta_ij = c_ij' Omega_jj c_ij  (exact when Omega is
    block diagonal; the cross-channel terms, of size Omega_jk, are dropped
    for stability -- they are second-order for sparse precision matrices,
    and their plug-in version can turn the rate negative).
    Update (damped, clamped):  w_ij <- (nu + K) / (nu + delta_ij).
    """
    n, p, _ = C.shape
    Ob = Omega.reshape(p, K, p, K)
    for _ in range(inner):
        delta = np.empty((n, p))
        for j in range(p):
            delta[:, j] = np.einsum("nm,mq,nq->n", C[:, j, :],
                                    Ob[j, :, j, :], C[:, j, :])
        w_new = (nu + K) / (nu + delta)
        w = (1 - damp) * w + damp * np.clip(w_new, 0.02, 20.0)
    return w


def _altt_weighted_cov(C: np.ndarray, w: np.ndarray, K: int,
                       psd: bool = True) -> np.ndarray:
    """Block-pair-normalized weighted covariance under model (b).

    S_jk = sum_i sqrt(w_ij w_ik) c_ij c_ik' / sum_i sqrt(w_ij w_ik),
    consistent for Sigma_jk since E[ sqrt(S_ij S_ik) c_ij c_ik' ] ~ Sigma_jk.
    psd=True projects onto the PSD cone (ridge + eigenvalue clipping) for
    numerical safety of the M-step.
    """
    n, p, _ = C.shape
    S = np.zeros((p * K, p * K))
    r = np.sqrt(np.maximum(w, 1e-12))                # (n,p)
    for j in range(p):
        for k in range(j, p):
            num = np.einsum("n,nm,nq->mq", r[:, j] * r[:, k],
                            C[:, j, :], C[:, k, :])
            den = (r[:, j] * r[:, k]).sum()
            B = num / max(den, 1e-12)
            if j == k:
                S[j*K:(j+1)*K, j*K:(j+1)*K] = 0.5 * (B + B.T)
            else:
                S[j*K:(j+1)*K, k*K:(k+1)*K] = B
                S[k*K:(k+1)*K, j*K:(j+1)*K] = B.T
    if psd:
        S = S + 1e-6 * np.trace(S) / (p * K) * np.eye(p * K)
        ev, V = np.linalg.eigh(0.5 * (S + S.T))
        floor = 1e-8 * max(1.0, np.abs(ev).max())
        S = (V * np.maximum(ev, floor)) @ V.T
    return S


def fglasso_altt_em(A: np.ndarray, lam: float, K: int, nu: float = 7.0,
                    max_iter: int = 30, tol: float = 1e-5,
                    inner: int = 8, damp: float = 0.5):
    """Penalized estimation for the functional ALTERNATIVE t-process (model b).

    a_ij = mu_j + S_ij^{-1/2} eps_ij,  eps_i ~ N_{pK}(0, Sigma),
    S_ij ~ Gamma(nu/2, rate nu/2) i.i.d.  Per-channel weights isolate
    channel-level contamination (blinks in one EEG electrode).

    A: (n, pK). Returns dict with Omega, mu (p,K), weights (n,p), n_iter.
    """
    n, d = A.shape
    p = d // K
    A3 = A.reshape(n, p, K)
    med = np.median(A3, axis=0)                       # (p,K)
    mad = np.maximum(1.4826 * np.median(np.abs(A3 - med), axis=0), 1e-8)
    delta0 = (((A3 - med) / mad) ** 2).sum(axis=2)    # (n,p)
    w = (nu + K) / (nu + delta0)
    mu = np.einsum("ij,ijk->jk", w, A3) / w.sum(axis=0)[:, None]
    C = A3 - mu
    S = _altt_weighted_cov(C, w, K)
    Omega = block_glasso(S, lam, K)
    for it in range(max_iter):
        w = _altt_inner_e_step(C, Omega, w, nu, K, inner=inner, damp=damp)
        mu = np.einsum("ij,ijk->jk", w, A3) / w.sum(axis=0)[:, None]
        C = A3 - mu
        S = _altt_weighted_cov(C, w, K)
        Omega_new = block_glasso(S, lam, K)
        rel = np.linalg.norm(Omega_new - Omega) / max(1e-12, np.linalg.norm(Omega))
        Omega = Omega_new
        if rel < tol:
            break
    return {"Omega": Omega, "mu": mu, "weights": w, "n_iter": it + 1}


# ----------------------------------------------------------------------------
# Metrics and model selection
# ----------------------------------------------------------------------------

def prf(est: set, tru: set) -> dict:
    tp = len(est & tru)
    prec = tp / len(est) if est else 0.0
    rec = tp / len(tru) if tru else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_edges_est": len(est), "n_edges_true": len(tru)}


def jaccard_dist(E1: set, E2: set) -> float:
    u = E1 | E2
    return 1.0 - len(E1 & E2) / len(u) if u else 0.0


def ebic(Omega: np.ndarray, S: np.ndarray, n: int, K: int, gamma: float = 1.0):
    """Extended BIC for the group-penalized Gaussian likelihood."""
    d = Omega.shape[0]
    p = d // K
    sign, logdet = np.linalg.slogdet(Omega)
    df = 0
    for a in range(p):
        for b in range(a + 1, p):
            B = Omega[a*K:(a+1)*K, b*K:(b+1)*K]
            df += int(np.count_nonzero(np.abs(B) > 1e-10))
    df += d  # diagonal blocks counted fully (upper-tri symmetry)
    ll = n * (np.trace(S @ Omega) - logdet)
    return ll + df * np.log(n) + 2 * gamma * df * np.log(d)


def tune_lambda(cov_list: list, lam_grid: np.ndarray, K: int, n: int,
                solver=block_glasso) -> float:
    """Pick lambda minimizing eBIC averaged over clean covariance samples."""
    scores = []
    for lam in lam_grid:
        eb = 0.0
        for S in cov_list:
            Om = solver(S, lam, K)
            eb += ebic(Om, S, n, K)
        scores.append(eb / len(cov_list))
    return float(lam_grid[int(np.argmin(scores))]), scores


# ----------------------------------------------------------------------------
# Cross-validated tuning (held-out log-likelihood on clean data)
# ----------------------------------------------------------------------------

def gaussian_heldout_loglik(Omega: np.ndarray, A_val: np.ndarray) -> float:
    d = A_val.shape[1]
    _, logdet = np.linalg.slogdet(Omega)
    q = np.einsum("ij,jk,ik->i", A_val, Omega, A_val)
    return float(0.5 * (logdet - q - d * np.log(2 * np.pi)).mean())


def t_heldout_loglik(Omega: np.ndarray, mu: np.ndarray, nu: float,
                     A_val: np.ndarray) -> float:
    from scipy.special import gammaln
    d = A_val.shape[1]
    diff = A_val - mu
    _, logdet = np.linalg.slogdet(Omega)
    delta = np.einsum("ij,jk,ik->i", diff, Omega, diff)
    ll = (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
          - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(delta / nu))
    return float(ll.mean())


def tune_lambda_cv(A_list: list, lam_grid: np.ndarray, K: int, fit_fn,
                   loglik_fn, n_folds: int = 3, seed: int = 0):
    """K-fold CV on clean coefficient data.

    fit_fn(A_train, lam) -> (Omega, aux); loglik_fn(Omega, aux, A_val) -> float.
    Returns (best lambda, mean scores per lambda).
    """
    scores = []
    for lam in lam_grid:
        tot = 0.0
        for A in A_list:
            n = A.shape[0]
            idx = np.random.default_rng(seed).permutation(n)
            for f in np.array_split(idx, n_folds):
                mask = np.ones(n, dtype=bool)
                mask[f] = False
                Om, aux = fit_fn(A[mask], float(lam))
                tot += loglik_fn(Om, aux, A[f])
        scores.append(tot / len(A_list))
    scores = np.asarray(scores, dtype=float)
    best = int(np.nanargmax(scores))
    return float(lam_grid[best]), scores
