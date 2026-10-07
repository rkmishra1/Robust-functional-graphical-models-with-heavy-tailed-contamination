"""Quick unit tests for fgm.py (run before the pilot sweep)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from fgm import (cosine_basis, curves_to_coeffs, coeffs_to_curves,
                 true_precision, random_graph, sample_data, block_glasso,
                 fglasso_t_em, edge_set_from_omega, true_edges, contaminate_amplitude)

rng = np.random.default_rng(7)
ok = True

# 1. Basis orthonormality
Phi = cosine_basis(64, 4)
err = np.abs(Phi.T @ Phi / 64 - np.eye(4)).max()
print(f"[1] basis orthonormality error: {err:.2e}")
ok &= err < 1e-10

# 2. Round trip curves -> coeffs -> curves
p, T, K = 3, 64, 4
A = rng.standard_normal((5, p * K))
X = coeffs_to_curves(A, Phi, p)
A2 = curves_to_coeffs(X, Phi)
err = np.abs(A - A2).max()
print(f"[2] round-trip coefficient error: {err:.2e}")
ok &= err < 1e-10

# 3. Simulation: empirical coefficient covariance matches truth
G = random_graph(6, 0.3, rng)
Omega, mineig = true_precision(G, K, rng)
print(f"[3] true Omega min eig: {mineig:.3f}, edges: {len(true_edges(G))}")
n = 20000
X = sample_data(Omega, Phi, n, rng)
A = curves_to_coeffs(X, Phi)
S = A.T @ A / n
Sigma_true = np.linalg.inv(Omega)
err = np.linalg.norm(S - Sigma_true) / np.linalg.norm(Sigma_true)
print(f"[3] covariance rel. error (n={n}): {err:.4f}")
ok &= err < 0.05

# 4. ADMM stationarity: gradient of objective at solution ~ 0
pG = G.shape[0]
S_small = Sigma_true + 0.01 * np.eye(pG * K)
lam = 0.05
Om = block_glasso(S_small, lam, K, max_iter=3000, tol=1e-10)
# gradient of smooth part: S - Omega^{-1}; KKT: S - Om^{-1} = lam * (subgradient blocks)
grad = S_small - np.linalg.inv(Om)
viol = 0.0
for a in range(pG):
    for b in range(a + 1, pG):
        B = grad[a*K:(a+1)*K, b*K:(b+1)*K]
        nrm = np.linalg.norm(Om[a*K:(a+1)*K, b*K:(b+1)*K])
        if nrm > 1e-8:
            viol = max(viol, abs(np.linalg.norm(B) - lam))
        else:
            viol = max(viol, max(0.0, np.linalg.norm(B) - lam))
print(f"[4] ADMM KKT violation: {viol:.2e}")
ok &= viol < 1e-4

# 5. Clean-data edge recovery sanity
n = 300
X = sample_data(Omega, Phi, n, rng)
A = curves_to_coeffs(X, Phi)
S = A.T @ A / n
Om = block_glasso(S, 0.08, K)
E_est = edge_set_from_omega(Om, K)
E_true = true_edges(G)
tp = len(E_est & E_true)
print(f"[5] clean fglasso: est {len(E_est)} edges, true {len(E_true)}, TP {tp}")
ok &= tp >= 0.6 * len(E_true)

# 6. t-EM: outliers get small weights
Xc, idx = contaminate_amplitude(X, 0.10, rng)
Ac = curves_to_coeffs(Xc, Phi)
res = fglasso_t_em(Ac, 0.08, K, nu=4.0, max_iter=40)
w = res["weights"]
print(f"[6] t-EM: {res['n_iter']} iters; mean weight outliers "
      f"{w[idx].mean():.3f} vs clean {np.delete(w, idx).mean():.3f}")
ok &= w[idx].mean() < 0.7 * np.delete(w, idx).mean()

# 7. Robust EM recovers more true edges than plain fglasso under contamination
S_plain = Ac.T @ Ac / n
Om_plain = block_glasso(S_plain, 0.08, K)
E_plain = edge_set_from_omega(Om_plain, K)
E_rob = edge_set_from_omega(res["Omega"], K)
print(f"[7] under eps=0.10: plain F1-edges TP={len(E_plain & E_true)}, "
      f"robust TP={len(E_rob & E_true)} (true {len(E_true)})")

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
