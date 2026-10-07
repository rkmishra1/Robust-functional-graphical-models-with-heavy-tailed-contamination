"""Unit tests for model (b): functional alternative t-process EM."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from fgm import (cosine_basis, true_precision, random_graph, sample_data,
                 curves_to_coeffs, block_glasso, fglasso_altt_em,
                 edge_set_from_omega, true_edges, contaminate_spikes)

rng = np.random.default_rng(11)
ok = True

p, K, T, n = 6, 3, 48, 300
G = random_graph(p, 0.25, rng)
Omega_true, mineig = true_precision(G, K, rng)
Phi = cosine_basis(T, K)
X = sample_data(Omega_true, Phi, n, rng)
A = curves_to_coeffs(X, Phi)

# 1. Clean data: alt-t EM should behave like a sane estimator
res = fglasso_altt_em(A, 0.10, K, nu=7.0)
E = edge_set_from_omega(res["Omega"], K)
tp = len(E & true_edges(G))
print(f"[1] clean alt-t: est {len(E)} edges, TP {tp}/{len(true_edges(G))}, "
      f"iters {res['n_iter']}")
ok &= tp >= 0.6 * len(true_edges(G))

# 2. Channel-selective weights: contaminate ONE channel of SOME observations
Xc = X.copy()
m = 25
idx = rng.choice(n, size=m, replace=False)
chan = 2
t = (np.arange(T) + 0.5) / T
for i in idx:
    sd = X[:, chan, :].std()
    Xc[i, chan] += rng.uniform(7, 10) * sd * np.exp(-0.5 * ((t - 0.35) / 0.15) ** 2)
Ac = curves_to_coeffs(Xc, Phi)
res2 = fglasso_altt_em(Ac, 0.20, K, nu=7.0)
W = res2["weights"]                     # (n, p)
w_out_channel = W[idx, chan].mean()     # contaminated channel of outlier trials
w_out_other = W[idx][:, [c for c in range(p) if c != chan]].mean()
w_clean = np.delete(W, idx, axis=0).mean()
print(f"[2] weights (m={m} spike trials): contaminated (i,j) {w_out_channel:.3f} | "
      f"other channels of same trial {w_out_other:.3f} | clean trials {w_clean:.3f}")
ok &= w_out_channel < 0.75 * w_out_other        # channel-selectivity
ok &= w_out_other >= 0.7 * w_clean               # other channels unaffected

# 3. Under channel contamination, alt-t recovers graph better than plain glasso
S_plain = Ac.T @ Ac / n
Om_plain = block_glasso(S_plain, 0.20, K)
E_plain = edge_set_from_omega(Om_plain, K)
E_altt = edge_set_from_omega(res2["Omega"], K)
E_true = true_edges(G)
f1 = lambda E: (2*len(E&E_true)/(len(E)+len(E_true))) if (len(E)+len(E_true)) else 0
print(f"[3] wide spike (blink-like) in 1 channel: fGLasso F1 {f1(E_plain):.2f} "
      f"({len(E_plain)} edges) vs alt-t F1 {f1(E_altt):.2f} ({len(E_altt)} edges)")
ok &= f1(E_altt) >= f1(E_plain) - 0.05

print("\nALL ALT-T TESTS PASSED" if ok else "\nSOME ALT-T TESTS FAILED")
sys.exit(0 if ok else 1)
