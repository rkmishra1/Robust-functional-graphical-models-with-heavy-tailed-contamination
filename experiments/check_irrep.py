"""
Numerical check of the block irrepresentable condition (appendix
Condition 1) for the simulation designs.

For a drawn block precision Omega* with active set S, and each inactive
(j,k), compute the K x K block
    M_jk = [ Omega*_{jS} (Omega*_{SS})^{-1} ]_k
and the certified margin
    gamma_cert = 1 - sqrt(K) * max_{inactive (j,k)} ||M_jk||_op ,
where the sqrt(K) factor bounds the Frobenius norm of M_jk V_k for any
block V_k with ||V_k||_op <= 1 (so gamma_cert > 0 certifies the
condition of Theorem 2; gamma_cert < 0 means the condition is not
certified by this bound).

Run:  python3 experiments/check_irrep.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fgm import cosine_basis, true_precision, random_graph, true_edges

RES = os.path.join(ROOT, "results")


def irrep_margin(G, Omega, K):
    """Nodewise block irrepresentable margin (the block extension of the
    Zhao & Yu 2006 / Ravikumar et al. 2011 condition), for one draw.

    For each node j with neighborhood N(j) (nodes sharing an edge with j):
        margin_j = 1 - max_{i not in N(j) u {j}} | row_i of
                   Sigma*_{N(j)^c, N(j)} (Sigma*_{N(j),N(j)})^{-1}
                   sgn(Omega*_{j,N(j)}) | ,
    with Sigma* = (Omega*)^{-1}; the max runs over scalar coefficient
    entries.  Returns min_j margin_j.
    """
    p = G.shape[0]
    d = Omega.shape[0]
    Sig = np.linalg.inv(Omega)
    nbrs = {j: sorted({b for (a, b) in true_edges(G) if a == j} |
                      {a for (a, b) in true_edges(G) if b == j})
            for j in range(p)}
    margin_min = np.inf
    for j in range(p):
        N = nbrs[j]
        if not N:
            continue
        idxN = np.concatenate([np.arange(v * K, (v + 1) * K) for v in N])
        others = [i for i in range(p) if i != j and i not in N]
        if not others:
            continue
        idxO = np.concatenate([np.arange(v * K, (v + 1) * K) for v in others])
        A = Sig[np.ix_(idxO, idxN)] @ np.linalg.inv(Sig[np.ix_(idxN, idxN)])
        V = np.sign(Omega[j * K:(j + 1) * K, :][:, idxN]).T   # K|N| x K
        M = A @ V                                             # |others|*K x K
        m = np.abs(M).max()
        margin_min = min(margin_min, 1.0 - m)
    return margin_min


def main():
    rng = np.random.default_rng(20261299)
    out = []
    for tag, p, K, edge_prob, escale in [
            ("p12", 12, 4, 0.20, (0.3, 0.8)),
            ("p20", 20, 4, 0.20, (0.3, 0.8)),
            ("p50", 50, 4, 0.20, (0.3, 0.8)),
            ("p12 weak", 12, 4, 0.20, (0.05, 0.15)),
            ("p20 weak", 20, 4, 0.20, (0.05, 0.15))]:
        margins, n_edges = [], []
        for r in range(200):
            g = random_graph(p, edge_prob, rng)
            Om, _ = true_precision(g, K, rng, edge_scale=escale)
            gam = irrep_margin(g, Om, K)
            margins.append(gam)
            n_edges.append(len(true_edges(g)))
        margins = np.array(margins)
        out.append(dict(design=tag, p=p, K=K, draws=200,
                        gamma_cert_mean=round(float(margins.mean()), 3),
                        gamma_cert_sd=round(float(margins.std()), 3),
                        gamma_cert_min=round(float(margins.min()), 3),
                        frac_positive=round(float((margins > 0).mean()), 3),
                        ))
        print(f"{tag}: gamma_cert mean {margins.mean():.3f} "
              f"sd {margins.std():.3f} min {margins.min():.3f} "
              f"| P(gamma>0) {(margins>0).mean():.2f} "
              f"(mean edges {np.mean(n_edges):.0f})")
    pd = __import__("pandas")
    pd.DataFrame(out).to_csv(os.path.join(RES, "irrep_margins.csv"),
                             index=False)
    print("wrote results/irrep_margins.csv")


if __name__ == "__main__":
    main()
