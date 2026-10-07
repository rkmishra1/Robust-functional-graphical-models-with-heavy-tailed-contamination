"""Theory-regime bridge cell (M3): weak-to-moderate edges where Condition 1
holds AND the edges are estimable (n=800, d=48, edge norms 0.10-0.25).

Verifies the certified margin of the exact drawn graph, then runs the
four estimators with density-matched penalties.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import numpy as np
from fgm import cosine_basis, true_precision, random_graph, true_edges
from check_irrep import irrep_margin
from run_full_sim import run_cell

out = os.path.join(ROOT, "results", "full_sim_bridge.csv")
SEED = 20261302
P, K, N = 12, 4, 800

rng = np.random.default_rng(SEED)
G = random_graph(P, 0.20, rng)
rng2 = np.random.default_rng(SEED)
random_graph(P, 0.20, rng2)
Om, _ = true_precision(G, K, rng2, edge_scale=(0.10, 0.25))
gam = irrep_margin(G, Om, K)
print(f"bridge graph: edges={len(true_edges(G))}, margin gamma={gam:.3f}",
      flush=True)
assert gam > 0, "drawn graph must satisfy Condition 1"

import pandas as pd, os
if os.path.exists(out):
    os.remove(out)
run_cell("bridge", P, N, K, [0.0, 0.10], "amplitude", 6, 7.0,
         ["fglasso", "t_em", "altt_em", "wglasso"],
         out_csv=out, log=print, graph_seed=SEED,
         edge_scale=(0.10, 0.25), density_target=len(true_edges(G)))
