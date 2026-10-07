"""Weak-edge cell (M3 fix): edge spectral norms in [0.05, 0.15], where the
nodewise block irrepresentability condition holds in every draw.

Also records the certified margin of the exact graph drawn, so the cell
and the condition check refer to the same Omega*.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import numpy as np
from fgm import cosine_basis, true_precision, random_graph, true_edges
from check_irrep import irrep_margin
from run_full_sim import run_cell

out = os.path.join(ROOT, "results", "full_sim_weakedge.csv")
SEED = 20261299          # explicit, process-stable
P, K = 12, 4

rng = np.random.default_rng(SEED)
G = random_graph(P, 0.20, rng)
# identical construction to run_cell's true_precision call:
rng2 = np.random.default_rng(SEED)
random_graph(P, 0.20, rng2)                 # consume identically
Om, mineig = true_precision(G, K, rng2, edge_scale=(0.05, 0.15))
gam = irrep_margin(G, Om, K)
print(f"weak-edge graph: edges={len(true_edges(G))}, "
      f"certified margin gamma={gam:.3f} (must be > 0)", flush=True)
assert gam > 0, "drawn graph must satisfy Condition 1"

import pandas as pd, os
if os.path.exists(out):
    os.remove(out)          # fresh run with density-matched penalties
run_cell("weakedge", P, 200, K, [0.0, 0.10, 0.20], "amplitude", 10, 7.0,
         ["fglasso", "t_em", "altt_em", "wglasso"],
         out_csv=out, log=print, graph_seed=SEED, edge_scale=(0.05, 0.15),
         density_target=len(true_edges(G)))
