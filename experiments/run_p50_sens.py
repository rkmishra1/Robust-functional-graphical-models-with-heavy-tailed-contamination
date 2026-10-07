"""p50 operating-point sensitivity: density targets 150 / 245 / 400."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from run_full_sim import run_cell

out = os.path.join(ROOT, "results", "full_sim_p50_sens.csv")
for tag, target in [("p50_t150", 150), ("p50_t400", 400)]:
    run_cell(tag, 50, 400, 4, [0.0, 0.10, 0.20], "amplitude", 6, 7.0,
             ["fglasso", "t_em", "altt_em", "wglasso"],
             out_csv=out, density_target=target)
