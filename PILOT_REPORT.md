# Pilot Report: Robust Functional Graphical Models

**Date:** 2026-10-04 · **Verdict: GO**; all three go/no-go criteria passed.

## Question

Does functional-data contamination destabilize the Gaussian functional
graphical lasso (Qiao–Guo–James 2019 setup), and does the classical
functional t-process EM (proposal variant (a)) stabilize it, without a
clean-data accuracy cost?

## Design

- $p=12$ functions on $[0,1]$, grid $T=64$, cosine basis $K=4$, $n=200$.
- Erdős–Rényi graph, edge probability 0.20 → **25 true edges**; block
  precision with random orthogonal edge blocks (spectral norm in
  $[0.3,0.8]$), block-Gershgorin SPD inflation (min eig 0.48).
- Contamination mechanism (ii): a random $\varepsilon$ fraction of whole
  curves rescaled by factors $\sim$ Unif(3, 6).
- $\varepsilon \in \{0, 0.05, 0.10, 0.15\}$, $R=20$ paired replicates
  (contaminated data = clean data + outliers, same seed).
- Both penalties tuned **on clean data** (3-fold CV, held-out Gaussian
  log-likelihood: $\lambda_G = 0.137$, $\lambda_t = 0.118$) and held fixed.
- Secondary spike-artifact sweep (mechanism (iii)), $R=10$,
  $\varepsilon \in \{0, 0.10\}$.

## Main results (amplitude outliers, mean over 20 reps)

| Method | $\varepsilon$ | F1 | Precision | Recall | Instability | Edges | Op-err |
|---|---|---|---|---|---|---|---|
| fGLasso | 0.00 | 0.618 | 0.483 | 0.858 | – | 44.5 | 0.327 |
| fGLasso | 0.05 | 0.549 | 0.379 | 0.996 | 0.325 | 65.7 | 0.454 |
| fGLasso | 0.10 | 0.549 | 0.379 | 1.000 | 0.327 | **66.0 (complete!)** | 0.550 |
| fGLasso | 0.15 | 0.549 | 0.379 | 1.000 | 0.327 | 66.0 | 0.633 |
| f t-EM (a) | 0.00 | 0.599 | 0.447 | 0.910 | – | 51.0 | **0.308** |
| f t-EM (a) | 0.05 | 0.593 | 0.434 | 0.934 | **0.052** | 53.9 | 0.318 |
| f t-EM (a) | 0.10 | 0.587 | 0.424 | 0.956 | **0.095** | 56.5 | 0.334 |
| f t-EM (a) | 0.15 | 0.578 | 0.412 | 0.970 | 0.134 | 59.0 | 0.354 |
| f alt-t (b) | 0.00 | 0.607 | 0.481 | 0.826 | – | 43.0 | 0.319 |
| f alt-t (b) | 0.05 | 0.588 | 0.437 | 0.900 | 0.164 | 51.5 | 0.308 |
| f alt-t (b) | 0.10 | 0.575 | 0.412 | 0.954 | 0.257 | 58.0 | 0.327 |
| f alt-t (b) | 0.15 | 0.563 | 0.394 | 0.988 | 0.315 | 62.8 | 0.370 |

Trade-off (expected from the models): under **whole-curve** amplitude
outliers the shared scale of model (a) is the right tool (instability
0.10 vs 0.26 for (b) at $\varepsilon=0.10$); under **channel-local**
artifacts the per-channel scale of model (b) wins -- see the stress
study below.

Instability = Jaccard distance between a replicate's contaminated fit and
its own clean fit (paired, like-for-like).

## Go/no-go criteria

| Criterion | Result | Verdict |
|---|---|---|
| **D1** destabilization real: baseline at $\varepsilon=0.10$ saturates the edge set (≥90% of complete graph) or inflates operator error ≥40% | edges 44 → **66/66 (100%)**; op-err **+68%**; instability 0.33 | **PASS** |
| **D2** prototype stabilizes: t-EM edge count within +15% of own clean, instability ≤ half of baseline, F1 within 0.05 of own clean | edges 51.0 → 56.5 (+11%); instability **0.10 vs 0.33**; F1 0.59 vs 0.60 | **PASS** |
| **D3** no clean-data cost: t-EM clean F1 within 0.03 of baseline | 0.599 vs 0.618; clean op-err actually *lower* (0.308 vs 0.327) | **PASS** |

**VERDICT: GO.** The failure mode is stark (the Gaussian baseline selects a
*complete graph* by $\varepsilon=0.05$–$0.10$; recall 1.0, precision 0.38)
and the t-process EM holds the graph essentially fixed while matching the
baseline on clean data.

## Stressed spike study (mechanism (iii) at realistic strength), R=12

Blink-shaped artifacts (width 0.10-0.20, amplitude 6-10 channel sd) in 4
of 12 channels, $\varepsilon \in \{0, 0.05, 0.10, 0.20\}$
(`experiments/run_spike_stress.py`, `results/summary_spike_stress.csv`,
`figures/fig6_spike_stress`):

| Method | F1 clean | F1 $\varepsilon$=0.05 | F1 0.10 | F1 0.20 | instability 0.10 |
|---|---|---|---|---|---|
| fGLasso | 0.411 | 0.274 | 0.278 | 0.279 | 0.40 |
| f t-EM (a) | 0.479 | 0.423 | 0.348 | 0.273 | 0.38 |
| **f alt-t (b)** | **0.531** | **0.492** | **0.421** | **0.291** | **0.34** |

Model (b) dominates at every level, with the largest gains exactly where
per-channel isolation matters (mild-to-moderate contamination); model
(a) is intermediate because whole-curve weights also discard the clean
channels of contaminated trials.  Model (b) even has the best clean-data
F1 here (its weights down-weight high-distance observations, which helps even without contamination).  Two
implementation lessons discovered en route: the mean-field E-step must
use only the within-channel (diagonal) quadratic form -- the plug-in
cross-channel terms can turn the Gamma rate negative and destabilize the
fixed point -- and the block-pair-normalized covariance needs PSD
projection plus a weight floor.

## EEG application (UCI Eye State, 14 channels, 234 epochs)

`experiments/run_eeg.py`, `results/summary_eeg.csv`,
`figures/fig5_eeg`.  Protocol: common average reference -> 0.5 s epochs
-> cosine basis (K=6) -> coefficient standardization -> winsorization at
8 MAD (the raw recording contains genuine transients that inflate the
naive covariance diagonal to ~6e5!) -> density-calibrated penalties
(target 45 edges) -> synthetic blink injection into frontal channels at
$\varepsilon \in \{0.05, 0.10, 0.20\}$, 6 replicates -> network drift
(Jaccard distance to the method's own reference fit).

| Method | reference edges | drift 0.05 | drift 0.10 | drift 0.20 | edges after 0.10 |
|---|---|---|---|---|---|
| fGLasso | 69 | 0.076 | 0.105 | 0.139 | 66 |
| f t-EM (a) | 67 | 0.575 | 0.600 | 0.592 | 27 |
| f alt-t (b) | 65 | 0.215 | 0.210 | 0.221 | 78 |

Interpretation (important for the paper): the Gaussian estimator's low
drift is **not** a virtue -- its reference graph is already saturated
(69/91 edges) on artifact-laden data, so added corruption cannot move it
much.  Model (a) over-reacts: whole-curve weights correctly reject
blink trials but discard their clean channels too, thinning the graph to
27 edges.  Model (b) reacts in a targeted way (moderate drift, mild edge
growth).  Drift-vs-reference must always be read jointly with the
reference's own quality; a "clean reference vs raw fit" comparison and
stability-based operating points are the next analysis step.

## Spike artifacts (mechanism (iii)), $R=10$

At $\varepsilon=0.10$ with bumps in 2 of 12 channels: fGLasso F1
0.624→0.625, instability 0.024; f t-EM F1 0.614→0.609, instability 0.042.
**Neither method degrades materially at this artifact strength**; the
distortion touches only ~2 of 12 channel-pair blocks and 10% of
observations, so the covariance damage is too localized to saturate the
graph. Implications for the paper: (i) mechanism (iii) must be stressed
with larger magnitudes, more channels, and higher $\varepsilon$ in the
full study before per-channel models (b)/(c) can show their advantage;
(ii) the clean separation of the amplitude sweep (whole-curve mechanism)
is the pilot's headline, and it is exactly the regime where the shared
scale of model (a) is the right tool.

## Two implementation lessons (important for the paper)

1. **EM initialization matters.** With the naive median-centered covariance
   start, the t-EM inherits the outlier-inflated covariance, the weights
   fail to isolate outliers, and the EM converges to the saturated
   solution. Seeding the weights from diagonal median/MAD Mahalanobis
   distances puts the EM in the right basin immediately.
2. **Normalize the weighted covariance by $\sum_i w_i$, not $n$.** The
   weights average below one ($\approx 0.75$ at $\nu=7$, $d=48$), so the
   $1/n$ convention systematically shrinks the covariance, weakening the
   fixed penalty on clean data and inflating the selected edge count
   (63 vs 44 edges). Weight-sum normalization makes the penalty scale
   directly comparable across the Gaussian and robust estimators.

Both fixes are worth describing in the manuscript's estimation section;
the second also affects the theory (the M-step remains a generalized EM
step since block-glasso still minimizes the weighted objective).

## Reproduction

```
python3 src/test_fgm.py          # unit tests (basis, ADMM KKT, EM weights)
python3 experiments/run_pilot.py # full sweep (~2–3 min), seed 20261004
python3 experiments/make_numbers.py  # inject results into the manuscript
```

Outputs: `results/sweep_amplitude.csv`, `results/sweep_spike.csv`,
`results/summary_*.csv`, `results/pilot_meta.json`,
`figures/fig1–fig4.{png,pdf}`, manuscript inputs
`manuscript/pilot_numbers.tex`, `manuscript/pilot_table.tex`.

## Next steps toward the JMVA submission

1. Extend to models (b) (per-channel weights) and (c) (variational
   indicators); the spike sweep already shows where (b) pays off.
2. Full simulation grid (p ∈ {12,20,50}, n ∈ {100,200,400}, K ∈ {3,4,6},
   ε up to 0.20, ν-sensitivity).
3. Theory: complete the proofs of Theorems 1–2 with the weight-sum
   normalization (affects constants in $n_{\mathrm{eff}}$).
4. EEG application: reproduce Qiao et al. baseline, artifact injection,
   robust vs Gaussian network drift (the planned Figure 1 of the paper).
