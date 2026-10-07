"""
Aggregate the full simulation groups and build manuscript figures/tables.

Inputs : results/full_sim_{main1,main2,extra}.csv
Outputs: figures/fig7_naxis.{png,pdf}, figures/fig8_paxis_nu.{png,pdf},
         manuscript/sim_tables/*.tex, results/summary_full_sim.csv

Run:  python3 -u experiments/make_full_sim_figs.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "manuscript", "sim_tables")
os.makedirs(TAB, exist_ok=True)

MLAB = {"fglasso": "fGLasso (Gaussian)",
        "t_em": "f t-EM (model a)",
        "altt_em": "f alt-t EM (model b)",
        "wglasso": "winsorize + glasso",
        "t_em_prof": "f t-EM, profiled $\\nu$"}
MCOL = {"fglasso": "tab:blue", "t_em": "tab:orange",
        "altt_em": "tab:green", "wglasso": "tab:red",
        "t_em_prof": "tab:purple"}
MMK = {"fglasso": "o", "t_em": "s", "altt_em": "^",
       "wglasso": "D", "t_em_prof": "v"}


def load():
    frames = []
    for g in ["main1", "main2", "extra"]:
        p = os.path.join(RES, f"full_sim_{g}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    return df


def agg(df, keys):
    g = df.groupby(keys)[["f1", "op_err", "jaccard", "n_edges", "fit_s"]]
    out = g.agg(["mean", "std"]).round(3)
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def fig_naxis(df):
    """One panel per method: F1 vs eps, lines = sample size."""
    base = df[df.cell.isin(["n100", "base", "n400"])].copy()
    nmap = {"n100": 100, "base": 200, "n400": 400}
    base["n"] = base.cell.map(nmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, m in zip(axes, ["fglasso", "t_em", "altt_em"]):
        sub = base[base.method == m]
        for n, ls in [(100, ":"), (200, "-"), (400, "--")]:
            g = sub[sub.n == n].groupby("eps")["f1"]
            ax.errorbar(g.mean().index, g.mean().values,
                        yerr=g.std().values, marker=MMK[m], ls=ls,
                        color=MCOL[m], capsize=3,
                        label=f"$n={n}$")
        ax.set_title(MLAB[m], fontsize=10)
        ax.set_xlabel("contamination level $\\epsilon$")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("edge F1")
    axes[0].legend(fontsize=8, title="sample size", title_fontsize=8)
    fig.suptitle("Sample-size axis ($p=12$, $K=4$, amplitude contamination)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIGS, f"fig7_naxis.{ext}"), dpi=300)
    plt.close(fig)


def fig_paxis_nu(df):
    """(a) F1 vs p at eps=0.10; (b) nu table: F1 vs nu at eps 0 and 0.10."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    dim = df[df.cell.isin(["base", "p20", "p50"]) &
             (df.eps == 0.10)].copy()
    pmap = {"base": 12, "p20": 20, "p50": 50}
    dim["p"] = dim.cell.map(pmap)
    for m in ["fglasso", "t_em", "altt_em", "wglasso"]:
        sub = dim[dim.method == m]
        if not len(sub):
            continue
        g = sub.groupby("p")["f1"]
        ax.plot(g.mean().index, g.mean().values, marker=MMK[m],
                color=MCOL[m], label=MLAB[m])
        ax.fill_between(g.mean().index,
                        (g.mean() - g.std()).clip(lower=0),
                        (g.mean() + g.std()), color=MCOL[m], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks([12, 20, 50])
    ax.set_xticklabels(["12", "20", "50"])
    ax.set_xlabel("number of functions $p$  ($n=200$, $\\epsilon=0.10$)")
    ax.set_ylabel("edge F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    nu_cells = df[df.cell.str.startswith("nu")].copy()
    nu_cells["nu_plot"] = nu_cells.nu
    fixed = nu_cells[nu_cells.method == "t_em"]
    for eps, mk in [(0.0, "o"), (0.10, "s")]:
        g = fixed[fixed.eps == eps].groupby("nu_plot")["f1"]
        ax.plot(g.mean().index, g.mean().values, marker=mk, ls="-",
                color="tab:orange", label=f"model (a), fixed $\\nu$, "
                                          f"$\\epsilon={eps}$")
    prof = nu_cells[nu_cells.method == "t_em_prof"]
    for eps, mk in [(0.0, "o"), (0.10, "s")]:
        g = prof[prof.eps == eps]["f1"]
        if len(g):
            ax.axhline(g.mean(), ls=":", color="tab:purple",
                       label=f"model (a), profiled $\\nu$, $\\epsilon={eps}$")
    altt = nu_cells[(nu_cells.method == "altt_em") & (nu_cells.eps == 0.10)]
    if len(altt):
        ax.axhline(altt["f1"].mean(), ls="--", color="tab:green",
                   label="model (b), $\\epsilon=0.10$")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 7, 10, 15])
    ax.set_xticklabels(["3", "5", "7", "10", "15"])
    ax.minorticks_off()
    ax.set_xlabel("degrees of freedom $\\nu$")
    ax.set_ylabel("edge F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIGS, f"fig8_paxis_nu.{ext}"), dpi=300)
    plt.close(fig)


def write_tables(df, summ):
    # dimension axis table at eps = 0.10
    def tab(df_sub, row_keys, caption, fname):
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Cell & " + " & ".join(MLAB[m] for m in
                                          ["fglasso", "t_em", "altt_em",
                                           "wglasso"]) + r" \\", r"\midrule"]
        for rk in row_keys:
            vals = []
            for m in ["fglasso", "t_em", "altt_em", "wglasso"]:
                sel = df_sub[(df_sub.method == m)]
                row = sel[sel[row_keys[0]] == rk[1]] if len(rk) > 1 else sel
                if len(row):
                    r0 = row.iloc[0]
                    vals.append(f"{r0.f1_mean:.2f} ({r0.jaccard_mean:.2f})")
                else:
                    vals.append("--")
            lines.append(rk[0] + " & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(TAB, fname), "w") as f:
            f.write("% AUTO-GENERATED\n" + "\n".join(lines) + "\n")

    eps01 = summ[summ.eps == 0.10]
    # n axis
    n_rows = [("n=100", "n100"), ("n=200", "base"), ("n=400", "n400")]
    s = eps01[eps01.cell.isin([r[1] for r in n_rows])].copy()
    # need per-cell grouping: pivot manually
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"$n$ & fGLasso & f t-EM (a) & f alt-t EM (b) & winsorize+lasso \\",
             r"\midrule"]
    for lab, cell in n_rows:
        vals = []
        for m in ["fglasso", "t_em", "altt_em", "wglasso"]:
            row = eps01[(eps01.cell == cell) & (eps01.method == m)]
            vals.append(f"{row.f1_mean.iloc[0]:.2f} "
                        f"({row.jaccard_mean.iloc[0]:.2f})" if len(row) else "--")
        lines.append(f"${lab[2:]}$ & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_naxis.tex"), "w").write(
        "% AUTO-GENERATED\n" + "\n".join(lines) + "\n")

    # p axis
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"$p$ & fGLasso & f t-EM (a) & f alt-t EM (b) & winsorize+lasso \\",
             r"\midrule"]
    for lab, cell in [(12, "base"), (20, "p20"), (50, "p50")]:
        vals = []
        for m in ["fglasso", "t_em", "altt_em", "wglasso"]:
            row = eps01[(eps01.cell == cell) & (eps01.method == m)]
            vals.append(f"{row.f1_mean.iloc[0]:.2f} "
                        f"({row.jaccard_mean.iloc[0]:.2f})" if len(row) else "--")
        lines.append(f"${lab}$ & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_paxis.tex"), "w").write(
        "% AUTO-GENERATED\n" + "\n".join(lines) + "\n")

    # nu table
    nu_cells = df[df.cell.str.startswith("nu")]
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Estimator & F1 ($\epsilon=0$) & F1 ($\epsilon=0.10$) \\",
             r"\midrule"]
    for nu in [3.0, 5.0, 7.0, 10.0, 15.0]:
        sub = nu_cells[(nu_cells.method == "t_em") & (nu_cells.nu == nu)]
        v0 = sub[sub.eps == 0.0]["f1"].mean()
        v1 = sub[sub.eps == 0.10]["f1"].mean()
        lines.append(f"model (a), $\\nu={int(nu)}$ & {v0:.2f} & {v1:.2f} \\\\")
    sub = nu_cells[(nu_cells.method == "t_em_prof")]
    v0 = sub[sub.eps == 0.0]["f1"].mean()
    v1 = sub[sub.eps == 0.10]["f1"].mean()
    vh = sub[sub.eps == 0.0]["nu_hat"].mean()
    lines.append(rf"\midrule model (a), profiled $\nu$ (fit $\approx {vh:.0f}$)"
                 rf" & {v0:.2f} & {v1:.2f} \\")
    sub = nu_cells[(nu_cells.method == "altt_em")]
    v0 = sub[sub.eps == 0.0]["f1"].mean()
    v1 = sub[sub.eps == 0.10]["f1"].mean()
    lines.append(f"model (b), $\\nu=7$ & {v0:.2f} & {v1:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_nu.tex"), "w").write(
        "% AUTO-GENERATED\n" + "\n".join(lines) + "\n")

    # mechanism table
    mech = summ[summ.cell.str.startswith("mech")]
    mech = mech[mech.eps.isin([0.10])]
    mech = mech.copy()
    mech["mech"] = mech.cell.str.replace("mech_", "", regex=False)
    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Mechanism & Estimator & F1 & Instability \\", r"\midrule"]
    for mech_name in ["amplitude", "blink"]:
        for m in ["fglasso", "t_em", "altt_em", "wglasso"]:
            row = mech[(mech.mech == mech_name) & (mech.method == m)]
            if len(row):
                lines.append(f"{mech_name} & {MLAB[m]} & "
                             f"{row.f1_mean.iloc[0]:.2f} & "
                             f"{row.jaccard_mean.iloc[0]:.2f} \\\\")
        if mech_name == "amplitude":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_mech.tex"), "w").write(
        "% AUTO-GENERATED\n" + "\n".join(lines) + "\n")


def main():
    df = load()
    print("rows:", len(df), "| cells:", sorted(df.cell.unique()))
    summ = agg(df, ["cell", "method", "eps"])
    summ.to_csv(os.path.join(RES, "summary_full_sim.csv"), index=False)
    fig_naxis(df)
    fig_paxis_nu(df)
    write_tables(df, summ)
    print("wrote fig7, fig8, sim_tables/*.tex, summary_full_sim.csv")


if __name__ == "__main__":
    main()
