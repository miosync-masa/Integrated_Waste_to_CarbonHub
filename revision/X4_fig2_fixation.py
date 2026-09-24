"""
X4_fig2_fixation.py -- Fig. 2 of the revised manuscript: fixation of biogas-CO2 carbon as solid carbon
in the X4 base configuration (2b: all biogas CO2 to Stage 2, y_CH4 = 0.60, no DAC, no electrolysis).

Bars per membrane H2 split h in {0.35, 0.50 (representative), 0.75}:
  * once-through, EQUILIBRIUM                      (F1_cases.csv, "2b_y0.60 eq once-through h=...")
  * converged recycle, EQUILIBRIUM                 (F1_cases.csv, "2b_y0.60 eq recycle h=...")
  * converged recycle, KINETIC design case, 1 atm  (solved here with F1_full_biogas_CO2.solve_recycle)
  * converged recycle, KINETIC, Stage 2/3 at 5 bar, pressure order n = 1 (T3b)     (solved here)
Kinetic = tau1* = 8.905 s, Stage 2 two-reaction model phi = 1 (tau2 = 3 s), Stage 3 reduced model
(650 K, tau3 = 3 s); rate constants UNCALIBRATED. Recycle: r_CH4 = 0.95, purge p = 0.05.
Fixation [%] = species-resolved net-conversion tracer (P1) / biogas CO2 carbon (23.96 t C/d).

Outputs (revision/Result/): X4_fig2_kinetic_cases.csv (cache of the six kinetic solves, F1 make_row
columns), X4_fig2_data.csv (plotted values), X4_fig2_fixation.png / .pdf, X4_fig2_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python X4_fig2_fixation.py  (~3 min first run)
"""
import os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import F1_full_biogas_CO2 as F1m

ATM, BAR = F1m.ATM, F1m.BAR
H_GRID = [0.35, 0.50, 0.75]; PHI = 1.0
KIN_CSV = os.path.join(RESULT, "X4_fig2_kinetic_cases.csv")

def kin_case(args):
    h, five_bar = args; t0 = time.time(); fd = F1m.feeds()["2b_y0.60"]
    P2 = P3 = (5 * BAR if five_bar else ATM); n = 1.0 if five_bar else None
    cfg = dict(config="2b_y0.60", feed=fd, mode="kinetic", phi=PHI, k2=F1m.k2_from_phi(PHI), n=n, variant="V1", P1_Pa=ATM, P2_Pa=P2, P3_Pa=P3, h=h, p=0.05, r=0.95)
    label = f"2b_y0.60 kin recycle phi={PHI:g} h={h:g} " + ("(1.01,5,5) bar n=1" if five_bar else "1 atm")
    rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label); row["wall_s"] = time.time() - t0; row["five_bar"] = five_bar; return row

def kinetic_cases():
    if os.path.exists(KIN_CSV): return pd.read_csv(KIN_CSV)
    with Pool(6) as pool: rows = pool.map(kin_case, [(h, fb) for h in H_GRID for fb in (False, True)])
    df = pd.DataFrame(rows); df.to_csv(KIN_CSV, index=False); return df

def main():
    t0 = time.time(); F1 = pd.read_csv(os.path.join(RESULT, "F1_cases.csv")); K = kinetic_cases()
    C_CO2 = float(F1[F1.config == "2b_y0.60"].CO2_total_carbon_tpd.iloc[0])          # 23.96 t C/d
    series = [("Once-through, equilibrium", lambda h: F1[F1.case == f"2b_y0.60 eq once-through h={h:g}"].iloc[0], "#9ecae1", ""),
              ("Recycle, equilibrium", lambda h: F1[F1.case == f"2b_y0.60 eq recycle h={h:g}"].iloc[0], "#08519c", ""),
              ("Recycle, kinetic design case, 1 atm (uncalibrated)", lambda h: K[(K.h_H2_to_CFR == h) & (~K.five_bar)].iloc[0], "#fdae6b", "//"),
              ("Recycle, kinetic, Stage 2/3 at 5 bar, n = 1 (uncalibrated)", lambda h: K[(K.h_H2_to_CFR == h) & (K.five_bar)].iloc[0], "#e6550d", "//")]
    rows = []
    for name, get, col, hatch in series:
        for h in H_GRID:
            r = get(h); rows.append(dict(series=name, h_H2_split=h, value_type=r.value_type, fixation_pct=100 * r.CO2_carbon_fixed_fraction,
                                         C_from_CO2_tpd=r.C_from_biogasCO2_tpd, C_total_tpd=r.C_total_tpd, C_stage1_tpd=r.C_stage1_tpd, C_stage3_tpd=r.C_stage3_tpd,
                                         converged=r.converged, iterations=r.iterations))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESULT, "X4_fig2_data.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=200)
    x = np.arange(len(H_GRID)); w = 0.2
    ax.axvspan(1 - 0.5, 1 + 0.5, color="0.93", zorder=0)
    ax.text(1, 121, "representative operating point (h = 0.50)", ha="center", va="top", fontsize=8, color="0.35")
    for i, (name, get, col, hatch) in enumerate(series):
        v = D[D.series == name].sort_values("h_H2_split")
        bars = ax.bar(x + (i - 1.5) * w, v.fixation_pct, w, label=name, color=col, hatch=hatch, edgecolor="k", linewidth=0.5, zorder=2)
        for b, pct, tpd in zip(bars, v.fixation_pct, v.C_from_CO2_tpd):
            ax.text(b.get_x() + b.get_width() / 2, pct + 1.0, f"{pct:.1f}%\n({tpd:.1f})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"h = {h:.2f}" for h in H_GRID]); ax.set_xlabel("membrane H$_2$ split to Stage 3, h")
    ax.set_ylabel("biogas-CO$_2$ carbon fixed as solid carbon [%]"); ax.set_ylim(0, 124); ax.set_yticks(range(0, 101, 20))
    ax.axhline(100, color="k", lw=0.8, ls="--", zorder=1); ax.text(-0.58, 100.8, f"100 % = all biogas CO$_2$ carbon, {C_CO2:.1f} t C/d", ha="left", va="bottom", fontsize=7.5)
    ax2 = ax.twinx(); ax2.set_ylim(0, 1.24 * C_CO2); ax2.set_ylabel("CO$_2$-derived solid carbon [t/d]")
    ax.legend(loc="upper center", fontsize=7.5, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.13))
    ax.set_title("Fixation of biogas CO$_2$ as solid carbon, X4 configuration\n(all biogas CO$_2$ to Stage 2, $y_{CH_4}$ = 0.60, no DAC, no electrolysis; recycle r$_{CH_4}$ = 0.95, purge 0.05)", fontsize=9.5)
    fig.text(0.5, 0.012, "Equilibrium: Gibbs minimisation at every stage.  Kinetic: design case $\\tau_1^*$ = 8.9 s, Stage 2 two-reaction model ($\\varphi$ = 1), Stage 3 reduced model at 650 K;\n"
             "rate constants uncalibrated.  Bar labels: fixation % (CO$_2$-derived solid carbon, t C/d).", fontsize=6.5, color="0.3", ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(RESULT, "X4_fig2_fixation.png")); fig.savefig(os.path.join(RESULT, "X4_fig2_fixation.pdf"))
    pd.set_option("display.width", 200)
    L = [f"X4 Fig. 2 data (wall {time.time()-t0:.0f} s). Fixation = CO2-derived solid carbon / biogas CO2 carbon ({C_CO2:.2f} t C/d). value_type column labels equilibrium vs kinetic (uncalibrated).",
         D.to_string(index=False, float_format=lambda v: f"{v:.2f}"), "",
         "Kinetic phi range at h = 0.50, 1 atm (E7_kinetic_X4.csv): see that file; phi = 1 is the design case used here and in E7."]
    open(os.path.join(RESULT, "X4_fig2_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
