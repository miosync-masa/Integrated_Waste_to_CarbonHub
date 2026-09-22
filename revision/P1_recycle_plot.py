# -*- coding: utf-8 -*-
"""
Figure for P1: CO2-derived solid carbon vs purge fraction (reads the CSVs written by
P1_recycle_analysis.py; run after it).  ../.venv/bin/python P1_recycle_plot.py
Left: Tier 2 equilibrium (T3 = 650 K, r_CH4 = 0.95), one line per H2 split h.
Right: Tier 3 kinetic (uncalibrated; r_CH4 = 0.95), lines per phi at h = 0.35 and h = 1.0.
Dashed: Tier 1b stoichiometric asymptote (all fresh CO2 carbon = 11.46 t/d).
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
eq = pd.read_csv(os.path.join(RESULT, "P1_recycle_equilibrium_sweep.csv"))
kin = pd.read_csv(os.path.join(RESULT, "P1_recycle_kinetic_sweep.csv"))
t1 = pd.read_csv(os.path.join(RESULT, "P1_recycle_tier1_stoichiometric.csv"))
asym = float(t1.iloc[2]["C_from_CO2_tpd"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
d = eq[(eq.T3_K == 650) & (eq.r_CH4 == 0.95) & (eq.purge_p < 1)]
for h, g in d.groupby("h_H2_to_CFR"):
    g = g.sort_values("purge_p"); ax[0].plot(g.purge_p, g.C_from_CO2_total_tpd, marker="o", label=f"h = {h:g}")
ax[0].axhline(asym, ls="--", color="k", label=f"stoichiometric asymptote {asym:.2f} t/d")
ax[0].set_title("Tier 2: EQUILIBRIUM recycle (T3 = 650 K, r_CH4 = 0.95)"); ax[0].set_xscale("log")
ax[0].set_xlabel("purge fraction p"); ax[0].set_ylabel("CO2-derived solid carbon [t/d]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
d = kin[(kin.r_CH4 == 0.95) & (kin.purge_p < 1)]
for (phi, h), g in d[d.h_H2_to_CFR.isin([0.35, 1.0])].groupby(["phi", "h_H2_to_CFR"]):
    g = g.sort_values("purge_p"); ax[1].plot(g.purge_p, g.C_from_CO2_total_tpd, marker="o", ls="-" if h == 0.35 else ":", label=f"phi = {phi:g}, h = {h:g}")
ax[1].axhline(asym, ls="--", color="k")
ax[1].set_title("Tier 3: KINETIC recycle (uncalibrated; r_CH4 = 0.95, T3 = 650 K)"); ax[1].set_xscale("log")
ax[1].set_xlabel("purge fraction p"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("P1: CO2-derived solid carbon under recycle (Stage 3 as methanator)"); fig.tight_layout()
out = os.path.join(RESULT, "P1_recycle_co2_carbon.png"); fig.savefig(out, dpi=160); print("saved", out)
