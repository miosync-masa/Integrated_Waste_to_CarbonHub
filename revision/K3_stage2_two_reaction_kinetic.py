# -*- coding: utf-8 -*-
"""
K3 (part 2/2): Stage 2 reduced kinetic model with CO2 methanation added
=======================================================================
Revision task K3 for FUPROC-D-26-00269 (reviewer comments R1-3, R2-4, R4-2).

The submitted reduced kinetic model of Stage 2 (Workflow_cantera.py,
run_stage2_rwgs_kinetic) contains ONE reversible reaction (RWGS) and treats CH4
as inert. Its large-tau plateau is therefore the RWGS-only equilibrium, not the
Gibbs equilibrium that includes CH4 formation. This script adds a second
reversible reaction and re-runs the tau sweep at 950 K, 1 atm.

Reaction network (5 species, 3 elements -> 2 independent reactions)
  R1  RWGS            : CO2 +  H2 <=> CO  +  H2O
  R2  CO2 methanation : CO2 + 4H2 <=> CH4 + 2H2O
Rate laws (thermodynamically consistent: equilibrium of each reaction is
enforced by Kp/Kc from GRI-3.0 thermo, so the tau -> infinity limit is the
5-species chemical equilibrium irrespective of the rate constants)
  r1 = k1 * ( c_CO2 c_H2 - c_CO c_H2O / Kc1 )                     [mol m-3 s-1]
       k1 = 0.05 m3 mol-1 s-1 at 950 K  (identical to the submitted model)
  r2 = k2 * ( a_CO2 a_H2^4 - a_CH4 a_H2O^2 / Kp2 ),  a_i = y_i P/P0  [mol m-3 s-1]
       same elementary power-law form as the CO2-methanation step already used in
       the Stage 3 reduced model (run_stage3_cfr_kinetic), for internal consistency.
  k2 is UNCALIBRATED. It is parametrised by the dimensionless ratio
       phi = r2,fwd(inlet) / r1,fwd(inlet)
  i.e. the methanation-to-RWGS forward-rate ratio at reactor inlet conditions,
  swept over phi = 0 (submitted 1-reaction model), 0.01, 0.1, 1, 10.
  No literature calibration is claimed for either k1 or k2; the sweep answers
  only the structural question "does the 2-reaction model approach the Gibbs
  equilibrium, and how does the stage duty Q2 change".
Reactor: isothermal, isobaric plug flow integrated in reactor volume
  V = tau * vdot_inlet (nominal residence time defined at inlet conditions, as in
  the submitted model). Because R2 reduces the mole number (5 -> 3), the true
  residence time is slightly longer than the nominal tau when R2 is active.
  Integration: scipy solve_ivp (BDF, rtol 1e-10). The submitted model used an
  explicit Euler scheme (400 steps); the phi = 0 case is compared against the
  submitted CSV to quantify the integration difference.

Two feed definitions (see K3_stage2_extent_decomposition.py):
  Feed A "EQ chain" : Stage 1 Gibbs at 1200 K  (main-text baseline, Q2 = -196 kW)
  Feed B "KIN chain": Stage 1 kinetic tau=3 s  (feed of validation_stage2_rwgs_950K.csv)

All results in this script are KINETIC values, except the columns/lines marked
"Gibbs" or "equilibrium".

Reproducibility
---------------
  cd <repo>/revision
  ../.venv/bin/python K3_stage2_two_reaction_kinetic.py
Requires cantera>=3.0, numpy, scipy, pandas, matplotlib. Imports Workflow_cantera.py
read-only for feeds and the Gibbs reference. Baseline outputs are NOT touched.

Outputs (written to revision/Result/):
  K3_stage2_2rxn_tau_sweep.csv     full sweep (feed x phi x tau)
  K3_stage2_2rxn_tau3s_table.csv   tau = 3 s slice (design point) for the paper
  K3_stage2_2rxn_summary.txt       summary incl. verification against baseline
  K3_stage2_2rxn_tau_sweep.png     X_CO2 and Q2 vs tau
"""
import os, sys
import numpy as np
import pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W

OUT_CSV   = os.path.join(RESULT, "K3_stage2_2rxn_tau_sweep.csv")
OUT_TAU3  = os.path.join(RESULT, "K3_stage2_2rxn_tau3s_table.csv")
OUT_TXT   = os.path.join(RESULT, "K3_stage2_2rxn_summary.txt")
OUT_PNG   = os.path.join(RESULT, "K3_stage2_2rxn_tau_sweep.png")
BASELINE_VAL_CSV = os.path.join(BASE, "CanteraResult", "validation_stage2_rwgs_950K.csv")

T2, P2 = W.T_rwgs, W.P_rwgs                 # 950 K, 1 atm
R = ct.gas_constant / 1000.0                # J mol-1 K-1
P0 = ct.one_atm
SP = ["CO2", "H2", "CO", "H2O", "CH4"]
IDX = {s: i for i, s in enumerate(SP)}
NU1 = np.array([-1, -1, +1, +1, 0], float)  # RWGS
NU2 = np.array([-1, -4, 0, +2, +1], float)  # CO2 methanation
K1_REF = W.rwgs_kref                        # 0.05 m3/mol/s at 950 K (submitted value)
PHI_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]      # uncalibrated sweep variable
TAU_GRID = list(W.rwgs_tau_grid)            # 0.1 ... 100 s (same grid as submitted)
KMOLD_TO_KW = 1000.0 / 86400.0

def thermo(T):
    g = ct.Solution("gri30.yaml"); g.TP = T, P0
    gRT = g.standard_gibbs_RT; hRT = g.standard_enthalpies_RT
    i = [g.species_index(s) for s in SP]
    Kp1 = float(np.exp(-NU1 @ gRT[i])); Kp2 = float(np.exp(-NU2 @ gRT[i]))
    Kc1 = Kp1                                   # dn = 0
    dH1 = float(NU1 @ hRT[i]) * ct.gas_constant * T / 1e6   # kJ/mol
    dH2 = float(NU2 @ hRT[i]) * ct.gas_constant * T / 1e6
    return Kp1, Kc1, Kp2, dH1, dH2

def feeds():
    s1a = W.run_stage1_pyrolysis(W.CH4_tpd, W.T_pyro, W.P_pyro)
    fa, _ = W.build_rwgs_feed(s1a, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    s1b = W.run_stage1_pyrolysis_kinetic(W.CH4_tpd, W.T_pyro, W.P_pyro,
            tau_s=W.pyro_tau_s, n_steps=W.pyro_n_steps, kref=W.pyro_kref, Ea=W.pyro_Ea, Tref=W.pyro_Tref)
    fb, _ = W.build_rwgs_feed(s1b, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    return {"A_eq_chain": fa, "B_kin_chain": fb}

def gibbs_reference(feed):
    s2 = W.run_stage2_rwgs(feed, T2, P2)
    out = s2["result"]["gas_kmol_d"]
    X = 1 - out.get("CO2", 0) / feed["CO2"]
    return X, s2["Q_kW"], out.get("CH4", 0), out.get("CO", 0)

def rwgs_only_X(feed, Kc1):
    a, b = feed.get("CO2", 0.0), feed.get("H2", 0.0)
    A = 1 - Kc1; B = Kc1 * (a + b); C = -Kc1 * a * b
    x = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    return x / a

def run_pfr(feed, tau, k1, k2, Kc1, Kp2, T, P):
    """Integrate F_i(V) [mol/s] for the 5 species; returns outlet flows and extents."""
    F0 = np.array([feed.get(s, 0.0) for s in SP]) * 1000.0 / 86400.0
    Ftot0 = F0.sum()
    vdot0 = Ftot0 * R * T / P
    V = tau * vdot0
    cT = P / (R * T)                       # total concentration mol/m3
    def rhs(v, F):
        F = np.maximum(F, 0.0)
        y = F / F.sum()
        c = y * cT
        a = y * (P / P0)
        r1 = k1 * (c[0] * c[1] - c[2] * c[3] / Kc1)
        r2 = k2 * (a[0] * a[1] ** 4 - a[4] * a[3] ** 2 / Kp2)
        return NU1 * r1 + NU2 * r2
    sol = solve_ivp(rhs, (0.0, V), F0, method="BDF", rtol=1e-10, atol=1e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    F = np.maximum(sol.y[:, -1], 0.0)
    dF = F - F0
    xi1 = dF[IDX["CO"]]; xi2 = dF[IDX["CH4"]]
    conv = 86400.0 / 1000.0
    return {s: F[IDX[s]] * conv for s in SP}, xi1 * conv, xi2 * conv

def k2_from_phi(phi, feed, k1, T, P):
    F0 = np.array([feed.get(s, 0.0) for s in SP]); y = F0 / F0.sum()
    cT = P / (R * T); c = y * cT; a = y * (P / P0)
    r1f = k1 * c[0] * c[1]
    r2f_unit = a[0] * a[1] ** 4
    return phi * r1f / r2f_unit

def main():
    Kp1, Kc1, Kp2, dH1, dH2 = thermo(T2)
    k1 = K1_REF
    fd = feeds()
    rows, txt = [], []
    txt.append("K3 Stage 2 two-reaction reduced kinetic model: tau sweep at 950 K, 1 atm")
    txt.append(f"Kp1(RWGS) = {Kp1:.4f}, Kp2(CO2 methanation) = {Kp2:.4e}; dH1 = {dH1:+.2f}, dH2 = {dH2:+.2f} kJ/mol (950 K, GRI-3.0)")
    txt.append(f"k1 = {k1} m3/mol/s (submitted value, uncalibrated). k2 set via phi = r2,fwd/r1,fwd at inlet; UNCALIBRATED.")
    txt.append("Kinetic values unless marked Gibbs/equilibrium.\n")
    for tag, feed in fd.items():
        Xg, Qg, CH4g, COg = gibbs_reference(feed)
        Xr1 = rwgs_only_X(feed, Kc1)
        txt.append(f"===== Feed {tag} =====")
        txt.append("feed [kmol/d]: " + ", ".join(f"{s} {feed.get(s,0):.1f}" for s in SP if feed.get(s,0) > 0))
        txt.append(f"Gibbs equilibrium reference : X_CO2 = {Xg:.4f}, Q2 = {Qg:+.1f} kW, CH4_out = {CH4g:.1f}, CO_out = {COg:.1f} kmol/d")
        txt.append(f"RWGS-only equilibrium (plateau of submitted 1-rxn model): X_CO2 = {Xr1:.4f}")
        for phi in PHI_GRID:
            k2 = k2_from_phi(phi, feed, k1, T2, P2)
            for tau in TAU_GRID:
                out, xi1, xi2 = run_pfr(feed, tau, k1, k2, Kc1, Kp2, T2, P2)
                X = 1 - out["CO2"] / feed["CO2"]
                Q1 = xi1 * dH1 * KMOLD_TO_KW; Q2m = xi2 * dH2 * KMOLD_TO_KW
                rows.append({"feed_case": tag, "phi_meth_to_rwgs_inlet_rate_ratio": phi,
                             "k2_uncalibrated_mol_m3_s": k2, "tau_s": tau,
                             "X_CO2_kinetic": X, "X_CO2_gibbs_equilibrium": Xg,
                             "X_CO2_rwgs_only_equilibrium": Xr1,
                             "approach_to_gibbs": X / Xg,
                             "xi1_RWGS_kmol_d": xi1, "xi2_CO2_methanation_kmol_d": xi2,
                             "CO_out_kmol_d": out["CO"], "CH4_out_kmol_d": out["CH4"],
                             "H2O_out_kmol_d": out["H2O"], "H2_out_kmol_d": out["H2"], "CO2_out_kmol_d": out["CO2"],
                             "Q_RWGS_kW": Q1, "Q_meth_kW": Q2m, "Q2_total_kinetic_kW": Q1 + Q2m,
                             "Q2_gibbs_equilibrium_kW": Qg})
        df_t = pd.DataFrame([r for r in rows if r["feed_case"] == tag])
        for phi in PHI_GRID:
            d = df_t[df_t.phi_meth_to_rwgs_inlet_rate_ratio == phi]
            r3 = d[np.isclose(d.tau_s, 3.0)].iloc[0]; r100 = d[np.isclose(d.tau_s, 100.0)].iloc[0]
            txt.append(f"phi = {phi:>5}: tau=3 s -> X_CO2 {r3.X_CO2_kinetic:.4f}, Q2 {r3.Q2_total_kinetic_kW:+7.1f} kW, CH4_out {r3.CH4_out_kmol_d:7.1f} | "
                       f"tau=100 s -> X_CO2 {r100.X_CO2_kinetic:.4f} (X/X_Gibbs {r100.approach_to_gibbs:.4f}), Q2 {r100.Q2_total_kinetic_kW:+7.1f} kW, CH4_out {r100.CH4_out_kmol_d:7.1f}")
        txt.append("")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    df[np.isclose(df.tau_s, 3.0)].to_csv(OUT_TAU3, index=False)

    # Verification: phi = 0 on Feed B must reproduce the submitted 1-reaction series
    if os.path.exists(BASELINE_VAL_CSV):
        base = pd.read_csv(BASELINE_VAL_CSV)
        mine = df[(df.feed_case == "B_kin_chain") & (df.phi_meth_to_rwgs_inlet_rate_ratio == 0.0)].set_index("tau_s")
        txt.append("===== Verification: phi = 0 (1 reaction) on Feed B vs submitted CanteraResult/validation_stage2_rwgs_950K.csv =====")
        txt.append("tau_s   X_submitted(Euler,400 steps)   X_this(BDF)   diff        Q_submitted   Q_this   diff_kW")
        maxd = 0.0
        for _, b in base.iterrows():
            m = mine.loc[b.tau_s]
            d = m.X_CO2_kinetic - b.X_CO2_kinetic; maxd = max(maxd, abs(d))
            txt.append(f"{b.tau_s:5.1f}   {b.X_CO2_kinetic:.6f}                     {m.X_CO2_kinetic:.6f}    {d:+.2e}   {b.Q_kW:8.2f}   {m.Q2_total_kinetic_kW:8.2f}   {m.Q2_total_kinetic_kW-b.Q_kW:+.3f}")
        txt.append(f"max |dX| = {maxd:.2e} (integration-scheme difference only; same rate law and k1)\n")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j, tag in enumerate(fd):
        d_all = df[df.feed_case == tag]
        Xg = d_all.X_CO2_gibbs_equilibrium.iloc[0]; Xr1 = d_all.X_CO2_rwgs_only_equilibrium.iloc[0]; Qg = d_all.Q2_gibbs_equilibrium_kW.iloc[0]
        ax = axes[0, j]
        for phi in PHI_GRID:
            d = d_all[d_all.phi_meth_to_rwgs_inlet_rate_ratio == phi]
            ax.plot(d.tau_s, d.X_CO2_kinetic, marker="o", ms=3, label=f"phi = {phi:g}" + (" (submitted 1-rxn)" if phi == 0 else ""))
        ax.axhline(Xg, color="k", ls="--", label=f"Gibbs eq. {Xg:.3f}")
        ax.axhline(Xr1, color="gray", ls=":", label=f"RWGS-only eq. {Xr1:.3f}")
        ax.set_xscale("log"); ax.set_ylabel("X_CO2 [-] (kinetic)"); ax.set_title(f"Feed {tag}: CO2 conversion, 950 K")
        ax.legend(fontsize=7)
        ax = axes[1, j]
        for phi in PHI_GRID:
            d = d_all[d_all.phi_meth_to_rwgs_inlet_rate_ratio == phi]
            ax.plot(d.tau_s, d.Q2_total_kinetic_kW, marker="o", ms=3, label=f"phi = {phi:g}")
        ax.axhline(Qg, color="k", ls="--", label=f"Gibbs eq. {Qg:+.0f} kW")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xscale("log"); ax.set_xlabel("nominal residence time tau [s]"); ax.set_ylabel("Q2 [kW] (kinetic; +endothermic)")
        ax.set_title(f"Feed {tag}: Stage 2 duty"); ax.legend(fontsize=7)
    fig.suptitle("K3: Stage 2 reduced model with CO2 methanation added (k2 uncalibrated, parametrised by phi)")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=160); plt.close(fig)

    with open(OUT_TXT, "w") as fh: fh.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"Saved: {OUT_CSV}\n       {OUT_TAU3}\n       {OUT_TXT}\n       {OUT_PNG}")

if __name__ == "__main__":
    main()
