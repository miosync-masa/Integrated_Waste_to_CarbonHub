# -*- coding: utf-8 -*-
"""
K3 (part 1/2): Stage 2 reaction-extent decomposition and heat apportionment
===========================================================================
Revision task K3 for FUPROC-D-26-00269 (reviewer comments R1-3, R2-4, R4-2).

What this script does
---------------------
1. Rebuilds the Stage 2 (RWGS reactor, 950 K, 1 atm) Gibbs-equilibrium result
   for two feed definitions that both appear in the submitted material:
     Feed A ("EQ chain")  : Stage 1 = Gibbs equilibrium at 1200 K
                            -> main-text baseline (Q2 = -196 kW, CH4_in = 90.3 kmol/d)
     Feed B ("KIN chain") : Stage 1 = reduced kinetic model, tau = 3 s
                            -> feed used by CanteraResult/validation_stage2_rwgs_950K.csv
                               (X_CO2,eq = 0.818 in that file)
2. Decomposes the equilibrium composition change into two independent extents
     R1  RWGS            : CO2 +  H2 -> CO  +  H2O      xi1 = Delta n(CO)
     R2  CO2 methanation : CO2 + 4H2 -> CH4 + 2H2O      xi2 = Delta n(CH4)
   (5 species CO2/H2/CO/H2O/CH4, 3 elements -> exactly 2 independent reactions)
   and checks that the predicted Delta n(CO2), Delta n(H2), Delta n(H2O) close
   against the Gibbs result (residual = trace species in GRI-3.0 only).
3. Apportions the isothermal reactor duty Q2 = sum_i xi_i * DeltaH_i(950 K), with
   DeltaH_i from Cantera (GRI-3.0 thermo), and compares with the directly computed
   enthalpy difference used in the baseline code.

All numbers here are EQUILIBRIUM values (Gibbs minimisation), not kinetic values.

Reproducibility
---------------
  cd <repo>/revision
  ../.venv/bin/python K3_stage2_extent_decomposition.py
Requires cantera>=3.0, numpy, pandas. Imports the baseline module
Workflow_cantera.py (read-only; its __main__ block is not executed) so that the
feeds and equilibrium calls are bit-identical to the submitted baseline.
Baseline outputs in CanteraResult/ are NOT touched.

Outputs (written to revision/Result/):
  K3_extent_decomposition.csv   per-feed streams, extents, closure residuals
  K3_heat_apportionment.csv     DeltaH_i, xi_i, Q_i, sum vs direct Q2
  K3_extent_summary.txt         human-readable summary
"""
import os, sys
import numpy as np
import pandas as pd
import cantera as ct

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W   # baseline module (no side effects except thermo loading)

OUT_CSV_EXT  = os.path.join(RESULT, "K3_extent_decomposition.csv")
OUT_CSV_HEAT = os.path.join(RESULT, "K3_heat_apportionment.csv")
OUT_TXT      = os.path.join(RESULT, "K3_extent_summary.txt")

T2, P2 = W.T_rwgs, W.P_rwgs          # 950 K, 1 atm
SPECIES = ["CO2", "H2", "CO", "H2O", "CH4"]
REACTIONS = {
    "R1_RWGS":            {"CO2": -1, "H2": -1, "CO": +1, "H2O": +1},
    "R2_CO2_methanation": {"CO2": -1, "H2": -4, "CH4": +1, "H2O": +2},
}
KMOLD_TO_KW = 1000.0 / 86400.0       # (kmol/d * kJ/mol) -> kW

# ------------------------------------------------------------------
# Feeds (identical construction to the baseline code)
# ------------------------------------------------------------------
def feed_A_eq_chain():
    s1 = W.run_stage1_pyrolysis(W.CH4_tpd, W.T_pyro, W.P_pyro)
    rf, h2s = W.build_rwgs_feed(s1, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    return rf, s1

def feed_B_kin_chain():
    s1 = W.run_stage1_pyrolysis_kinetic(W.CH4_tpd, W.T_pyro, W.P_pyro,
            tau_s=W.pyro_tau_s, n_steps=W.pyro_n_steps,
            kref=W.pyro_kref, Ea=W.pyro_Ea, Tref=W.pyro_Tref)
    rf, h2s = W.build_rwgs_feed(s1, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    return rf, s1

# ------------------------------------------------------------------
# Thermo helpers
# ------------------------------------------------------------------
def reaction_dH_kJmol(T, stoich):
    """Standard reaction enthalpy at T from GRI-3.0 (ideal gas => equals actual)."""
    g = ct.Solution("gri30.yaml"); g.TP = T, ct.one_atm
    hRT = g.standard_enthalpies_RT
    return sum(nu * hRT[g.species_index(sp)] for sp, nu in stoich.items()) * ct.gas_constant * T / 1e6

def reaction_Kp(T, stoich):
    g = ct.Solution("gri30.yaml"); g.TP = T, ct.one_atm
    gRT = g.standard_gibbs_RT
    return float(np.exp(-sum(nu * gRT[g.species_index(sp)] for sp, nu in stoich.items())))

def rwgs_only_equilibrium_X(feed, T):
    """X_CO2 if ONLY R1 were allowed (CH4 inert). Plateau of the 1-reaction reduced model."""
    K = reaction_Kp(T, REACTIONS["R1_RWGS"])
    a, b = feed.get("CO2", 0.0), feed.get("H2", 0.0)
    c0, d0 = feed.get("CO", 0.0), feed.get("H2O", 0.0)
    # (c0+x)(d0+x) = K (a-x)(b-x)
    A = 1 - K; B = c0 + d0 + K * (a + b); C = c0 * d0 - K * a * b
    x = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A) if abs(A) > 1e-12 else -C / B
    return x / a, x

# ------------------------------------------------------------------
def analyse(tag, feed, s1):
    s2 = W.run_stage2_rwgs(feed, T2, P2)
    out = s2["result"]["gas_kmol_d"]
    n_in  = {sp: feed.get(sp, 0.0) for sp in SPECIES}
    n_out = {sp: out.get(sp, 0.0) for sp in SPECIES}
    dn = {sp: n_out[sp] - n_in[sp] for sp in SPECIES}
    xi = {"R1_RWGS": dn["CO"], "R2_CO2_methanation": dn["CH4"]}
    pred = {sp: sum(xi[r] * REACTIONS[r].get(sp, 0) for r in REACTIONS) for sp in SPECIES}
    resid = {sp: dn[sp] - pred[sp] for sp in SPECIES}
    trace = {k: v for k, v in out.items() if k not in SPECIES and v > 1e-9}
    X = -dn["CO2"] / n_in["CO2"]
    X_r1only, _ = rwgs_only_equilibrium_X(feed, T2)

    dH = {r: reaction_dH_kJmol(T2, st) for r, st in REACTIONS.items()}
    Q = {r: xi[r] * dH[r] * KMOLD_TO_KW for r in REACTIONS}
    Q_sum = sum(Q.values())
    Q_direct = s2["Q_kW"]

    rows_ext = []
    for sp in SPECIES:
        rows_ext.append({"feed_case": tag, "species": sp,
                         "n_in_kmol_d": n_in[sp], "n_out_kmol_d": n_out[sp],
                         "delta_n_kmol_d": dn[sp], "delta_n_predicted_from_xi": pred[sp],
                         "closure_residual_kmol_d": resid[sp]})
    rows_heat = []
    for r in REACTIONS:
        rows_heat.append({"feed_case": tag, "reaction": r,
                          "stoichiometry": " ".join(f"{v:+d}{k}" for k, v in REACTIONS[r].items()),
                          "dH_950K_kJ_per_mol": dH[r], "xi_kmol_d": xi[r],
                          "Q_kW": Q[r], "share_of_net_Q2": Q[r] / Q_sum})
    rows_heat.append({"feed_case": tag, "reaction": "SUM(xi_i*dH_i)", "stoichiometry": "",
                      "dH_950K_kJ_per_mol": np.nan, "xi_kmol_d": np.nan,
                      "Q_kW": Q_sum, "share_of_net_Q2": 1.0})
    rows_heat.append({"feed_case": tag, "reaction": "Q2_direct_enthalpy_difference(baseline method)",
                      "stoichiometry": "", "dH_950K_kJ_per_mol": np.nan, "xi_kmol_d": np.nan,
                      "Q_kW": Q_direct, "share_of_net_Q2": np.nan})

    lines = []
    lines.append(f"===== Feed case {tag} =====")
    lines.append(f"Stage 1 X_CH4 = {1 - s1['result']['gas_kmol_d'].get('CH4',0)/s1['feed']['CH4']:.4f}"
                 f"  (CH4 residual {s1['result']['gas_kmol_d'].get('CH4',0):.1f} kmol/d)")
    lines.append("Stage 2 feed  [kmol/d]: " + ", ".join(f"{sp} {n_in[sp]:.1f}" for sp in SPECIES if n_in[sp] > 0))
    lines.append("Stage 2 out   [kmol/d]: " + ", ".join(f"{sp} {n_out[sp]:.1f}" for sp in SPECIES))
    lines.append(f"Trace species beyond the 5-species basis (sum {sum(trace.values()):.4f} kmol/d): "
                 + ", ".join(f"{k} {v:.4f}" for k, v in sorted(trace.items(), key=lambda kv: -kv[1])[:4]))
    lines.append(f"X_CO2 (Gibbs, equilibrium)            = {X:.4f}")
    lines.append(f"X_CO2 (RWGS-only equilibrium, CH4 inert) = {X_r1only:.4f}   <- plateau of 1-reaction reduced model")
    lines.append(f"xi1 (RWGS)            = {xi['R1_RWGS']:+.1f} kmol/d")
    lines.append(f"xi2 (CO2 methanation) = {xi['R2_CO2_methanation']:+.1f} kmol/d"
                 + ("   (negative = net steam reforming of CH4)" if xi['R2_CO2_methanation'] < 0 else ""))
    lines.append("Element-closure residuals dn - dn_pred [kmol/d]: "
                 + ", ".join(f"{sp} {resid[sp]:+.4f}" for sp in ["CO2", "H2", "H2O"])
                 + f"   (max relative {max(abs(resid[sp])/max(abs(dn[sp]),1e-30) for sp in ['CO2','H2','H2O']):.1e})")
    lines.append(f"dH(950 K): RWGS {dH['R1_RWGS']:+.2f} kJ/mol ; CO2 methanation {dH['R2_CO2_methanation']:+.2f} kJ/mol")
    lines.append(f"Q_RWGS = {Q['R1_RWGS']:+.1f} kW ; Q_meth = {Q['R2_CO2_methanation']:+.1f} kW ; "
                 f"sum = {Q_sum:+.1f} kW ; direct Q2 = {Q_direct:+.1f} kW ; difference {Q_sum-Q_direct:+.3f} kW")
    lines.append("")
    return rows_ext, rows_heat, lines, dict(X=X, X_r1only=X_r1only, xi=xi, Q=Q, Q_sum=Q_sum, Q_direct=Q_direct, dH=dH)

def main():
    fa, s1a = feed_A_eq_chain()
    fb, s1b = feed_B_kin_chain()
    ext, heat, txt = [], [], []
    txt.append("K3 Stage 2 extent decomposition and heat apportionment")
    txt.append(f"Conditions: T = {T2:.0f} K, P = {P2/ct.one_atm:.2f} atm, Gibbs equilibrium (GRI-3.0 gas phase). ALL VALUES EQUILIBRIUM.")
    txt.append("Feed A = EQ chain (Stage 1 Gibbs 1200 K) -> main-text baseline.  Feed B = KIN chain (Stage 1 kinetic tau=3 s) -> validation_stage2 CSV feed.")
    txt.append("")
    for tag, f, s1 in [("A_eq_chain", fa, s1a), ("B_kin_chain", fb, s1b)]:
        e, h, l, _ = analyse(tag, f, s1)
        ext += e; heat += h; txt += l
    pd.DataFrame(ext).to_csv(OUT_CSV_EXT, index=False)
    pd.DataFrame(heat).to_csv(OUT_CSV_HEAT, index=False)
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"Saved: {OUT_CSV_EXT}\n       {OUT_CSV_HEAT}\n       {OUT_TXT}")

if __name__ == "__main__":
    main()
