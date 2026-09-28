"""
C2_table12_X4_once_through.py -- inter-stage coupling vs Stage 1 residence time for the CURRENT base
configuration (manuscript Table 12), replacing the rows that were computed with the submitted feed.

Chain (once-through, X4 feed): Stage 1 kinetic pyrolysis 1200 K at tau1 in {3, 8.905 (tau1*), 10, 30} s, or Gibbs;
fresh CO2 = all biogas CO2 (1,995 kmol/d, y = 0.60), no electrolytic H2; Stage 2 two-reaction model at 950 K
(tau2 = 3 s, RWGS k1 = 0.05, methanation k2 from phi in {0, 0.1, 1} defined at the Feed A inlet as in KIN/F1) or Gibbs;
water removal 0.95; membrane h = 0.50; Stage 3 at 650 K: reduced kinetic model (tau3 = 3 s) for kinetic rows,
robust Gibbs (F1) for the equilibrium row. For every kinetic Stage 2 the Gibbs destination of the SAME Stage 2 feed
is reported (X_CO2, xi2, Q2 at equilibrium). KINETIC rows use uncalibrated rate constants.

Outputs (revision/Result/): C2_table12_X4.csv, C2_table12_X4_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python C2_table12_X4_once_through.py   (~1 min)
"""
import os, time
import numpy as np, pandas as pd
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W
import KIN_chain_design_case as KC
import F1_full_biogas_CO2 as F1m
from P1_recycle_analysis import FRESH_CH4, tpd, carbon, add

T1, P1, T2, P2, T3, P3 = KC.T1, KC.P1, KC.T2, KC.P2, W.T_cfr, KC.P3
WRF, H = W.water_remove_frac, 0.50
FD = F1m.feeds()["2b_y0.60"]; CO2_IN = FD["co2_bio"] + FD["co2_dac"]

def chain(label, s1, tau1, mode2, phi=None, k2=None):
    gas1, C1 = s1["result"]["gas_kmol_d"], s1["result"]["Csolid_kmol_d"]
    rf = W.clean_species_dict(add(gas1, {"CO2": CO2_IN}))
    if mode2 == "kinetic": s2 = KC.stage2_2rxn(rf, KC.TAU2, KC.K1, k2, T2, P2); g2 = s2["result"]["gas_kmol_d"]; Q2 = s2["Q_kW"]
    else: r2 = W.run_stage2_rwgs(rf, T2, P2); g2 = r2["result"]["gas_kmol_d"]; Q2 = r2["Q_kW"]
    r2e = W.run_stage2_rwgs(rf, T2, P2); g2e = r2e["result"]["gas_kmol_d"]
    pw2, w2 = W.remove_species(g2, "H2O", WRF); cf, perm = W.membrane_split(pw2, {"H2": H, "CO2": 1.0, "CO": 1.0, "CH4": 1.0}, 1.0)
    if mode2 == "kinetic":
        r3 = W.run_stage3_cfr_kinetic(dict(cf), T3, P3, eta=1.0, tau_s=KC.TAU3, n_steps=W.cfr_n_steps); g3, C3, Q3 = r3["result"]["gas_kmol_d"], r3["result"]["Csolid_kmol_d"], r3["Q_kW"]
    else: g3, C3, Q3 = F1m.stage3_eq_robust(cf, P3)
    pw3, w3 = W.remove_species(g3, "H2O", WRF)
    x1 = 1 - gas1.get("CH4", 0) / FRESH_CH4
    return dict(case=label, value_type=("kinetic" if mode2 == "kinetic" else "equilibrium"), tau1_s=tau1, phi=phi,
                X_CH4_stage1=x1, approach_stage1=x1 / X1_EQ, CH4_into_stage2_kmol_d=rf.get("CH4", 0), H2_into_stage2_kmol_d=rf.get("H2", 0), S2_inlet_H2_to_CO2=rf.get("H2", 0) / CO2_IN,
                X_CO2_stage2=1 - g2.get("CO2", 0) / CO2_IN, xi1_RWGS_kmol_d=g2.get("CO", 0) - rf.get("CO", 0), xi2_CO2_methanation_kmol_d=g2.get("CH4", 0) - rf.get("CH4", 0),
                stage2_character=("methanation (xi2>0)" if g2.get("CH4", 0) - rf.get("CH4", 0) > 0 else "reforming (xi2<0)"), Q2_kW=Q2,
                X_CO2_stage2_gibbs_ref_same_feed=1 - g2e.get("CO2", 0) / CO2_IN, xi2_gibbs_ref_same_feed_kmol_d=g2e.get("CH4", 0) - rf.get("CH4", 0), Q2_gibbs_ref_same_feed_kW=r2e["Q_kW"],
                stage2_character_at_equilibrium=("methanation (xi2>0)" if g2e.get("CH4", 0) - rf.get("CH4", 0) > 0 else "reforming (xi2<0)"),
                CO_out_stage2_kmol_d=g2.get("CO", 0), CH4_out_stage2_kmol_d=g2.get("CH4", 0), S3_inlet_H2_to_CO2=cf.get("H2", 0) / max(cf.get("CO2", 0), 1e-12),
                C_stage1_tpd=tpd("C(s)", C1), C_stage3_tpd=tpd("C(s)", C3), C_total_tpd=tpd("C(s)", C1 + C3), CFR_share_of_carbon=C3 / (C1 + C3),
                water_recovered_tpd=tpd("H2O", w2.get("H2O", 0) + w3.get("H2O", 0)), H2_membrane_permeate_tpd=tpd("H2", perm.get("H2", 0)), H2_out_stage3_tpd=tpd("H2", g3.get("H2", 0)),
                H2_surplus_total_tpd=tpd("H2", perm.get("H2", 0) + g3.get("H2", 0)), CH4_out_stage3_tpd=tpd("CH4", g3.get("CH4", 0)), CO_out_stage3_tpd=tpd("CO", g3.get("CO", 0)), CO2_out_stage3_tpd=tpd("CO2", g3.get("CO2", 0)),
                Q1_kW=s1["Q_kW"], Q3_kW=Q3, Q_total_kW=s1["Q_kW"] + Q2 + Q3,
                closure_C_rel=(carbon(pw3) + C1 + C3 - (FRESH_CH4 + CO2_IN)) / (FRESH_CH4 + CO2_IN))

def main():
    global X1_EQ; t0 = time.time()
    s1e = KC.stage1_eq(); X1_EQ = 1 - s1e["result"]["gas_kmol_d"].get("CH4", 0) / s1e["feed"]["CH4"]
    rfA, _ = W.build_rwgs_feed(s1e, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    k2_map = {phi: KC.k2_from_phi(phi, rfA, KC.K1, T2, P2) for phi in KC.PHI_LIST}
    rows = []
    for tau1, lab in [(3.0, "3 s"), (8.905, "tau1* = 8.905 s"), (10.0, "10 s"), (30.0, "30 s")]:
        s1 = KC.stage1_kin(tau1)
        for phi in KC.PHI_LIST: rows.append(chain(f"X4 once-through, kinetic, tau1 = {lab}, phi = {phi:g}", s1, tau1, "kinetic", phi, k2_map[phi]))
    rows.append(chain("X4 once-through, equilibrium (Gibbs all stages)", s1e, np.nan, "equilibrium"))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESULT, "C2_table12_X4.csv"), index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    cols = ["tau1_s", "phi", "X_CH4_stage1", "H2_into_stage2_kmol_d", "S2_inlet_H2_to_CO2", "X_CO2_stage2", "xi1_RWGS_kmol_d", "xi2_CO2_methanation_kmol_d", "Q2_kW",
            "X_CO2_stage2_gibbs_ref_same_feed", "xi2_gibbs_ref_same_feed_kmol_d", "Q2_gibbs_ref_same_feed_kW", "C_stage1_tpd", "C_stage3_tpd", "C_total_tpd", "H2_surplus_total_tpd", "water_recovered_tpd", "Q1_kW", "Q3_kW", "closure_C_rel"]
    L = [f"C2: Table 12 recomputed for the X4 feed, once-through (wall {time.time()-t0:.0f} s). Stage 1 equilibrium X_CH4 = {X1_EQ:.4f}; fresh CO2 {CO2_IN:,.0f} kmol/d, no electrolytic H2; h = 0.50; T3 = 650 K.",
         "k2 per phi (Feed A inlet definition, as KIN/F1): " + ", ".join(f"phi={p:g}: {k:.4g}" for p, k in k2_map.items()) + "  (UNCALIBRATED)", "",
         D[cols].to_string(index=False, float_format=lambda x: f"{x:,.4g}")]
    open(os.path.join(RESULT, "C2_table12_X4_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
