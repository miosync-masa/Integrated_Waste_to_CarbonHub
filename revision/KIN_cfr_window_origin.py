"""
KIN_cfr_window_origin.py -- origin (waste CH4 vs CO2) of the Stage 3 solid carbon in the SUBMITTED
kinetic once-through chain across the 750-850 K "CFR carbon window" (SI table, kref multiplier = 1).
Reviewer R4 Major 3: is the 750-850 K deposition CO2 fixation or CH4 cracking?

Chain (identical to submitted_v1/Workflow_cantera.run_baseline, Stage 3 temperature varied):
  Stage 1 kinetic pyrolysis 1200 K, tau1 in {3 s (submitted), 8.905 s (design case tau1*)}
  Stage 2 kinetic RWGS 950 K, tau2 = 3 s, fresh CO2 42 t/d + solar H2 (submitted feed)
  water removal 0.95, membrane h = 0.35 (CO2, CO, CH4 fully to Stage 3)
  Stage 3 reduced 3-reaction kinetic model (R1 CO2 methanation, R2 CO + H2 -> C + H2O, R3 CH4 -> C + 2H2),
  tau3 = 3 s, eta = 1, kref multiplier 1, T3 in {650, 700, 750, 775, 800, 825, 850} K.
KINETIC values, rate constants UNCALIBRATED (submitted reduced models).

Two attributions of the Stage 3 solid carbon:
  (a) species-resolved net-conversion tracer (P1_recycle_analysis._reactor_origins), the method used
      throughout the revision; the origin of CO/CH4 entering Stage 3 is tracked through Stage 1-2;
  (b) reaction-resolved: R2 carbon carries the origin of the CO consumed, R3 carbon the origin of the CH4
      consumed (when R3 runs in reverse, the net solid comes from R2 only).
Check: C_stage3 at tau1 = 3 s must reproduce submitted_v1/CanteraResult/cfr_carbon_window.csv (mult = 1).

Outputs (revision/Result/): KIN_cfr_window_origin.csv, KIN_cfr_window_origin_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python KIN_cfr_window_origin.py   (~1 min)
"""
import os, time
import numpy as np, pandas as pd
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W
from P1_recycle_analysis import _reactor_origins, _mix_origins, c_atoms, carbon, tpd

T3_GRID = [650.0, 700.0, 750.0, 775.0, 800.0, 825.0, 850.0]
TAU1 = {"submitted tau1 = 3 s": 3.0, "design case tau1* = 8.905 s": 8.905}
E_CH4 = np.array([1.0, 0.0, 0.0])
CO2_BIO = W.tpd_to_kmol_per_day(W.CO2_from_biogas_tpd, W.MW["CO2"]); CO2_DAC = W.tpd_to_kmol_per_day(W.CO2_from_DAC_tpd, W.MW["CO2"])
E_CO2 = np.array([0.0, CO2_BIO, CO2_DAC]) / (CO2_BIO + CO2_DAC)

def main():
    t0 = time.time(); rows = []
    ref = pd.read_csv(os.path.join(BASE, "CanteraResult", "cfr_carbon_window.csv")); ref = ref[ref.co_carb_mult == 1].set_index("T_K")
    for lab, tau1 in TAU1.items():
        s1 = W.run_stage1_pyrolysis_kinetic(W.CH4_tpd, W.T_pyro, W.P_pyro, tau_s=tau1, n_steps=W.pyro_n_steps, kref=W.pyro_kref, Ea=W.pyro_Ea, Tref=W.pyro_Tref)
        gas1, C1 = s1["result"]["gas_kmol_d"], s1["result"]["Csolid_kmol_d"]; n_ch4 = s1["feed"]["CH4"]
        rf, h2s = W.build_rwgs_feed(s1, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
        s2 = W.run_stage2_rwgs_kinetic(rf, W.T_rwgs, W.P_rwgs, tau_s=W.rwgs_tau_s, n_steps=W.rwgs_n_steps, kref=W.rwgs_kref, Ea=W.rwgs_Ea, Tref=W.rwgs_Tref)
        gas2 = s2["result"]["gas_kmol_d"]
        sep = W.build_cfr_feed(gas2, W.water_remove_frac, W.h2_to_cfr_frac, W.co2_to_cfr_frac, W.co_to_cfr_frac, W.ch4_to_cfr_frac); cf = sep["cfr_feed"]
        # carbon origins through Stage 1 and 2
        o_gas1, _ = _reactor_origins({"CH4": n_ch4}, {"CH4": E_CH4}, gas1, C1)
        o_feed2 = _mix_origins([(gas1, o_gas1), ({"CO2": CO2_BIO + CO2_DAC}, {"CO2": E_CO2})])
        o_gas2, _ = _reactor_origins(rf, o_feed2, gas2, 0.0)
        o_cf = {sp: o_gas2[sp] for sp in cf if sp in o_gas2}
        for T3 in T3_GRID:
            s3 = W.run_stage3_cfr_kinetic(dict(cf), T3, W.P_cfr, eta=1.0, tau_s=W.cfr_tau_s, n_steps=W.cfr_n_steps)
            gas3, C3 = s3["result"]["gas_kmol_d"], s3["result"]["Csolid_kmol_d"]; ext = s3.get("extent_kmol_d", {})
            e_meth, e_carb, e_crack = ext.get("CO2_methanation", 0.0), ext.get("CO_carbon", 0.0), ext.get("CH4_cracking", 0.0)
            # (a) species-resolved net-conversion tracer
            _, f_sp = _reactor_origins(cf, o_cf, gas3, C3)
            # (b) reaction-resolved
            oCO, oCH4 = o_cf.get("CO", E_CO2), o_cf.get("CH4", E_CH4)
            if C3 <= 1e-9: f_rx = np.full(3, np.nan)
            elif e_crack >= 0: f_rx = (e_carb * oCO + e_crack * oCH4) / (e_carb + e_crack)
            else: f_rx = oCO.copy()   # reverse cracking removes solid; the net solid is R2 carbon
            rows.append(dict(stage1_case=lab, tau1_s=tau1, T3_K=T3, value_type="kinetic (uncalibrated)",
                             X_CH4_stage1=s1["X_CH4"], CH4_into_stage3_kmol_d=cf.get("CH4", 0), CO_into_stage3_kmol_d=cf.get("CO", 0),
                             CO2_into_stage3_kmol_d=cf.get("CO2", 0), H2_into_stage3_kmol_d=cf.get("H2", 0),
                             CO_origin_CO2_fraction=float(oCO[1] + oCO[2]), CH4_origin_CO2_fraction=float(oCH4[1] + oCH4[2]),
                             ext_CO2_methanation_kmol_d=e_meth, ext_CO_carbon_kmol_d=e_carb, ext_CH4_cracking_kmol_d=e_crack,
                             C_stage3_kmol_d=C3, C_stage3_tpd=tpd("C(s)", C3), C_stage3_submitted_window_tpd=(float(ref.C_solid_tpd.get(T3, np.nan)) if tau1 == 3.0 else np.nan),
                             C_stage1_tpd=tpd("C(s)", C1),
                             tracer_CH4_fraction=float(f_sp[0]), tracer_CO2_fraction=float(f_sp[1] + f_sp[2]),
                             tracer_C3_from_CH4_tpd=tpd("C(s)", C3) * float(f_sp[0]), tracer_C3_from_CO2_tpd=tpd("C(s)", C3) * float(f_sp[1] + f_sp[2]),
                             reaction_CH4_fraction=float(f_rx[0]), reaction_CO2_fraction=float(f_rx[1] + f_rx[2]),
                             reaction_C3_from_CH4_tpd=tpd("C(s)", C3) * float(f_rx[0]), reaction_C3_from_CO2_tpd=tpd("C(s)", C3) * float(f_rx[1] + f_rx[2]),
                             CO2_fixed_fraction_of_fresh_CO2_tracer=tpd("C(s)", C3) * float(f_sp[1] + f_sp[2]) / tpd("C(s)", CO2_BIO + CO2_DAC),
                             Q3_kW=s3["Q_kW"]))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESULT, "KIN_cfr_window_origin.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["tau1_s", "T3_K", "C_stage3_tpd", "C_stage3_submitted_window_tpd", "ext_CO2_methanation_kmol_d", "ext_CO_carbon_kmol_d", "ext_CH4_cracking_kmol_d",
            "tracer_CO2_fraction", "tracer_C3_from_CO2_tpd", "tracer_C3_from_CH4_tpd", "reaction_CO2_fraction", "reaction_C3_from_CO2_tpd", "reaction_C3_from_CH4_tpd", "CO2_fixed_fraction_of_fresh_CO2_tracer"]
    L = [f"Stage 3 carbon origin across the CFR window (wall {time.time()-t0:.0f} s). KINETIC, uncalibrated submitted reduced models; once-through, h = 0.35, kref mult = 1.",
         f"Origin of CO entering Stage 3 (CO2-derived fraction): " + ", ".join(f"{lab}: {D[D.stage1_case==lab].CO_origin_CO2_fraction.iloc[0]:.3f}" for lab in TAU1),
         f"Origin of CH4 entering Stage 3 (CO2-derived fraction): " + ", ".join(f"{lab}: {D[D.stage1_case==lab].CH4_origin_CO2_fraction.iloc[0]:.3f}" for lab in TAU1),
         f"Stage 3 inlet [kmol/d] CH4 / CO / CO2 / H2: " + "; ".join(f"{lab}: {r.CH4_into_stage3_kmol_d:.0f} / {r.CO_into_stage3_kmol_d:.0f} / {r.CO2_into_stage3_kmol_d:.0f} / {r.H2_into_stage3_kmol_d:.0f}" for lab in TAU1 for r in [D[D.stage1_case==lab].iloc[0]]),
         f"max |C_stage3 - submitted window (tau1 = 3 s)| = {(D.C_stage3_tpd - D.C_stage3_submitted_window_tpd).abs().max():.2e} t/d", ""]
    for lab in TAU1: L += [f"=== {lab}", D[D.stage1_case == lab][cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"), ""]
    open(os.path.join(RESULT, "KIN_cfr_window_origin_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
