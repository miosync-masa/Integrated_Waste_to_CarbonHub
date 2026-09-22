# -*- coding: utf-8 -*-
"""
Kinetic chain as a design case (revision tasks K5 / R1-5 / m2 / m8; follows K3)
================================================================================
FUPROC-D-26-00269 Major Revision.

Motivation (from K3): with the submitted Stage 1 residence time tau1 = 3 s the
reduced kinetic model leaves 1,247 kmol/d CH4 unconverted, which enters Stage 2
and turns it from an RWGS reactor into a net CH4 reformer ("Feed B problem").
This script (1) fixes a Stage 1 design residence time by a stated criterion and
(2) re-runs the whole kinetic chain Stage 1 -> 2 -> 3 for several tau1, with the
two-reaction Stage 2 model of K3, and (3) places the equilibrium chain (Feed A,
main text) next to it.

1. Design residence time tau* (Stage 1, 1200 K, 1 atm)
   Criterion: approach to equilibrium X_CH4,kin / X_CH4,eq >= 0.90.
   Found by bisection on the submitted reduced model itself
   (run_stage1_pyrolysis_kinetic, Euler, 1000 steps, kref/Ea as submitted), not
   by interpolating the submitted tau sweep.
2. Kinetic chain cases
   tau1 in {3, tau*, 10, 30} s
   Stage 2: two-reaction reduced model (RWGS + CO2 methanation) of
            K3_stage2_two_reaction_kinetic.py, tau2 = 3 s, 950 K.
            phi in {0, 0.1, 1}. phi = 0 is the submitted one-reaction model.
            k2 is UNCALIBRATED. To keep k2 a property of the reactor and not of
            the feed, k2 is fixed per phi from the Feed A (equilibrium-chain)
            inlet: phi = r2,fwd / r1,fwd at that inlet. The effective inlet ratio
            realised in each kinetic case is reported as a column.
   Stage 3: submitted CFR reduced model (run_stage3_cfr_kinetic, 3 reactions,
            tau3 = 3 s) at 650 K (submitted) and at 800 K.
   Separation between Stage 2 and 3: water removal 0.95, H2 split to CFR 0.35,
   CO2/CO/CH4 fully to CFR (identical to the submitted baseline).
3. Equilibrium chain (Feed A): Stage 1 Gibbs 1200 K -> Stage 2 Gibbs 950 K ->
   same separation -> Stage 3 Gibbs at 650 K and 800 K. These are EQUILIBRIUM
   values; every other case in the tables is a KINETIC value.

Reproducibility
---------------
  cd <repo>/revision
  ../.venv/bin/python KIN_chain_design_case.py
Requires cantera>=3.0, numpy, scipy, pandas. Imports Workflow_cantera.py
read-only (its __main__ block is not executed). Baseline outputs in
CanteraResult/ are NOT touched.

Outputs (in revision/Result/):
  KIN_chain_cases.csv          one row per case: conversions, carbon, water,
                               H2, duties, closure residuals, reactor-volume ratio
  KIN_chain_streams.csv        long format: every stream of every case, kmol/d and t/d
  KIN_chain_tau_star.csv       bisection trace for tau*
  KIN_chain_summary.txt        human-readable summary
"""
import os, sys
import numpy as np
import pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W

OUT_CASES   = os.path.join(RESULT, "KIN_chain_cases.csv")
OUT_STREAMS = os.path.join(RESULT, "KIN_chain_streams.csv")
OUT_TAUSTAR = os.path.join(RESULT, "KIN_chain_tau_star.csv")
OUT_TXT     = os.path.join(RESULT, "KIN_chain_summary.txt")

# ---------------- settings (all taken from the submitted baseline where applicable)
T1, P1 = W.T_pyro, W.P_pyro           # 1200 K, 1 atm
T2, P2 = W.T_rwgs, W.P_rwgs           # 950 K, 1 atm
P3 = W.P_cfr
T3_LIST = [650.0, 800.0]
TAU1_REF = W.pyro_tau_s               # 3 s (submitted)
TAU2 = W.rwgs_tau_s                   # 3 s
TAU3 = W.cfr_tau_s                    # 3 s
PHI_LIST = [0.0, 0.1, 1.0]
APPROACH_TARGET = 0.90
WRF, H2F, CO2F, COF, CH4F = (W.water_remove_frac, W.h2_to_cfr_frac,
                             W.co2_to_cfr_frac, W.co_to_cfr_frac, W.ch4_to_cfr_frac)

R = ct.gas_constant / 1000.0; P0 = ct.one_atm
SP2 = ["CO2", "H2", "CO", "H2O", "CH4"]
NU1 = np.array([-1, -1, +1, +1, 0], float)
NU2 = np.array([-1, -4, 0, +2, +1], float)
K1 = W.rwgs_kref                      # 0.05 m3/mol/s (submitted)
MWC = W.MW_C_SOLID

def tpd(sp, kmold):
    mw = MWC if sp == "C(s)" else W.MW[sp]
    return kmold * mw / 1000.0

# ---------------- Stage 1 kinetic + tau*
def stage1_kin(tau1):
    return W.run_stage1_pyrolysis_kinetic(W.CH4_tpd, T1, P1, tau_s=tau1, n_steps=W.pyro_n_steps,
                                          kref=W.pyro_kref, Ea=W.pyro_Ea, Tref=W.pyro_Tref)

def stage1_eq():
    return W.run_stage1_pyrolysis(W.CH4_tpd, T1, P1)

def find_tau_star(x_eq, target=APPROACH_TARGET, lo=1.0, hi=30.0, tol=0.005):
    trace = []
    f = lambda tau: stage1_kin(tau)["X_CH4"] / x_eq
    flo, fhi = f(lo), f(hi)
    trace += [{"tau_s": lo, "approach": flo}, {"tau_s": hi, "approach": fhi}]
    assert flo < target < fhi, "bracket does not contain the target"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); fm = f(mid)
        trace.append({"tau_s": mid, "approach": fm})
        if fm >= target: hi = mid
        else: lo = mid
    return hi, pd.DataFrame(trace)   # hi is the smallest tested tau meeting the criterion

# ---------------- Stage 2 two-reaction model (identical to K3_stage2_two_reaction_kinetic.run_pfr)
def thermo2(T):
    g = ct.Solution("gri30.yaml"); g.TP = T, P0
    gRT = g.standard_gibbs_RT; i = [g.species_index(s) for s in SP2]
    Kp1 = float(np.exp(-NU1 @ gRT[i])); Kp2 = float(np.exp(-NU2 @ gRT[i]))
    return Kp1, Kp1, Kp2   # Kp1, Kc1 (dn=0), Kp2

def inlet_rates(feed, k1, k2, T, P):
    F0 = np.array([feed.get(s, 0.0) for s in SP2]); y = F0 / F0.sum()
    c = y * P / (R * T); a = y * (P / P0)
    return k1 * c[0] * c[1], k2 * a[0] * a[1] ** 4

def k2_from_phi(phi, ref_feed, k1, T, P):
    r1f, r2f_unit = inlet_rates(ref_feed, k1, 1.0, T, P)
    return phi * r1f / r2f_unit

def stage2_2rxn(feed, tau, k1, k2, T, P):
    feed = W.clean_species_dict(feed)
    Kp1, Kc1, Kp2 = thermo2(T)
    F0 = np.array([feed.get(s, 0.0) for s in SP2]) * 1000.0 / 86400.0
    others = {k: v for k, v in feed.items() if k not in SP2}   # inert pass-through (none expected)
    vdot0 = (F0.sum() + sum(others.values()) * 1000.0 / 86400.0) * R * T / P
    V = tau * vdot0; cT = P / (R * T)
    Fo = sum(others.values()) * 1000.0 / 86400.0
    def rhs(v, F):
        F = np.maximum(F, 0.0); Ft = F.sum() + Fo
        y = F / Ft; c = y * cT; a = y * (P / P0)
        r1 = k1 * (c[0] * c[1] - c[2] * c[3] / Kc1)
        r2 = k2 * (a[0] * a[1] ** 4 - a[4] * a[3] ** 2 / Kp2)
        return NU1 * r1 + NU2 * r2
    sol = solve_ivp(rhs, (0.0, V), F0, method="BDF", rtol=1e-10, atol=1e-13)
    if not sol.success: raise RuntimeError(sol.message)
    F = np.maximum(sol.y[:, -1], 0.0) * 86400.0 / 1000.0
    out = W.clean_species_dict({**{s: F[i] for i, s in enumerate(SP2)}, **others})
    xi1 = out.get("CO", 0) - feed.get("CO", 0); xi2 = out.get("CH4", 0) - feed.get("CH4", 0)
    fH = W.total_stream_enthalpy_J_per_day(feed, 0, T, P)
    pH = W.total_stream_enthalpy_J_per_day(out, 0, T, P)
    return {"feed": feed, "result": {"gas_kmol_d": out, "T_K": T, "P_Pa": P},
            "Q_kW": (pH - fH) / 86400.0 / 1000.0,
            "balance": W.element_balance_table(feed, out, 0),
            "X_CO2": 1 - out.get("CO2", 0) / feed["CO2"], "xi1": xi1, "xi2": xi2}

# ---------------- chain evaluation
def element_totals(streams_solid):
    """streams_solid: list of (gas_dict, solid_kmol) -> {C,H,O} kmol-atoms/d"""
    tot = {e: 0.0 for e in W.ELEMENTS}
    for gas, sc in streams_solid:
        inv = W.elemental_inventory_total(gas, sc)
        for e in W.ELEMENTS: tot[e] += inv[e]
    return tot

def evaluate_chain(case_id, chain, tau1, s1, s2_fn, T3, phi=None, k2=None):
    """s1: stage-1 result; s2_fn(feed)-> stage-2 result. Returns (row, stream_rows)."""
    rf, h2s = W.build_rwgs_feed(s1, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    s2 = s2_fn(rf)
    s2gas = s2["result"]["gas_kmol_d"]
    sep = W.build_cfr_feed(s2gas, WRF, H2F, CO2F, COF, CH4F)
    if chain == "kinetic":
        s3 = W.run_stage3_cfr_kinetic(sep["cfr_feed"], T3, P3, eta=1.0, tau_s=TAU3, n_steps=W.cfr_n_steps)
    else:
        s3 = W.run_stage3_cfr(sep["cfr_feed"], T3, P3)
        if not s3["result"]["converged"]: raise RuntimeError("Stage 3 equilibrium failed")
    s3gas = s3["result"]["gas_kmol_d"]; C1 = s1["result"]["Csolid_kmol_d"]; C3 = s3["result"]["Csolid_kmol_d"]

    # overall element closure: in = CH4 + CO2 + solar H2 ; out = C1 + recovered water + permeate + stage3 gas + C3
    feed_in = {"CH4": W.tpd_to_kmol_per_day(W.CH4_tpd, W.MW["CH4"]),
               "CO2": W.tpd_to_kmol_per_day(W.CO2_total_tpd, W.MW["CO2"]), "H2": h2s}
    tin = element_totals([(feed_in, 0.0)])
    tout = element_totals([({}, C1), (sep["recovered_water"], 0.0), (sep["membrane_permeate"], 0.0), (s3gas, C3)])
    closure = {e: (tout[e] - tin[e]) / tin[e] for e in W.ELEMENTS}
    stage_max = max(abs(b["Rel Error"].fillna(0)).max() for b in [s1["balance"], s2["balance"], s3["balance"]])

    # Gibbs reference for the SAME Stage-2 feed (equilibrium value; shows the thermodynamic destination)
    s2eq = W.run_stage2_rwgs(rf, T2, P2); s2eq_gas = s2eq["result"]["gas_kmol_d"]
    x1 = s1.get("X_CH4", 1 - s1["result"]["gas_kmol_d"].get("CH4", 0) / s1["feed"]["CH4"])
    h2_perm = sep["membrane_permeate"].get("H2", 0); h2_s3 = s3gas.get("H2", 0)
    xi2 = s2.get("xi2", s2gas.get("CH4", 0) - rf.get("CH4", 0)); xi1 = s2.get("xi1", s2gas.get("CO", 0))
    r1f, r2f = (inlet_rates(rf, K1, k2, T2, P2) if k2 is not None else (np.nan, np.nan))
    row = {
        "case_id": case_id, "chain": chain, "value_type": "kinetic" if chain == "kinetic" else "equilibrium",
        "stage1_basis": f"kinetic tau1={tau1:.2f} s" if chain == "kinetic" else "Gibbs 1200 K",
        "tau1_s": tau1 if chain == "kinetic" else np.nan, "V1_ratio_vs_3s": tau1 / TAU1_REF if chain == "kinetic" else np.nan,
        "X_CH4_stage1": x1, "approach_stage1": x1 / X1_EQ,
        "CH4_into_stage2_kmol_d": rf.get("CH4", 0), "H2_into_stage2_kmol_d": rf.get("H2", 0),
        "phi_k2_definition": phi, "k2_uncalibrated_mol_m3_s": k2,
        "phi_effective_at_inlet": (r2f / r1f if k2 is not None else np.nan),
        "tau2_s": TAU2 if chain == "kinetic" else np.nan, "X_CO2_stage2": s2.get("X_CO2", 1 - s2gas.get("CO2", 0) / rf["CO2"]),
        "xi1_RWGS_kmol_d": xi1, "xi2_CO2_methanation_kmol_d": xi2,
        "stage2_character": "methanation (xi2>0)" if xi2 > 0 else "reforming (xi2<0)",
        "X_CO2_stage2_gibbs_ref_same_feed": 1 - s2eq_gas.get("CO2", 0) / rf["CO2"],
        "xi2_gibbs_ref_same_feed_kmol_d": s2eq_gas.get("CH4", 0) - rf.get("CH4", 0),
        "Q2_gibbs_ref_same_feed_kW": s2eq["Q_kW"],
        "stage2_character_at_equilibrium": "methanation (xi2>0)" if (s2eq_gas.get("CH4", 0) - rf.get("CH4", 0)) > 0 else "reforming (xi2<0)",
        "CO_out_stage2_kmol_d": s2gas.get("CO", 0), "CH4_out_stage2_kmol_d": s2gas.get("CH4", 0),
        "T3_K": T3, "tau3_s": TAU3 if chain == "kinetic" else np.nan,
        "C_stage1_tpd": tpd("C(s)", C1), "C_stage3_tpd": tpd("C(s)", C3), "C_total_tpd": tpd("C(s)", C1 + C3),
        "CFR_share_of_carbon": C3 / (C1 + C3) if (C1 + C3) > 0 else np.nan,
        "water_recovered_stage2_tpd": tpd("H2O", sep["recovered_water"].get("H2O", 0)),
        "H2O_out_stage3_tpd": tpd("H2O", s3gas.get("H2O", 0)),
        "H2_membrane_permeate_tpd": tpd("H2", h2_perm), "H2_out_stage3_tpd": tpd("H2", h2_s3),
        "H2_surplus_total_tpd": tpd("H2", h2_perm + h2_s3),
        "CH4_out_stage3_tpd": tpd("CH4", s3gas.get("CH4", 0)), "CO_out_stage3_tpd": tpd("CO", s3gas.get("CO", 0)),
        "CO2_out_stage3_tpd": tpd("CO2", s3gas.get("CO2", 0)),
        "Q1_kW": s1["Q_kW"], "Q2_kW": s2["Q_kW"], "Q3_kW": s3["Q_kW"], "Q_total_kW": s1["Q_kW"] + s2["Q_kW"] + s3["Q_kW"],
        "closure_C_rel": closure["C"], "closure_H_rel": closure["H"], "closure_O_rel": closure["O"],
        "closure_max_per_stage_rel": stage_max,
    }
    streams = []
    def add(name, gas, solid=0.0):
        for sp, n in sorted(W.clean_species_dict(gas).items(), key=lambda kv: -kv[1]):
            if n > 1e-6: streams.append({"case_id": case_id, "value_type": row["value_type"], "stream": name, "species": sp, "kmol_d": n, "t_d": tpd(sp, n) if sp in W.MW else np.nan})
        if solid > 1e-6: streams.append({"case_id": case_id, "value_type": row["value_type"], "stream": name, "species": "C(s)", "kmol_d": solid, "t_d": tpd("C(s)", solid)})
    add("stage1_out", s1["result"]["gas_kmol_d"], C1); add("stage2_feed", rf); add("stage2_out", s2gas)
    add("recovered_water", sep["recovered_water"]); add("membrane_permeate", sep["membrane_permeate"])
    add("cfr_feed", sep["cfr_feed"]); add("stage3_out", s3gas, C3)
    return row, streams

# ---------------- main
def main():
    global X1_EQ
    s1e = stage1_eq(); X1_EQ = 1 - s1e["result"]["gas_kmol_d"].get("CH4", 0) / s1e["feed"]["CH4"]
    tau_star, trace = find_tau_star(X1_EQ)
    trace.to_csv(OUT_TAUSTAR, index=False)
    s1_star = stage1_kin(tau_star)

    # reference inlet for k2 definition = Feed A (equilibrium chain)
    rfA, _ = W.build_rwgs_feed(s1e, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    k2_map = {phi: k2_from_phi(phi, rfA, K1, T2, P2) for phi in PHI_LIST}

    rows, streams = [], []
    tau1_cases = [(TAU1_REF, "3s"), (tau_star, "taustar"), (10.0, "10s"), (30.0, "30s")]
    for tau1, lab in tau1_cases:
        s1 = stage1_kin(tau1)
        for phi in PHI_LIST:
            k2 = k2_map[phi]
            s2_fn = lambda feed, k2=k2: stage2_2rxn(feed, TAU2, K1, k2, T2, P2)
            for T3 in T3_LIST:
                cid = f"kin_tau1-{lab}_phi-{phi:g}_T3-{int(T3)}"
                r, s = evaluate_chain(cid, "kinetic", tau1, s1, s2_fn, T3, phi, k2); rows.append(r); streams += s
    for T3 in T3_LIST:
        cid = f"eq_FeedA_T3-{int(T3)}"
        r, s = evaluate_chain(cid, "equilibrium", np.nan, s1e, lambda feed: W.run_stage2_rwgs(feed, T2, P2), T3); rows.append(r); streams += s

    df = pd.DataFrame(rows); df.to_csv(OUT_CASES, index=False)
    pd.DataFrame(streams).to_csv(OUT_STREAMS, index=False)

    L = []
    L.append("Kinetic chain design case — summary (KINETIC values unless the row says equilibrium)")
    L.append(f"Stage 1 equilibrium X_CH4 at 1200 K = {X1_EQ:.4f}. Criterion approach >= {APPROACH_TARGET}.")
    L.append(f"tau* = {tau_star:.3f} s (bisection on the submitted reduced model, tol 5 ms); X_CH4(tau*) = {s1_star['X_CH4']:.4f}, approach = {s1_star['X_CH4']/X1_EQ:.4f}")
    L.append(f"For comparison: tau = 10 s -> X_CH4 = {stage1_kin(10.0)['X_CH4']:.4f} (approach {stage1_kin(10.0)['X_CH4']/X1_EQ:.4f}); tau = 3 s -> {stage1_kin(3.0)['X_CH4']:.4f} ({stage1_kin(3.0)['X_CH4']/X1_EQ:.4f})")
    L.append("k2 fixed per phi from the Feed A inlet: " + ", ".join(f"phi={p:g}: k2={k:.4g} mol/m3/s" for p, k in k2_map.items()) + "  (UNCALIBRATED)")
    L.append("")
    cols = ["case_id", "CH4_into_stage2_kmol_d", "xi2_CO2_methanation_kmol_d", "xi2_gibbs_ref_same_feed_kmol_d", "X_CO2_stage2", "Q2_gibbs_ref_same_feed_kW", "C_stage1_tpd", "C_stage3_tpd", "C_total_tpd",
            "CFR_share_of_carbon", "water_recovered_stage2_tpd", "H2_surplus_total_tpd", "Q1_kW", "Q2_kW", "Q3_kW", "Q_total_kW", "closure_max_per_stage_rel"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.4g}"):
        L.append(df[cols].to_string(index=False))
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nSaved: {OUT_CASES}\n       {OUT_STREAMS}\n       {OUT_TAUSTAR}\n       {OUT_TXT}")

if __name__ == "__main__":
    main()
