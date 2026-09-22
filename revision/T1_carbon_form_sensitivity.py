# -*- coding: utf-8 -*-
"""
T1: Sensitivity to the solid-carbon phase (reviewer R1-2, "carbon form")
=========================================================================
FUPROC-D-26-00269 Major Revision. Builds on KIN_chain (tau1* = 8.905 s), P1 (recycle
flowsheet, tracers) and X1 (solid carbon forms essentially only in Stage 1).

The baseline treats solid carbon as graphite (NASA polynomials, graphite.yaml).
Amorphous, turbostratic or filamentous carbons are less stable than graphite by a
few to some ten kJ/mol. This script re-runs the equilibrium and kinetic design
calculations with a carbon phase whose Gibbs energy is raised by
    dG in {0, 5, 10, 20} kJ/mol   (0 = graphite, submitted baseline)
at all temperatures. Implementation: the NASA7 enthalpy constant a6 of C(gr) is
shifted by dG/R in both temperature ranges, i.e. dH = dG, dS = 0, so the offset is
exactly dG at every temperature (verified). The modified phase replaces graphite in
the three baseline functions that touch the solid (Gibbs minimisation with graphite,
stream enthalpy, reaction Kp for the Stage 3 reduced model); every other line of the
submitted code is unchanged. Because dH = dG, the heat of carbon formation in
Stage 1 rises by dG per mol C and is included in Q1.

Parts
  A  Once-through equilibrium chain (Feed A; Stage 1 Gibbs 1200 K, Stage 2 Gibbs
     950 K, separation 0.95 / h = 0.35, Stage 3 Gibbs 650 K).      EQUILIBRIUM.
  B  Equilibrium recycle (P1 topology; r_CH4 = 0.95, h = 0.35, T3 = 650 K)
     at p = 0.05 (representative) and p = 0.2.                      EQUILIBRIUM.
  C  Kinetic Stage 1 design case: X_CH4 at tau1* = 8.905 s with the shifted Kp,
     approach to the shifted equilibrium, tau* re-derived for approach >= 0.90,
     and the once-through kinetic design chain (phi = 0) total carbon.  KINETIC,
     rate constants UNCALIBRATED (Kp of the reduced model is rebuilt from the
     shifted equilibrium as in the submitted code).

Reproducibility
---------------
  cd <repo>/revision
  ../.venv/bin/python T1_carbon_form_sensitivity.py      (~2 min, serial)
Imports Workflow_cantera.py, P1_recycle_analysis.py, KIN_chain_design_case.py.
Baseline outputs in CanteraResult/ are NOT touched.

Outputs (in revision/Result/)
  T1_carbon_form_sensitivity.csv    one row per (part, dG, case)
  T1_carbon_form_summary.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
import cantera as ct

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import P1_recycle_analysis as P1m
import KIN_chain_design_case as KC
from P1_recycle_analysis import FRESH_CH4, tpd

OUT_CSV = os.path.join(RESULT, "T1_carbon_form_sensitivity.csv")
OUT_TXT = os.path.join(RESULT, "T1_carbon_form_summary.txt")
DG_GRID = [0.0, 5.0, 10.0, 20.0]      # kJ/mol above graphite
T1, T2, T3 = W.T_pyro, W.T_rwgs, W.T_cfr
P_ATM = ct.one_atm
REC = dict(r=0.95, h=0.35)

# ---------------- carbon phase with Gibbs offset
_GRAPHITE_MODEL = ct.Solution("graphite.yaml").thermo_model
def make_carbon(dG_kJ):
    sp = ct.Species.list_from_file("graphite.yaml")[0]; th = sp.thermo; c = th.coeffs.copy()
    shift = dG_kJ * 1e6 / ct.gas_constant       # J/kmol / (J/kmol/K) -> K
    c[6] += shift; c[13] += shift                # a6 of high-T and low-T ranges
    sp.thermo = ct.NasaPoly2(th.min_temp, th.max_temp, th.reference_pressure, c)
    return ct.Solution(thermo=_GRAPHITE_MODEL, species=[sp])

SOLID = None
def patch_solid(dG_kJ):
    """Replace the three baseline functions that build graphite internally."""
    global SOLID
    SOLID = make_carbon(dG_kJ)
    def equilibrate_gas_plus_graphite(feed, T, P, ism=None):
        g = ct.Solution("gri30.yaml"); s = make_carbon(dG_kJ)
        x, n = W.normalize_mole_dict(feed); g.TPX = T, P, x
        return W._equilibrate_mixture([(g, n), (s, 0)], T, P, g.species_names, s.species_names[0], ism, W.EQ_TRIES_MULTIPHASE)
    def total_stream_enthalpy_J_per_day(gsm, sc, T, P):
        H = 0.0; gsm = W.clean_species_dict(gsm)
        if gsm:
            g = ct.Solution("gri30.yaml"); x, n = W.normalize_mole_dict(gsm); g.TPX = T, P, x; H += g.enthalpy_mole * n
        if sc > 0:
            s = make_carbon(dG_kJ); s.TP = T, P; H += s.enthalpy_mole * sc
        return H
    def reaction_Kp(T, stoich):
        g = ct.Solution("gri30.yaml"); s = make_carbon(dG_kJ)
        g.TPX = T, ct.one_atm, "H2:1.0"; s.TP = T, ct.one_atm
        dG_RT = 0.0
        for sp, nu in stoich.items():
            dG_RT += nu * (s.standard_gibbs_RT[0] if sp == "C(s)" else g.standard_gibbs_RT[g.species_index(sp)])
        return np.exp(-dG_RT)
    W.equilibrate_gas_plus_graphite = equilibrate_gas_plus_graphite
    W.total_stream_enthalpy_J_per_day = total_stream_enthalpy_J_per_day
    W.reaction_Kp = reaction_Kp

def verify_offset(dG_kJ):
    s0, s1 = make_carbon(0.0), make_carbon(dG_kJ); out = []
    for T in (650.0, 950.0, 1200.0):
        s0.TP = T, P_ATM; s1.TP = T, P_ATM
        out.append(((s1.gibbs_mole - s0.gibbs_mole) / 1e6, (s1.enthalpy_mole - s0.enthalpy_mole) / 1e6))
    return out

def base_row(part, dG, case, cfg, rec, info, ref=None):
    row = P1m.make_row(cfg, rec, info, ref)
    row.update(part=part, dG_kJ_per_mol=dG, case=case)
    return row

def main():
    t0 = time.time(); rows = []; L = []
    L.append("T1 carbon-form sensitivity. Parts A/B EQUILIBRIUM, Part C KINETIC (uncalibrated). dG = Gibbs energy of the solid carbon phase above graphite.")
    for dG in DG_GRID:
        v = verify_offset(dG)
        L.append(f"dG = {dG:g} kJ/mol: offset check (dG, dH) at 650/950/1200 K = " + ", ".join(f"({a:.3f}, {b:.3f})" for a, b in v))
        patch_solid(dG)
        # ---- Part A: once-through equilibrium chain
        cfgA = dict(tier="T1_A", mode="equilibrium", T3=T3, phi=np.nan, k2=np.nan, h=REC["h"], p=1.0, r=0.0)
        recA, infoA = P1m.solve_recycle(cfgA); refA = base_row("A_once_through_eq", dG, "eq once-through Feed A chain", cfgA, recA, infoA)
        Kp1 = W.pyro_Kp_from_equilibrium(T1, P_ATM); refA["Kp_pyrolysis_1200K_1atm"] = Kp1
        refA["S3_CH4_formed_kmol_d"] = recA["gas3"].get("CH4", 0) - recA["cfr_feed"].get("CH4", 0)
        rows.append(refA)
        # ---- Part B: equilibrium recycle
        for p in (0.05, 0.2):
            cfgB = dict(tier="T1_B", mode="equilibrium", T3=T3, phi=np.nan, k2=np.nan, h=REC["h"], p=p, r=REC["r"])
            recB, infoB = P1m.solve_recycle(cfgB)
            rB = base_row("B_recycle_eq", dG, f"eq recycle p={p:g}", cfgB, recB, infoB, refA)
            rB["S3_CH4_formed_kmol_d"] = recB["gas3"].get("CH4", 0) - recB["cfr_feed"].get("CH4", 0); rows.append(rB)
        # ---- Part C: kinetic Stage 1 design case
        s1eq = W.run_stage1_pyrolysis(W.CH4_tpd, T1, P_ATM); x_eq = 1 - s1eq["result"]["gas_kmol_d"].get("CH4", 0) / s1eq["feed"]["CH4"]
        s1k = KC.stage1_kin(KC.TAU1_REF * 0 + 8.905)
        tau_star, _ = KC.find_tau_star(x_eq)
        s1k_star = KC.stage1_kin(tau_star)
        cfgC = dict(tier="T1_C", mode="kinetic", T3=T3, phi=0.0, k2=0.0, h=REC["h"], p=1.0, r=0.0)
        recC, infoC = P1m.solve_recycle(cfgC); rC = base_row("C_kinetic_design_once_through", dG, "kin once-through design chain (tau1=8.905 s, phi=0)", cfgC, recC, infoC)
        rC.update(Kp_pyrolysis_1200K_1atm=Kp1, S1_X_CH4_eq_shifted=x_eq, S1_X_CH4_kin_tau8905=s1k["X_CH4"], S1_approach_tau8905=s1k["X_CH4"] / x_eq,
                  C1_kin_tau8905_tpd=tpd("C(s)", s1k["result"]["Csolid_kmol_d"]), Q1_kin_tau8905_kW=s1k["Q_kW"],
                  tau_star_for_approach_0p9_s=tau_star, S1_X_CH4_kin_at_new_tau_star=s1k_star["X_CH4"],
                  C1_kin_at_new_tau_star_tpd=tpd("C(s)", s1k_star["result"]["Csolid_kmol_d"]), V1_ratio_new_tau_star_vs_3s=tau_star / 3.0)
        rows.append(rC)
        L.append(f"  Part A eq once-through: X_CH4 {refA['S1_X_CH4']:.4f}, C1 {refA['C_stage1_tpd']:.2f}, C3 {refA['C_stage3_tpd']:.3f}, total {refA['C_total_tpd']:.2f} t/d, Q1 {refA['Q1_kW']:.0f} kW, Kp {Kp1:.3f}")
        for r in rows[-3:-1]:
            L.append(f"  Part B eq recycle p={r['purge_p']:g}: C_total {r['C_total_tpd']:.2f}, CO2-derived {r['C_from_CO2_total_tpd']:.2f} t/d, S1 X {r['S1_X_CH4']:.4f}, S1 inlet CH4 x{r['S1_inlet_CH4_ratio_vs_fresh']:.3f}, loop {r['recycle_to_S2_total_kmol_d']:.0f} kmol/d, Q1 {r['Q1_kW']:.0f} kW, H2 out {r['H2_net_exportable_tpd']:.2f}, water {r['water_total_tpd']:.1f}, it {r['iterations']}")
        L.append(f"  Part C kinetic: X_eq {x_eq:.4f}; at tau 8.905 s X_kin {s1k['X_CH4']:.4f} (approach {s1k['X_CH4']/x_eq:.3f}, C1 {tpd('C(s)', s1k['result']['Csolid_kmol_d']):.2f} t/d); new tau* {tau_star:.3f} s -> X {s1k_star['X_CH4']:.4f}, C1 {tpd('C(s)', s1k_star['result']['Csolid_kmol_d']):.2f} t/d; once-through design chain total {rC['C_total_tpd']:.2f} t/d")
    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    L.append(f"\nwall {time.time()-t0:.0f} s; unconverged: {int((~df.converged.astype(bool)).sum())}; max |closure C| {df.closure_C_rel.abs().max():.1e}")
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", OUT_CSV, OUT_TXT)

if __name__ == "__main__":
    main()
