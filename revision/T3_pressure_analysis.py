# -*- coding: utf-8 -*-
"""
T3: Pressure effect on all three stages (reviewer R1-2), after the X1 redefinition
==================================================================================
FUPROC-D-26-00269 Major Revision. Builds on K3, KIN_chain (tau1* = 8.905 s) and P1
(recycle flowsheet, tracers). Baseline outputs in CanteraResult/ are NOT touched.

Expected signs (mole-number argument): Stage 1 pyrolysis (CH4 -> C + 2H2, dn = +1)
is disfavoured by pressure; Stage 2 RWGS (dn = 0) is neutral but the parallel CO2
methanation (dn = -2) is favoured; Stage 3 methanation is favoured. Hence a
stage-wise pressure arrangement (low P pyrolysis, high P methanation) is a design
question. This script quantifies it in four parts.

Part 1  Single-stage Gibbs equilibrium at design temperatures (1200 / 950 / 650 K)
        for P in {1 atm (submitted), 1, 2, 5, 10, 20} bar. Feeds: equilibrium chain
        (Feed A) and kinetic design case (tau1* = 8.905 s, Stage 2 two-reaction
        model with phi = 0), both built at 1 atm.              EQUILIBRIUM values.
Part 2  Equilibrium recycle (P1 topology; r_CH4 = 0.95, p = 0.05, h = 0.35):
        uniform pressure P in {1 atm, 1, 2, 5, 10, 20} bar, and split pressure
        Stage 1 = 1 bar with Stage 2 = Stage 3 in {1, 5, 10, 20} bar.  EQUILIBRIUM.
Part 3  Kinetic design case (tau1* = 8.905 s, Stage 2 two-reaction model phi in
        {0, 0.1, 1}, Stage 3 submitted 3-reaction model at 650 K, tau2 = tau3 = 3 s)
        at (P1, P2, P3) in {(1,1,1), (1,5,5), (1,10,10)} bar, once-through and
        recycle (p = 0.05, r_CH4 = 0.95, h = 0.35).               KINETIC values.
        Rate constants are UNCALIBRATED and are held fixed; pressure enters the
        rates only through concentrations (Stage 2 RWGS term) and activities
        a_i = y_i P/P0 (Stage 2 methanation term, all Stage 3 reactions). NOTE:
        the elementary power-law forms make the forward methanation rate scale as
        P^5 (CO2 x H2^4); the kinetic high-pressure results therefore mainly show
        how fast the models are pushed to their equilibrium limit and are not a
        physical rate prediction.
Part 4  Compression duty (input to P3): multistage compression with intercooling
        to 313 K, max pressure ratio 3 per stage, isentropic efficiency 0.75,
        ideal gas with gamma from Cantera at the inlet state. Streams compressed:
        split arrangement - Stage 1 outlet gas, fresh CO2 and solar H2 from 1 bar
        to P2 (recycle streams are already at P2 = P3; CH4 recycle to Stage 1 is
        let down, no work recovered); uniform arrangement - fresh CH4, fresh CO2
        and solar H2 from 1 bar to P. Solar H2 is conservatively assumed at 1 bar
        (pressurised electrolysers would remove that term).

Reproducibility
---------------
  cd <repo>/revision
  ../.venv/bin/python T3_pressure_analysis.py
Imports Workflow_cantera.py (read-only) and P1_recycle_analysis.py (helpers,
tracers, Stage 2 two-reaction kinetics). Requires cantera>=3.0, numpy, scipy, pandas.

Outputs (in revision/Result/)
  T3_pressure_single_stage.csv     Part 1
  T3_pressure_eq_recycle.csv       Part 2 (+ compression duties)
  T3_pressure_kin_cases.csv        Part 3 (+ compression duties)
  T3_pressure_summary.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import P1_recycle_analysis as P1m
from P1_recycle_analysis import (FRESH_CH4, FRESH_CO2_BIO, FRESH_CO2_DAC, SOLAR_H2, TAU1_STAR, TAU2, TAU3, T3_KIN,
                                 WRF, tpd, carbon, add, scale, thermo2, k2_from_phi, tracer, tracer_netconv,
                                 tear_error, SP2, NU1, NU2, K1, R, P0)

OUT_P1 = os.path.join(RESULT, "T3_pressure_single_stage.csv")
OUT_P2 = os.path.join(RESULT, "T3_pressure_eq_recycle.csv")
OUT_P3 = os.path.join(RESULT, "T3_pressure_kin_cases.csv")
OUT_TXT = os.path.join(RESULT, "T3_pressure_summary.txt")

T1, T2, T3 = W.T_pyro, W.T_rwgs, W.T_cfr            # 1200, 950, 650 K
BAR = 1.0e5; ATM = ct.one_atm
P_GRID_BAR = [ATM / BAR, 1.0, 2.0, 5.0, 10.0, 20.0]   # first entry = 1 atm (submitted)
def plabel(pbar): return "1 atm" if abs(pbar - ATM / BAR) < 1e-9 else f"{pbar:g} bar"
REC = dict(p=0.05, r=0.95, h=0.35)                    # representative recycle condition (P1)
TOL, MAXIT = 1e-8, 500

# ---------------- stage wrappers with explicit pressure [Pa]
def stage1(ch4_kmold, mode, P):
    t = tpd("CH4", ch4_kmold)
    if mode == "kinetic":
        r = W.run_stage1_pyrolysis_kinetic(t, T1, P, tau_s=TAU1_STAR, n_steps=W.pyro_n_steps,
                                           kref=W.pyro_kref, Ea=W.pyro_Ea, Tref=W.pyro_Tref)
    else:
        r = W.run_stage1_pyrolysis(t, T1, P)
    return r["result"]["gas_kmol_d"], r["result"]["Csolid_kmol_d"], r["Q_kW"]

def stage2_kinetic_P(feed, k2, P):
    feed = W.clean_species_dict(feed); Kc1, Kp2 = thermo2(T2)
    F0 = np.array([feed.get(s, 0.0) for s in SP2]) * 1000.0 / 86400.0
    others = {k: v for k, v in feed.items() if k not in SP2}; Fo = sum(others.values()) * 1000.0 / 86400.0
    V = TAU2 * (F0.sum() + Fo) * R * T2 / P; cT = P / (R * T2)
    def rhs(v, F):
        F = np.maximum(F, 0.0); y = F / (F.sum() + Fo); c = y * cT; a = y * (P / P0)
        r1 = K1 * (c[0] * c[1] - c[2] * c[3] / Kc1)
        r2 = k2 * (a[0] * a[1] ** 4 - a[4] * a[3] ** 2 / Kp2)
        return NU1 * r1 + NU2 * r2
    sol = solve_ivp(rhs, (0.0, V), F0, method="BDF", rtol=1e-10, atol=1e-13)
    if not sol.success: raise RuntimeError(sol.message)
    F = np.maximum(sol.y[:, -1], 0.0) * 86400.0 / 1000.0
    out = W.clean_species_dict({**{s: F[i] for i, s in enumerate(SP2)}, **others})
    Q = (W.total_stream_enthalpy_J_per_day(out, 0, T2, P) - W.total_stream_enthalpy_J_per_day(feed, 0, T2, P)) / 86400e3
    return out, Q

def stage2(feed, mode, k2, P):
    if mode == "kinetic": return stage2_kinetic_P(feed, k2, P)
    r = W.run_stage2_rwgs(feed, T2, P); return r["result"]["gas_kmol_d"], r["Q_kW"]

def stage3(feed, mode, P, T=T3):
    if mode == "kinetic":
        r = W.run_stage3_cfr_kinetic(feed, T, P, eta=1.0, tau_s=TAU3, n_steps=W.cfr_n_steps)
        return r["result"]["gas_kmol_d"], r["result"]["Csolid_kmol_d"], r["Q_kW"]
    r = W.run_stage3_cfr(feed, T, P)
    if not r["result"]["converged"]: raise RuntimeError("Stage 3 Gibbs failed: " + str(r["result"]["error"]))
    return r["result"]["gas_kmol_d"], r["result"]["Csolid_kmol_d"], r["Q_kW"]

# ---------------- flowsheet (P1 topology) with per-stage pressures
def one_pass(x, cfg):
    p, r_ch4, h, mode, k2 = cfg["p"], cfg["r"], cfg["h"], cfg["mode"], cfg["k2"]
    Pa1, Pa2, Pa3 = cfg["P1_Pa"], cfg["P2_Pa"], cfg["P3_Pa"]
    ch4_rec = r_ch4 * x.get("CH4", 0.0)
    rest = dict(x); rest["CH4"] = x.get("CH4", 0.0) - ch4_rec; rest = W.clean_species_dict(rest)
    purge, to_s2 = scale(rest, p), scale(rest, 1.0 - p)
    gas1, C1, Q1 = stage1(FRESH_CH4 + ch4_rec, mode, Pa1)
    feed2 = add(gas1, {"CO2": FRESH_CO2_BIO + FRESH_CO2_DAC, "H2": SOLAR_H2}, to_s2)
    gas2, Q2 = stage2(feed2, mode, k2, Pa2)
    pw2, water2 = W.remove_species(gas2, "H2O", WRF)
    cfr_feed, permeate = W.membrane_split(pw2, {"H2": h, "CO2": 1.0, "CO": 1.0, "CH4": 1.0}, 1.0)
    gas3, C3, Q3 = stage3(cfr_feed, mode, Pa3)
    pw3, water3 = W.remove_species(gas3, "H2O", WRF)
    rec = dict(ch4_rec=ch4_rec, purge=purge, to_s2=to_s2, gas1=gas1, C1=C1, Q1=Q1, feed2=feed2, gas2=gas2, Q2=Q2,
               water2=water2, pw2=pw2, cfr_feed=cfr_feed, permeate=permeate, gas3=gas3, C3=C3, Q3=Q3, ext3={},
               water3=water3, pw3=pw3)
    return pw3, rec

def solve_recycle(cfg, x0=None):
    """Identical algorithm to P1_recycle_analysis.solve_recycle, with the pressure-aware one_pass."""
    if cfg["r"] == 0.0 and cfg["p"] == 1.0:
        g, rec = one_pass({}, cfg); rec["purge"] = rec["pw3"]
        return rec, dict(iterations=1, method="once-through", converged=True, err=0.0)
    x = dict(x0) if x0 else {}
    method, it, err_hist, budget = "direct", 0, [], MAXIT
    x_prev = g_prev = None
    while True:
        it += 1
        g, rec = one_pass(x, cfg)
        err = tear_error(x, g) if x else np.inf; err_hist.append(err)
        if err < TOL: return rec, dict(iterations=it, method=method, converged=True, err=err)
        if it >= budget:
            if method == "direct": method, budget = "wegstein", MAXIT + it
            else: return rec, dict(iterations=it, method=method, converged=False, err=err)
        if method == "direct" and it >= 20 and len(err_hist) >= 6:
            rho = (err_hist[-1] / err_hist[-6]) ** 0.2
            if 0 < rho < 1:
                if np.log(TOL / err) / np.log(rho) > budget - it: method, budget = "wegstein", MAXIT + it
            elif rho >= 1: method, budget = "wegstein", MAXIT + it
        if method == "wegstein" and x_prev is not None and g_prev is not None:
            x_new = {}
            for k in set(g) | set(x):
                xk, xk1, gk, gk1 = x.get(k, 0.0), x_prev.get(k, 0.0), g.get(k, 0.0), g_prev.get(k, 0.0)
                if abs(xk - xk1) > 1e-14 * max(xk, 1.0):
                    s_ = (gk - gk1) / (xk - xk1); q = s_ / (s_ - 1.0) if abs(s_ - 1.0) > 1e-12 else 0.0
                    q = min(0.0, max(-10.0, q))
                else: q = 0.0
                x_new[k] = max(0.0, q * xk + (1.0 - q) * gk)
        else: x_new = dict(g)
        x_prev, g_prev, x = x, g, W.clean_species_dict(x_new)

# ---------------- compression duty (Part 4)
def compress_kW(stream, P_in_Pa, P_out_Pa, T_in=313.15, eta_s=0.75, max_ratio=3.0):
    """Multistage intercooled compression of an ideal-gas stream [kmol/d]; returns (kW, n_stages, gamma)."""
    stream = W.clean_species_dict(stream); n = sum(stream.values())
    if n <= 0 or P_out_Pa <= P_in_Pa * 1.0000001: return 0.0, 0, np.nan
    g = ct.Solution("gri30.yaml"); x, _ = W.normalize_mole_dict(stream); g.TPX = T_in, P_in_Pa, x
    gamma = g.cp_mole / g.cv_mole
    Pr = P_out_Pa / P_in_Pa; ns = int(np.ceil(np.log(Pr) / np.log(max_ratio) - 1e-12)); ns = max(ns, 1)
    pr = Pr ** (1.0 / ns); k = (gamma - 1.0) / gamma
    w = ns * (1.0 / k) * ct.gas_constant * T_in * (pr ** k - 1.0) / eta_s      # J/kmol
    return w * n / 86400.0 / 1000.0, ns, gamma

def compression_terms(rec, cfg):
    Pa1, Pa2 = cfg["P1_Pa"], cfg["P2_Pa"]
    terms = {}
    # split arrangement: whatever enters Stage 2 from 1 bar must be raised to P2
    terms["W_S1out_to_S2_kW"], terms["n_st_S1out"], _ = compress_kW(rec["gas1"], min(Pa1, BAR), Pa2)
    terms["W_freshCO2_kW"], _, _ = compress_kW({"CO2": FRESH_CO2_BIO + FRESH_CO2_DAC}, BAR, Pa2)
    terms["W_solarH2_kW"], _, _ = compress_kW({"H2": SOLAR_H2}, BAR, Pa2)
    terms["W_freshCH4_to_S1_kW"], _, _ = compress_kW({"CH4": FRESH_CH4}, BAR, Pa1)
    terms["W_compression_total_kW"] = terms["W_S1out_to_S2_kW"] + terms["W_freshCO2_kW"] + terms["W_solarH2_kW"] + terms["W_freshCH4_to_S1_kW"]
    terms["W_compression_excl_solarH2_kW"] = terms["W_compression_total_kW"] - terms["W_solarH2_kW"]
    terms["W_compression_share_of_5MW"] = terms["W_compression_total_kW"] / W.solar_power_kW
    return terms

# ---------------- rows
def make_row(cfg, rec, info, label):
    f1, f2 = tracer(rec, cfg["p"], cfg["r"]); nc = tracer_netconv(rec, cfg["p"], cfg["r"])
    C1, C3 = rec["C1"], rec["C3"]
    solid_nc = C1 * nc["f_C1"] + C3 * nc["f_C3"]; solid_mix = C1 * f1 + C3 * f2
    purge = rec["purge"]
    ins = W.elemental_inventory_total({"CH4": FRESH_CH4, "CO2": FRESH_CO2_BIO + FRESH_CO2_DAC, "H2": SOLAR_H2}, 0.0)
    outs = W.elemental_inventory_total(add(rec["water2"], rec["water3"], rec["permeate"], purge), C1 + C3)
    closure = {e: (outs[e] - ins[e]) / ins[e] for e in W.ELEMENTS}
    feed2, gas2, cf, gas3 = rec["feed2"], rec["gas2"], rec["cfr_feed"], rec["gas3"]
    row = dict(case=label, value_type="kinetic" if cfg["mode"] == "kinetic" else "equilibrium", mode=cfg["mode"],
               P1_bar=cfg["P1_Pa"] / BAR, P2_bar=cfg["P2_Pa"] / BAR, P3_bar=cfg["P3_Pa"] / BAR,
               phi=cfg.get("phi", np.nan), purge_p=cfg["p"], r_CH4=cfg["r"], h_H2_to_CFR=cfg["h"],
               converged=info["converged"], iterations=info["iterations"], method=info["method"],
               S1_inlet_CH4_kmol_d=FRESH_CH4 + rec["ch4_rec"], S1_inlet_CH4_ratio_vs_fresh=(FRESH_CH4 + rec["ch4_rec"]) / FRESH_CH4,
               S1_X_CH4=1 - rec["gas1"].get("CH4", 0) / (FRESH_CH4 + rec["ch4_rec"]),
               S2_inlet_total_kmol_d=sum(feed2.values()), recycle_to_S2_total_kmol_d=sum(rec["to_s2"].values()),
               S2_X_CO2_per_pass=1 - gas2.get("CO2", 0) / feed2["CO2"], S2_xi2_CH4_formed_kmol_d=gas2.get("CH4", 0) - feed2.get("CH4", 0),
               S3_CH4_formed_kmol_d=gas3.get("CH4", 0) - cf.get("CH4", 0),
               S3_COx_conversion=1 - (gas3.get("CO", 0) + gas3.get("CO2", 0)) / max(cf.get("CO", 0) + cf.get("CO2", 0), 1e-30),
               C_stage1_tpd=tpd("C(s)", C1), C_stage3_tpd=tpd("C(s)", C3), C_total_tpd=tpd("C(s)", C1 + C3),
               C_from_CO2_total_tpd=tpd("C(s)", solid_nc[1] + solid_nc[2]),
               fresh_CO2_carbon_fixed_fraction=(solid_nc[1] + solid_nc[2]) / (FRESH_CO2_BIO + FRESH_CO2_DAC),
               C_from_CO2_total_tpd_mixing_tracer_upper=tpd("C(s)", solid_mix[1] + solid_mix[2]),
               purge_carbon_tpd=tpd("C(s)", carbon(purge)), purge_carbon_as_CO2_if_oxidised_tpd=carbon(purge) * W.MW["CO2"] / 1000.0,
               H2_net_exportable_tpd=tpd("H2", rec["permeate"].get("H2", 0) + purge.get("H2", 0)),
               water_total_tpd=tpd("H2O", rec["water2"].get("H2O", 0) + rec["water3"].get("H2O", 0)),
               Q1_kW=rec["Q1"], Q2_kW=rec["Q2"], Q3_kW=rec["Q3"], Q_total_kW=rec["Q1"] + rec["Q2"] + rec["Q3"],
               closure_C_rel=closure["C"], closure_H_rel=closure["H"], closure_O_rel=closure["O"])
    row.update(compression_terms(rec, cfg))
    return row

# ---------------- Part 1: single-stage equilibrium
def part1():
    rows = []
    # reference feeds built at 1 atm
    s1e = W.run_stage1_pyrolysis(W.CH4_tpd, T1, ATM)
    feedA2, _ = W.build_rwgs_feed(s1e, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    s2e = W.run_stage2_rwgs(feedA2, T2, ATM)
    feedA3 = W.build_cfr_feed(s2e["result"]["gas_kmol_d"], WRF, W.h2_to_cfr_frac, 1, 1, 1)["cfr_feed"]
    gas1k, C1k, _ = stage1(FRESH_CH4, "kinetic", ATM)
    feedK2 = add(gas1k, {"CO2": FRESH_CO2_BIO + FRESH_CO2_DAC, "H2": SOLAR_H2})
    gas2k, _ = stage2_kinetic_P(feedK2, 0.0, ATM)
    feedK3 = W.build_cfr_feed(gas2k, WRF, W.h2_to_cfr_frac, 1, 1, 1)["cfr_feed"]
    for pbar in P_GRID_BAR:
        P = pbar * BAR
        g, C, Q = stage1(FRESH_CH4, "equilibrium", P)
        rows.append(dict(stage="Stage 1 pyrolysis 1200 K", feed="fresh CH4 (same for both chains)", P_bar=pbar, P_label=plabel(pbar),
                         X_CH4=1 - g.get("CH4", 0) / FRESH_CH4, C_solid_tpd=tpd("C(s)", C), H2_out_kmol_d=g.get("H2", 0), Q_kW=Q))
        for tag, f2 in [("Feed A (eq chain)", feedA2), ("kinetic design case (tau1*=8.905 s)", feedK2)]:
            g, Q = stage2(f2, "equilibrium", 0.0, P)
            rows.append(dict(stage="Stage 2 RWGS 950 K", feed=tag, P_bar=pbar, P_label=plabel(pbar),
                             X_CO2=1 - g.get("CO2", 0) / f2["CO2"], xi1_CO_kmol_d=g.get("CO", 0), xi2_CH4_formed_kmol_d=g.get("CH4", 0) - f2.get("CH4", 0),
                             H2O_out_kmol_d=g.get("H2O", 0), Q_kW=Q))
        for tag, f3 in [("Feed A chain CFR feed", feedA3), ("kinetic design case CFR feed (phi=0)", feedK3)]:
            g, C, Q = stage3(f3, "equilibrium", P)
            rows.append(dict(stage="Stage 3 methanation/CFR 650 K", feed=tag, P_bar=pbar, P_label=plabel(pbar),
                             CH4_formed_kmol_d=g.get("CH4", 0) - f3.get("CH4", 0),
                             COx_conversion=1 - (g.get("CO", 0) + g.get("CO2", 0)) / (f3.get("CO", 0) + f3.get("CO2", 0)),
                             C_solid_tpd=tpd("C(s)", C), H2O_out_kmol_d=g.get("H2O", 0), Q_kW=Q))
    return pd.DataFrame(rows)

# ---------------- Parts 2 and 3 workers
def run_case(args):
    label, cfg = args
    t0 = time.time()
    rec, info = solve_recycle(cfg)
    row = make_row(cfg, rec, info, label); row["wall_s"] = time.time() - t0
    return row

def main():
    t0 = time.time()
    df1 = part1(); df1.to_csv(OUT_P1, index=False)
    # Part 2 cases
    cases2 = []
    for pbar in P_GRID_BAR:
        cases2.append((f"eq uniform {plabel(pbar)}", dict(mode="equilibrium", k2=0.0, P1_Pa=pbar * BAR, P2_Pa=pbar * BAR, P3_Pa=pbar * BAR, **REC)))
    for pbar in [1.0, 5.0, 10.0, 20.0]:
        cases2.append((f"eq split S1=1 bar, S2=S3={pbar:g} bar", dict(mode="equilibrium", k2=0.0, P1_Pa=BAR, P2_Pa=pbar * BAR, P3_Pa=pbar * BAR, **REC)))
    # Part 3 cases
    k2_map = {phi: k2_from_phi(phi) for phi in [0.0, 0.1, 1.0]}
    cases3 = []
    for (p1, p2, p3) in [(1, 1, 1), (1, 5, 5), (1, 10, 10)]:
        for phi in [0.0, 0.1, 1.0]:
            base = dict(mode="kinetic", k2=k2_map[phi], phi=phi, P1_Pa=p1 * BAR, P2_Pa=p2 * BAR, P3_Pa=p3 * BAR)
            cases3.append((f"kin once-through ({p1},{p2},{p3}) bar phi={phi:g}", dict(base, p=1.0, r=0.0, h=REC["h"])))
            cases3.append((f"kin recycle ({p1},{p2},{p3}) bar phi={phi:g}", dict(base, **REC)))
    with Pool(min(12, cpu_count())) as pool:
        rows2 = pool.map(run_case, cases2); rows3 = pool.map(run_case, cases3)
    df2 = pd.DataFrame(rows2); df3 = pd.DataFrame(rows3)
    df2.to_csv(OUT_P2, index=False); df3.to_csv(OUT_P3, index=False)
    L = [f"T3 pressure analysis (wall {time.time()-t0:.0f} s). Part 1/2 EQUILIBRIUM; Part 3 KINETIC (uncalibrated rate constants)."]
    with pd.option_context("display.width", 300, "display.max_columns", 60, "display.max_rows", 500, "display.float_format", lambda v: f"{v:.4g}"):
        L.append("\n===== Part 1: single-stage Gibbs =====\n" + df1.to_string(index=False))
        c2 = ["case", "converged", "iterations", "S1_X_CH4", "S1_inlet_CH4_ratio_vs_fresh", "S2_X_CO2_per_pass", "S2_xi2_CH4_formed_kmol_d", "S3_CH4_formed_kmol_d",
              "C_total_tpd", "C_from_CO2_total_tpd", "purge_carbon_tpd", "recycle_to_S2_total_kmol_d", "Q1_kW", "Q2_kW", "Q3_kW", "H2_net_exportable_tpd",
              "water_total_tpd", "W_compression_total_kW", "W_compression_excl_solarH2_kW", "closure_C_rel"]
        L.append("\n===== Part 2: equilibrium recycle (r=0.95, p=0.05, h=0.35) =====\n" + df2[c2].to_string(index=False))
        c3 = ["case", "converged", "iterations", "S1_X_CH4", "S2_X_CO2_per_pass", "S2_xi2_CH4_formed_kmol_d", "S3_CH4_formed_kmol_d", "S3_COx_conversion",
              "C_stage3_tpd", "C_total_tpd", "C_from_CO2_total_tpd", "purge_carbon_tpd", "recycle_to_S2_total_kmol_d", "Q1_kW", "Q2_kW", "Q3_kW",
              "H2_net_exportable_tpd", "water_total_tpd", "W_compression_total_kW", "closure_C_rel"]
        L.append("\n===== Part 3: kinetic design case =====\n" + df3[c3].to_string(index=False))
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", OUT_P1, OUT_P2, OUT_P3, OUT_TXT)

if __name__ == "__main__":
    main()
