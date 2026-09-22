# -*- coding: utf-8 -*-
"""
T3b: Sensitivity of the kinetic pressure result to the pressure order of the rate laws
======================================================================================
FUPROC-D-26-00269 Major Revision (R1-2, follow-up to T3_pressure_analysis.py).

Motivation: in T3 the kinetic design case reached its equilibrium limit as soon as
Stages 2 and 3 were run at 5 bar. The submitted elementary power-law form makes the
forward CO2-methanation rate scale as a_CO2 a_H2^4 ~ P^5 at fixed composition, far
above typical apparent pressure orders on real catalysts (about 0.5-1.5). This script
re-runs the kinetic pressure cases with the pressure order forced to n in
{0.5, 1, 1.5} and the native value for comparison.

Method: every affected rate is multiplied by a pressure-order correction factor
    r_j -> (P/P0)^(n - m_j) * r_j ,   P0 = 1 atm (reference of the activities)
where m_j is the native pressure order of reaction j at fixed composition
(RWGS concentration form: 2; CO2 methanation: 5; CO + H2 -> C + H2O: 2;
CH4 -> C + 2H2: 1). The factor multiplies the whole net rate, so the equilibrium
point of each reaction (bracket = 0) and the composition dependence are unchanged;
only how the forward rate scales with total pressure changes. At 1 atm the models
are identical to the submitted ones.
  Variant V1 "methanation only": factor applied to CO2 methanation in Stage 2 (R2)
             and in Stage 3 (CO2_methanation); RWGS, CO-carbon, CH4-cracking native.
  Variant V2 "all reactions"   : factor applied to every reaction of Stages 2 and 3.
Stage 1 is at 1 bar in all cases and is unaffected.

Cases: (P1, P2, P3) in {(1,5,5), (1,10,10)} bar x n in {0.5, 1, 1.5, native}
       x phi in {0, 0.1, 1} x variant {V1, V2} x {once-through, recycle p=0.05}.
Recycle condition as in T3/P1: r_CH4 = 0.95, p = 0.05, h = 0.35.
ALL VALUES KINETIC; all rate constants UNCALIBRATED (k2 fixed per phi as before).

Reproducibility
---------------
  cd <repo>/revision
  ../.venv/bin/python T3b_pressure_order_sensitivity.py      (~10 min, 12 processes)
Imports Workflow_cantera.py, P1_recycle_analysis.py and T3_pressure_analysis.py read-only.

Outputs (in revision/Result/)
  T3b_pressure_order_sensitivity.csv
  T3b_pressure_order_summary.txt
"""
import os, sys, time
import numpy as np
import pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import T3_pressure_analysis as T3m
from P1_recycle_analysis import (FRESH_CH4, FRESH_CO2_BIO, FRESH_CO2_DAC, SOLAR_H2, TAU1_STAR, TAU2, TAU3, WRF,
                                 add, scale, thermo2, k2_from_phi, tear_error, SP2, NU1, NU2, K1, R, P0)

OUT_CSV = os.path.join(RESULT, "T3b_pressure_order_sensitivity.csv")
OUT_TXT = os.path.join(RESULT, "T3b_pressure_order_summary.txt")
T1, T2, T3 = W.T_pyro, W.T_rwgs, W.T_cfr
BAR = 1.0e5; ATM = ct.one_atm
NATIVE = {"rwgs": 2.0, "meth": 5.0, "co_carb": 2.0, "ch4_crack": 1.0}
REC = dict(p=0.05, r=0.95, h=0.35)
TOL, MAXIT = 1e-8, 500

def order_factors(n, variant, P):
    """Multiplicative factors per reaction for pressure order n (None = native)."""
    if n is None: return {k: 1.0 for k in NATIVE}
    pr = P / P0
    f = {k: 1.0 for k in NATIVE}
    f["meth"] = pr ** (n - NATIVE["meth"])
    if variant == "V2":
        for k in ["rwgs", "co_carb", "ch4_crack"]: f[k] = pr ** (n - NATIVE[k])
    return f

def stage2_kinetic_n(feed, k2, P, f):
    feed = W.clean_species_dict(feed); Kc1, Kp2 = thermo2(T2)
    F0 = np.array([feed.get(s, 0.0) for s in SP2]) * 1000.0 / 86400.0
    others = {k: v for k, v in feed.items() if k not in SP2}; Fo = sum(others.values()) * 1000.0 / 86400.0
    V = TAU2 * (F0.sum() + Fo) * R * T2 / P; cT = P / (R * T2)
    k1 = K1 * f["rwgs"]; k2e = k2 * f["meth"]
    def rhs(v, F):
        F = np.maximum(F, 0.0); y = F / (F.sum() + Fo); c = y * cT; a = y * (P / P0)
        r1 = k1 * (c[0] * c[1] - c[2] * c[3] / Kc1)
        r2 = k2e * (a[0] * a[1] ** 4 - a[4] * a[3] ** 2 / Kp2)
        return NU1 * r1 + NU2 * r2
    sol = solve_ivp(rhs, (0.0, V), F0, method="BDF", rtol=1e-10, atol=1e-13)
    if not sol.success: raise RuntimeError(sol.message)
    F = np.maximum(sol.y[:, -1], 0.0) * 86400.0 / 1000.0
    out = W.clean_species_dict({**{s: F[i] for i, s in enumerate(SP2)}, **others})
    Q = (W.total_stream_enthalpy_J_per_day(out, 0, T2, P) - W.total_stream_enthalpy_J_per_day(feed, 0, T2, P)) / 86400e3
    return out, Q

def stage3_kinetic_n(feed, P, f):
    # Arrhenius factor is 1 at Tref = T_cfr = 650 K, so scaling kref scales the rate exactly.
    r = W.run_stage3_cfr_kinetic(feed, T3, P, eta=1.0, tau_s=TAU3, n_steps=W.cfr_n_steps,
                                 kref_co2_meth=W.cfr_kref_co2_meth * f["meth"],
                                 kref_co_carb=W.cfr_kref_co_carb * f["co_carb"],
                                 kref_ch4_carb=W.cfr_kref_ch4_carb * f["ch4_crack"])
    return r["result"]["gas_kmol_d"], r["result"]["Csolid_kmol_d"], r["Q_kW"]

def one_pass(x, cfg):
    p, r_ch4, h, k2 = cfg["p"], cfg["r"], cfg["h"], cfg["k2"]
    Pa1, Pa2, Pa3 = cfg["P1_Pa"], cfg["P2_Pa"], cfg["P3_Pa"]
    f2 = order_factors(cfg["n"], cfg["variant"], Pa2); f3 = order_factors(cfg["n"], cfg["variant"], Pa3)
    ch4_rec = r_ch4 * x.get("CH4", 0.0)
    rest = dict(x); rest["CH4"] = x.get("CH4", 0.0) - ch4_rec; rest = W.clean_species_dict(rest)
    purge, to_s2 = scale(rest, p), scale(rest, 1.0 - p)
    gas1, C1, Q1 = T3m.stage1(FRESH_CH4 + ch4_rec, "kinetic", Pa1)
    feed2 = add(gas1, {"CO2": FRESH_CO2_BIO + FRESH_CO2_DAC, "H2": SOLAR_H2}, to_s2)
    gas2, Q2 = stage2_kinetic_n(feed2, k2, Pa2, f2)
    pw2, water2 = W.remove_species(gas2, "H2O", WRF)
    cfr_feed, permeate = W.membrane_split(pw2, {"H2": h, "CO2": 1.0, "CO": 1.0, "CH4": 1.0}, 1.0)
    gas3, C3, Q3 = stage3_kinetic_n(cfr_feed, Pa3, f3)
    pw3, water3 = W.remove_species(gas3, "H2O", WRF)
    rec = dict(ch4_rec=ch4_rec, purge=purge, to_s2=to_s2, gas1=gas1, C1=C1, Q1=Q1, feed2=feed2, gas2=gas2, Q2=Q2,
               water2=water2, pw2=pw2, cfr_feed=cfr_feed, permeate=permeate, gas3=gas3, C3=C3, Q3=Q3, ext3={},
               water3=water3, pw3=pw3)
    return pw3, rec

def solve_recycle(cfg, x0=None):
    """Same algorithm as P1/T3 (successive substitution -> bounded Wegstein), with this module's one_pass."""
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

def run_case(args):
    label, cfg = args; t0 = time.time()
    rec, info = solve_recycle(cfg)
    row = T3m.make_row(cfg, rec, info, label)
    row.update(pressure_order_n="native" if cfg["n"] is None else cfg["n"], variant=cfg["variant"],
               factor_meth_at_P2=order_factors(cfg["n"], cfg["variant"], cfg["P2_Pa"])["meth"],
               factor_rwgs_at_P2=order_factors(cfg["n"], cfg["variant"], cfg["P2_Pa"])["rwgs"], wall_s=time.time() - t0)
    return row

def main():
    t0 = time.time()
    k2_map = {phi: k2_from_phi(phi) for phi in [0.0, 0.1, 1.0]}
    cases = []
    for (p1, p2, p3) in [(1, 5, 5), (1, 10, 10)]:
        for n in [0.5, 1.0, 1.5, None]:
            for variant in (["V1", "V2"] if n is not None else ["V1"]):
                for phi in [0.0, 0.1, 1.0]:
                    base = dict(mode="kinetic", k2=k2_map[phi], phi=phi, n=n, variant=variant,
                                P1_Pa=p1 * BAR, P2_Pa=p2 * BAR, P3_Pa=p3 * BAR)
                    nl = "native" if n is None else f"n={n:g}"
                    cases.append((f"kin once-through ({p1},{p2},{p3}) {nl} {variant} phi={phi:g}", dict(base, p=1.0, r=0.0, h=REC["h"])))
                    cases.append((f"kin recycle ({p1},{p2},{p3}) {nl} {variant} phi={phi:g}", dict(base, **REC)))
    with Pool(min(12, cpu_count())) as pool:
        rows = pool.map(run_case, cases)
    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    cols = ["case", "converged", "iterations", "factor_meth_at_P2", "S2_X_CO2_per_pass", "S2_xi2_CH4_formed_kmol_d", "S3_CH4_formed_kmol_d",
            "C_stage3_tpd", "C_total_tpd", "C_from_CO2_total_tpd", "fresh_CO2_carbon_fixed_fraction", "purge_carbon_tpd",
            "recycle_to_S2_total_kmol_d", "Q1_kW", "Q2_kW", "Q3_kW", "H2_net_exportable_tpd", "closure_C_rel"]
    L = [f"T3b pressure-order sensitivity (wall {time.time()-t0:.0f} s). ALL KINETIC, uncalibrated. Unconverged: {int((~df.converged.astype(bool)).sum())}"]
    with pd.option_context("display.width", 320, "display.max_columns", 60, "display.max_rows", 500, "display.float_format", lambda v: f"{v:.4g}"):
        L.append(df[cols].to_string(index=False))
        piv = df[df.purge_p < 1].pivot_table(index=["P2_bar", "variant", "pressure_order_n"], columns="phi", values="C_from_CO2_total_tpd")
        L.append("\n===== Recycle: CO2-derived solid carbon [t/d] by (P2, variant, n) x phi =====\n" + piv.to_string())
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", OUT_CSV, OUT_TXT)

if __name__ == "__main__":
    main()
