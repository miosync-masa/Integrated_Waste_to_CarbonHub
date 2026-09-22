# -*- coding: utf-8 -*-
"""
P1 (+P2): Quantitative recycle analysis with carbon-origin tracking
====================================================================
FUPROC-D-26-00269 Major Revision, reviewer R1-5. Builds on K3 and the kinetic
design case (KIN_chain_design_case.py, tau1* = 8.905 s).

Question: if Stage 3 (CFR) is used as a METHANATOR rather than as a carbon
depositor, how much CO2-derived carbon ends up as solid carbon via the indirect
route  CO2 -> CO -> CH4 -> (recycle) -> Stage 1 pyrolysis -> C(s)  under recycle?

Recycle topology (identical for all cases)
  Stage 3 outlet -> water removal (0.95) -> CH4 separation (recovery r_CH4, ideal,
  CH4-rich stream = pure CH4) -> CH4-rich stream to Stage 1 inlet;
  remainder (CO, CO2, H2, residual CH4, residual H2O, traces) -> purge fraction p
  leaves the system, (1-p) goes to the Stage 2 inlet.
  Between Stage 2 and 3: water removal 0.95, H2 split fraction h to CFR (P2 sweep),
  CO2/CO/CH4 fully to CFR, as in the submitted baseline. Separation work is NOT
  included here (task P3); separator duties are only recorded as flows.
  Once-through reference = same configuration with r_CH4 = 0, p = 1.

Three tiers
  Tier 1  Stoichiometric upper bound (no reactor model): all carbon leaving the
          once-through design-case Stage 3 (CO + CO2 + CH4) is methanated and
          cracked in Stage 1 at the design approach (X_CH4 = 0.90 x X_eq); plus the
          full-recycle asymptote (p -> 0, all fresh carbon to solid).
  Tier 2  Equilibrium recycle: all three stages Gibbs (GRI-3.0 + graphite where
          applicable). Sweep p x r_CH4 x h x T3 in {550, 600, 650} K.  EQUILIBRIUM.
  Tier 3  Kinetic recycle (design case): Stage 1 reduced model tau1 = 8.905 s,
          Stage 2 two-reaction reduced model (K3) tau2 = 3 s with phi in {0, 0.1, 1},
          Stage 3 submitted 3-reaction reduced model tau3 = 3 s at 650 K.
          Sweep p x r_CH4 x h x phi.  KINETIC. ALL RATE CONSTANTS UNCALIBRATED.
          k2 is fixed per phi from the Feed A inlet as in KIN_chain_design_case.py.

Convergence: tear stream = Stage 3 outlet after water removal. Successive
substitution until the max relative change of the tear vector (floor: 1e-6 of
the total tear flow) < 1e-8, max 500 iterations. If the observed contraction
ratio predicts that the budget will not suffice, the loop switches to bounded
Wegstein acceleration (q in [-10, 0], component-wise) for up to 500 further
iterations. Cases are warm-started from the neighbouring converged case.

Carbon-origin tracking, two tracer variants, both solved as fixed points on the
converged flows (origins: waste CH4 / biogas CO2 / DAC CO2):
  (a) "mixing" tracer  - perfect mixing at every node: one origin vector per
      stream applied to all carbon atoms; reactor outlets inherit the
      carbon-weighted inlet vector. This scrambles carbon among species even when
      no reaction connects them, so it is an UPPER BOUND on CO2-derived solid
      carbon (it assigns CO2 origin to solid carbon even when phi = 0 and Stage 3
      is inert).
  (b) "net-conversion" tracer - origin vectors are carried PER SPECIES; at a
      reactor the unreacted part of each species (min(in, out)) keeps its inlet
      origin, the carbon of net-consumed species is pooled and mixed, and the
      pool origin is assigned to net-produced species and to solid carbon.
      Splitters and mixers act per species. This is the physically meaningful
      estimate and is used for the headline numbers.

Element closure: per-stage closure of the reactor models is ~1e-15; the whole-system
closure reported per case is limited by the tear tolerance (1e-8 relative on the
recycle stream), i.e. up to ~1e-7 relative when the recycle-to-fresh ratio is large.

Stage duties Q_i are isothermal reaction duties at the stage temperature
(enthalpy out - in at T_i), exactly as in the baseline; sensible heat of mixing
recycle streams is not included.

Reproducibility
---------------
  cd <repo>/revision
  ../.venv/bin/python P1_recycle_analysis.py          # full sweep (~10-20 min, 12 processes)
  ../.venv/bin/python P1_recycle_analysis.py --quick  # subset for a smoke test
Requires cantera>=3.0, numpy, scipy, pandas, matplotlib. Imports Workflow_cantera.py
read-only. Baseline outputs in CanteraResult/ are NOT touched.

Outputs (in revision/Result/)
  P1_recycle_tier1_stoichiometric.csv
  P1_recycle_equilibrium_sweep.csv      Tier 2 (180 cases + once-through refs)
  P1_recycle_kinetic_sweep.csv          Tier 3 (180 cases + once-through refs)
  P1_recycle_records.pkl                converged stream records of every case (for re-analysis)
  P1_recycle_summary.txt
  P1_recycle_co2_carbon.png
"""
import os, sys, time, itertools, argparse
import numpy as np
import pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W

OUT_T1  = os.path.join(RESULT, "P1_recycle_tier1_stoichiometric.csv")
OUT_EQ  = os.path.join(RESULT, "P1_recycle_equilibrium_sweep.csv")
OUT_KIN = os.path.join(RESULT, "P1_recycle_kinetic_sweep.csv")
OUT_TXT = os.path.join(RESULT, "P1_recycle_summary.txt")
OUT_PNG = os.path.join(RESULT, "P1_recycle_co2_carbon.png")

# ---------------- fixed process data (submitted baseline)
T1, P1 = W.T_pyro, W.P_pyro
T2, P2 = W.T_rwgs, W.P_rwgs
P3 = W.P_cfr
TAU1_STAR = 8.905          # design case (KIN_chain_design_case.py)
TAU2, TAU3 = W.rwgs_tau_s, W.cfr_tau_s
T3_KIN = 650.0
WRF = W.water_remove_frac  # 0.95, used after Stage 2 and after Stage 3
FRESH_CH4 = W.tpd_to_kmol_per_day(W.CH4_tpd, W.MW["CH4"])                 # 2992 kmol/d
FRESH_CO2_BIO = W.tpd_to_kmol_per_day(W.CO2_from_biogas_tpd, W.MW["CO2"])  # 727
FRESH_CO2_DAC = W.tpd_to_kmol_per_day(W.CO2_from_DAC_tpd, W.MW["CO2"])     # 227
SOLAR_H2 = W.solar_h2_kmol_per_day(W.solar_power_kW, W.electrolyzer_eff)    # 1250
MWC = W.MW_C_SOLID

P_GRID = [0.2, 0.1, 0.05, 0.02, 0.01]      # descending for warm start
R_GRID = [0.9, 0.95, 0.99]
H_GRID = [0.35, 0.5, 0.75, 1.0]
T3_GRID_EQ = [550.0, 600.0, 650.0]
PHI_GRID = [0.0, 0.1, 1.0]
TOL, MAXIT = 1e-8, 500

R = ct.gas_constant / 1000.0; P0 = ct.one_atm
SP2 = ["CO2", "H2", "CO", "H2O", "CH4"]
NU1 = np.array([-1, -1, +1, +1, 0], float); NU2 = np.array([-1, -4, 0, +2, +1], float)
K1 = W.rwgs_kref

def tpd(sp, n): return n * (MWC if sp == "C(s)" else W.MW[sp]) / 1000.0
def carbon(stream): return W.elemental_inventory_gas(stream)["C"]
def add(*streams):
    out = {}
    for s in streams:
        for k, v in s.items(): out[k] = out.get(k, 0.0) + v
    return W.clean_species_dict(out)
def scale(s, f): return W.clean_species_dict({k: v * f for k, v in s.items()})

# ---------------- Stage 2 two-reaction kinetic model (identical rate law to K3 / KIN_chain scripts)
_THERMO2 = {}
def thermo2(T):
    if T not in _THERMO2:
        g = ct.Solution("gri30.yaml"); g.TP = T, P0
        gRT = g.standard_gibbs_RT; i = [g.species_index(s) for s in SP2]
        _THERMO2[T] = (float(np.exp(-NU1 @ gRT[i])), float(np.exp(-NU2 @ gRT[i])))
    return _THERMO2[T]

def k2_from_phi(phi):
    """k2 fixed from the Feed A (equilibrium-chain) Stage 2 inlet, as in KIN_chain_design_case.py."""
    s1e = W.run_stage1_pyrolysis(W.CH4_tpd, T1, P1)
    rfA, _ = W.build_rwgs_feed(s1e, W.CO2_total_tpd, W.solar_power_kW, W.electrolyzer_eff)
    F0 = np.array([rfA.get(s, 0.0) for s in SP2]); y = F0 / F0.sum()
    c = y * P2 / (R * T2); a = y * (P2 / P0)
    return phi * (K1 * c[0] * c[1]) / (a[0] * a[1] ** 4)

def stage2_kinetic(feed, k2):
    feed = W.clean_species_dict(feed); Kc1, Kp2 = thermo2(T2)
    F0 = np.array([feed.get(s, 0.0) for s in SP2]) * 1000.0 / 86400.0
    others = {k: v for k, v in feed.items() if k not in SP2}
    Fo = sum(others.values()) * 1000.0 / 86400.0
    V = TAU2 * (F0.sum() + Fo) * R * T2 / P2; cT = P2 / (R * T2)
    def rhs(v, F):
        F = np.maximum(F, 0.0); y = F / (F.sum() + Fo); c = y * cT; a = y * (P2 / P0)
        r1 = K1 * (c[0] * c[1] - c[2] * c[3] / Kc1)
        r2 = k2 * (a[0] * a[1] ** 4 - a[4] * a[3] ** 2 / Kp2)
        return NU1 * r1 + NU2 * r2
    sol = solve_ivp(rhs, (0.0, V), F0, method="BDF", rtol=1e-10, atol=1e-13)
    if not sol.success: raise RuntimeError(sol.message)
    F = np.maximum(sol.y[:, -1], 0.0) * 86400.0 / 1000.0
    out = W.clean_species_dict({**{s: F[i] for i, s in enumerate(SP2)}, **others})
    Q = (W.total_stream_enthalpy_J_per_day(out, 0, T2, P2) - W.total_stream_enthalpy_J_per_day(feed, 0, T2, P2)) / 86400e3
    return out, Q

# ---------------- stage wrappers (feed dicts in kmol/d)
def stage1(ch4_kmold, mode):
    t = tpd("CH4", ch4_kmold)
    if mode == "kinetic":
        r = W.run_stage1_pyrolysis_kinetic(t, T1, P1, tau_s=TAU1_STAR, n_steps=W.pyro_n_steps,
                                           kref=W.pyro_kref, Ea=W.pyro_Ea, Tref=W.pyro_Tref)
    else:
        r = W.run_stage1_pyrolysis(t, T1, P1)
    return r["result"]["gas_kmol_d"], r["result"]["Csolid_kmol_d"], r["Q_kW"]

def stage2(feed, mode, k2):
    if mode == "kinetic": return stage2_kinetic(feed, k2)
    r = W.run_stage2_rwgs(feed, T2, P2); return r["result"]["gas_kmol_d"], r["Q_kW"]

def stage3(feed, mode, T3):
    if mode == "kinetic":
        r = W.run_stage3_cfr_kinetic(feed, T3, P3, eta=1.0, tau_s=TAU3, n_steps=W.cfr_n_steps)
        return r["result"]["gas_kmol_d"], r["result"]["Csolid_kmol_d"], r["Q_kW"], r.get("extent_kmol_d", {})
    r = W.run_stage3_cfr(feed, T3, P3)
    if not r["result"]["converged"]: raise RuntimeError("Stage 3 Gibbs failed: " + str(r["result"]["error"]))
    return r["result"]["gas_kmol_d"], r["result"]["Csolid_kmol_d"], r["Q_kW"], {}

# ---------------- one pass of the flowsheet given the tear stream x (post-water Stage 3 outlet)
def one_pass(x, cfg):
    p, r_ch4, h, mode, T3, k2 = cfg["p"], cfg["r"], cfg["h"], cfg["mode"], cfg["T3"], cfg["k2"]
    ch4_rec = r_ch4 * x.get("CH4", 0.0)
    rest = dict(x); rest["CH4"] = x.get("CH4", 0.0) - ch4_rec; rest = W.clean_species_dict(rest)
    purge, to_s2 = scale(rest, p), scale(rest, 1.0 - p)
    gas1, C1, Q1 = stage1(FRESH_CH4 + ch4_rec, mode)
    feed2 = add(gas1, {"CO2": FRESH_CO2_BIO + FRESH_CO2_DAC, "H2": SOLAR_H2}, to_s2)
    gas2, Q2 = stage2(feed2, mode, k2)
    pw2, water2 = W.remove_species(gas2, "H2O", WRF)
    cfr_feed, permeate = W.membrane_split(pw2, {"H2": h, "CO2": 1.0, "CO": 1.0, "CH4": 1.0}, 1.0)
    gas3, C3, Q3, ext3 = stage3(cfr_feed, mode, T3)
    pw3, water3 = W.remove_species(gas3, "H2O", WRF)
    rec = dict(ch4_rec=ch4_rec, purge=purge, to_s2=to_s2, gas1=gas1, C1=C1, Q1=Q1, feed2=feed2, gas2=gas2, Q2=Q2,
               water2=water2, pw2=pw2, cfr_feed=cfr_feed, permeate=permeate, gas3=gas3, C3=C3, Q3=Q3, ext3=ext3,
               water3=water3, pw3=pw3)
    return pw3, rec

def tear_error(x, g):
    tot = max(sum(g.values()), 1e-12); floor = 1e-6 * tot
    return max(abs(g.get(k, 0.0) - x.get(k, 0.0)) / max(x.get(k, 0.0), floor) for k in set(x) | set(g))

def solve_recycle(cfg, x0=None):
    """Successive substitution with predictive switch to bounded Wegstein. Returns (rec, info)."""
    if cfg["r"] == 0.0 and cfg["p"] == 1.0:
        g, rec = one_pass({}, cfg)
        rec["purge"] = rec["pw3"]          # once-through: the whole Stage 3 outlet (after water removal) leaves the system
        return rec, dict(iterations=1, method="once-through", converged=True, err=0.0)
    x = dict(x0) if x0 else {}
    method, it, err_hist = "direct", 0, []
    x_prev = g_prev = None
    budget = MAXIT
    while True:
        it += 1
        g, rec = one_pass(x, cfg)
        err = tear_error(x, g) if x else np.inf; err_hist.append(err)
        if err < TOL: return rec, dict(iterations=it, method=method, converged=True, err=err)
        if it >= budget:
            if method == "direct":
                method, budget = "wegstein", MAXIT + it
            else:
                return rec, dict(iterations=it, method=method, converged=False, err=err)
        if method == "direct" and it >= 20 and len(err_hist) >= 6:
            rho = (err_hist[-1] / err_hist[-6]) ** 0.2
            if 0 < rho < 1:
                need = np.log(TOL / err) / np.log(rho)
                if need > budget - it: method, budget = "wegstein", MAXIT + it
            elif rho >= 1: method, budget = "wegstein", MAXIT + it
        if method == "wegstein" and x_prev is not None and g_prev is not None:
            x_new = {}
            for k in set(g) | set(x):
                xk, xk1, gk, gk1 = x.get(k, 0.0), x_prev.get(k, 0.0), g.get(k, 0.0), g_prev.get(k, 0.0)
                if abs(xk - xk1) > 1e-14 * max(xk, 1.0):
                    s = (gk - gk1) / (xk - xk1); q = s / (s - 1.0) if abs(s - 1.0) > 1e-12 else 0.0
                    q = min(0.0, max(-10.0, q))
                else: q = 0.0
                x_new[k] = max(0.0, q * xk + (1.0 - q) * gk)
        else:
            x_new = dict(g)
        x_prev, g_prev, x = x, g, W.clean_species_dict(x_new)

# ---------------- carbon-origin tracer on converged flows
def tracer(rec, p, r_ch4):
    C_fresh_ch4 = FRESH_CH4; C_bio, C_dac = FRESH_CO2_BIO, FRESH_CO2_DAC
    C_rec_ch4 = rec["ch4_rec"]; C_to_s2 = carbon(rec["to_s2"])
    C_gas1 = carbon(rec["gas1"])
    f2 = np.array([1.0, 0.0, 0.0])
    for _ in range(10000):
        f1 = (C_fresh_ch4 * np.array([1.0, 0.0, 0.0]) + C_rec_ch4 * f2) / (C_fresh_ch4 + C_rec_ch4)
        num = C_gas1 * f1 + C_bio * np.array([0, 1.0, 0]) + C_dac * np.array([0, 0, 1.0]) + C_to_s2 * f2
        f2_new = num / num.sum()
        if np.max(np.abs(f2_new - f2)) < 1e-13: f2 = f2_new; break
        f2 = f2_new
    return f1, f2   # f1: Stage 1 outlet (solid C1 and gas1); f2: Stage 2/3 outlet, recycle, purge, C3

_CAT = {}
def c_atoms(sp):
    if sp == "C(s)": return 1.0
    if sp not in _CAT:
        g = W.GAS_REF; _CAT[sp] = g.n_atoms(g.species_index(sp), g.element_index("C")) if sp in g.species_names else 0.0
    return _CAT[sp]

def _mix_origins(parts):
    """parts: list of (n_dict, orig_dict). Returns per-species carbon-weighted origin vectors."""
    acc, wt = {}, {}
    for n, o in parts:
        for sp, v in n.items():
            a = c_atoms(sp)
            if a <= 0 or v <= 0 or sp not in o: continue
            acc[sp] = acc.get(sp, 0.0) + v * a * o[sp]; wt[sp] = wt.get(sp, 0.0) + v * a
    return {sp: acc[sp] / wt[sp] for sp in acc}

def _reactor_origins(n_in, o_in, n_out, solid_out):
    pool, pool_c = np.zeros(3), 0.0
    for sp in set(n_in) | set(n_out):
        a = c_atoms(sp)
        if a <= 0: continue
        ni, no = n_in.get(sp, 0.0), n_out.get(sp, 0.0)
        if ni > no and sp in o_in: pool += (ni - no) * a * o_in[sp]; pool_c += (ni - no) * a
    if pool_c > 0: f_pool = pool / pool_c
    else:
        f_pool = _mix_origins([(n_in, o_in)]); f_pool = (sum(f_pool[sp] * n_in[sp] * c_atoms(sp) for sp in f_pool) / max(carbon(n_in), 1e-30)) if f_pool else np.array([1.0, 0, 0])
    o_out = {}
    for sp, no in n_out.items():
        a = c_atoms(sp)
        if a <= 0 or no <= 0: continue
        ni = n_in.get(sp, 0.0)
        if no <= ni and sp in o_in: o_out[sp] = o_in[sp]
        else:
            passed = ni * a if sp in o_in else 0.0; produced = (no - ni) * a if sp in o_in else no * a
            o_out[sp] = (passed * o_in.get(sp, f_pool) + produced * f_pool) / (passed + produced)
    return o_out, f_pool

def tracer_netconv(rec, p, r_ch4):
    """Species-resolved net-conversion tracer. Returns dict with origin vectors of C1, C3, purge carbon."""
    e_ch4 = np.array([1.0, 0, 0]); f_co2 = np.array([0.0, FRESH_CO2_BIO, FRESH_CO2_DAC]) / (FRESH_CO2_BIO + FRESH_CO2_DAC)
    o_rec = {sp: e_ch4.copy() for sp in rec["pw3"]}
    for it in range(50000):
        o_s1in = _mix_origins([({"CH4": FRESH_CH4}, {"CH4": e_ch4}), ({"CH4": rec["ch4_rec"]}, {"CH4": o_rec.get("CH4", e_ch4)})])
        o_gas1, f_C1 = _reactor_origins({"CH4": FRESH_CH4 + rec["ch4_rec"]}, o_s1in, rec["gas1"], rec["C1"])
        o_feed2 = _mix_origins([(rec["gas1"], o_gas1), ({"CO2": FRESH_CO2_BIO + FRESH_CO2_DAC}, {"CO2": f_co2}), (rec["to_s2"], o_rec)])
        o_gas2, _ = _reactor_origins(rec["feed2"], o_feed2, rec["gas2"], 0.0)
        o_gas3, f_C3 = _reactor_origins(rec["cfr_feed"], o_gas2, rec["gas3"], rec["C3"])
        diff = max((np.max(np.abs(o_gas3[sp] - o_rec[sp])) for sp in o_gas3 if sp in o_rec), default=0.0)
        o_rec = o_gas3
        if diff < 1e-12 and it > 2: break
    purge = rec["purge"]; C_purge = carbon(purge)
    pvec = sum(purge[sp] * c_atoms(sp) * o_rec[sp] for sp in purge if sp in o_rec and c_atoms(sp) > 0) / C_purge if C_purge > 0 else np.zeros(3)
    return dict(f_C1=f_C1, f_C3=f_C3 if rec["C3"] > 0 else np.zeros(3), f_purge=pvec, iterations=it + 1)

# ---------------- assemble a result row
ORIG = ["wasteCH4", "biogasCO2", "DACCO2"]
def make_row(cfg, rec, info, ref=None):
    p, r_ch4, h, mode = cfg["p"], cfg["r"], cfg["h"], cfg["mode"]
    f1, f2 = tracer(rec, p, r_ch4)
    C1, C3 = rec["C1"], rec["C3"]
    solid_by_origin = C1 * f1 + C3 * f2                      # mixing tracer (upper bound)
    nc = tracer_netconv(rec, p, r_ch4)
    solid_nc = C1 * nc["f_C1"] + C3 * nc["f_C3"]             # net-conversion tracer (headline)
    purge = rec["purge"]; C_purge = carbon(purge)
    # element closure of the whole system
    ins = W.elemental_inventory_total({"CH4": FRESH_CH4, "CO2": FRESH_CO2_BIO + FRESH_CO2_DAC, "H2": SOLAR_H2}, 0.0)
    outs = W.elemental_inventory_total(add(rec["water2"], rec["water3"], rec["permeate"], purge), C1 + C3)
    closure = {e: (outs[e] - ins[e]) / ins[e] for e in W.ELEMENTS}
    feed2 = rec["feed2"]; gas2 = rec["gas2"]
    X_CO2_pass = 1 - gas2.get("CO2", 0) / feed2["CO2"]
    xi2 = gas2.get("CH4", 0) - feed2.get("CH4", 0)
    row = dict(tier=cfg["tier"], value_type="kinetic" if mode == "kinetic" else "equilibrium", mode=mode,
               T3_K=cfg["T3"], phi=cfg.get("phi", np.nan), k2_uncalibrated=cfg["k2"] if mode == "kinetic" else np.nan,
               purge_p=p, r_CH4=r_ch4, h_H2_to_CFR=h, converged=info["converged"], iterations=info["iterations"],
               method=info["method"], tear_rel_err=info["err"],
               # Stage 1
               S1_inlet_CH4_kmol_d=FRESH_CH4 + rec["ch4_rec"], S1_recycle_CH4_kmol_d=rec["ch4_rec"],
               S1_inlet_CH4_ratio_vs_fresh=(FRESH_CH4 + rec["ch4_rec"]) / FRESH_CH4,
               S1_X_CH4=1 - rec["gas1"].get("CH4", 0) / (FRESH_CH4 + rec["ch4_rec"]),
               # recycle to Stage 2
               recycle_to_S2_total_kmol_d=sum(rec["to_s2"].values()),
               recycle_to_S2_H2=rec["to_s2"].get("H2", 0), recycle_to_S2_CO=rec["to_s2"].get("CO", 0),
               recycle_to_S2_CO2=rec["to_s2"].get("CO2", 0), recycle_to_S2_CH4=rec["to_s2"].get("CH4", 0),
               recycle_to_S2_H2O=rec["to_s2"].get("H2O", 0),
               S2_inlet_total_kmol_d=sum(feed2.values()), S2_X_CO2_per_pass=X_CO2_pass, S2_xi2_CH4_kmol_d=xi2,
               S2_character="methanation (xi2>0)" if xi2 > 0 else "reforming (xi2<0)",
               S3_inlet_total_kmol_d=sum(rec["cfr_feed"].values()), S3_CH4_out_kmol_d=rec["gas3"].get("CH4", 0),
               S3_CO_out_kmol_d=rec["gas3"].get("CO", 0), S3_CO2_out_kmol_d=rec["gas3"].get("CO2", 0),
               S3_ext_CO2_meth=rec["ext3"].get("CO2_methanation", np.nan), S3_ext_CO_carb=rec["ext3"].get("CO_carbon", np.nan),
               S3_ext_CH4_crack=rec["ext3"].get("CH4_cracking", np.nan),
               # solid carbon (headline origin split = net-conversion tracer; mixing tracer kept as upper bound)
               C_stage1_tpd=tpd("C(s)", C1), C_stage3_tpd=tpd("C(s)", C3), C_total_tpd=tpd("C(s)", C1 + C3),
               C_from_wasteCH4_tpd=tpd("C(s)", solid_nc[0]), C_from_biogasCO2_tpd=tpd("C(s)", solid_nc[1]),
               C_from_DACCO2_tpd=tpd("C(s)", solid_nc[2]),
               C_from_CO2_total_tpd=tpd("C(s)", solid_nc[1] + solid_nc[2]),
               fresh_CO2_carbon_fixed_fraction=(solid_nc[1] + solid_nc[2]) / (FRESH_CO2_BIO + FRESH_CO2_DAC),
               fresh_CH4_carbon_fixed_fraction=solid_nc[0] / FRESH_CH4,
               C_from_CO2_total_tpd_mixing_tracer_upper=tpd("C(s)", solid_by_origin[1] + solid_by_origin[2]),
               C_from_biogasCO2_tpd_mixing=tpd("C(s)", solid_by_origin[1]), C_from_DACCO2_tpd_mixing=tpd("C(s)", solid_by_origin[2]),
               tracer_netconv_iterations=nc["iterations"],
               f1_wasteCH4=f1[0], f1_biogasCO2=f1[1], f1_DACCO2=f1[2], f2_wasteCH4=f2[0], f2_biogasCO2=f2[1], f2_DACCO2=f2[2],
               # purge
               purge_total_kmol_d=sum(purge.values()), purge_carbon_kmol_d=C_purge, purge_carbon_tpd=tpd("C(s)", C_purge),
               purge_CO2_tpd=tpd("CO2", purge.get("CO2", 0)), purge_CO_tpd=tpd("CO", purge.get("CO", 0)),
               purge_CH4_tpd=tpd("CH4", purge.get("CH4", 0)), purge_H2_tpd=tpd("H2", purge.get("H2", 0)),
               purge_carbon_as_CO2_if_oxidised_tpd=C_purge * W.MW["CO2"] / 1000.0,
               purge_CO2origin_carbon_tpd=tpd("C(s)", C_purge * (nc["f_purge"][1] + nc["f_purge"][2])),
               purge_CO2origin_carbon_tpd_mixing=tpd("C(s)", C_purge * (f2[1] + f2[2])),
               # hydrogen
               H2_permeate_tpd=tpd("H2", rec["permeate"].get("H2", 0)),
               H2_net_exportable_tpd=tpd("H2", rec["permeate"].get("H2", 0) + purge.get("H2", 0)),
               external_H2_required=(rec["permeate"].get("H2", 0) + purge.get("H2", 0)) < 0,
               # water
               water_stage2_tpd=tpd("H2O", rec["water2"].get("H2O", 0)), water_stage3_tpd=tpd("H2O", rec["water3"].get("H2O", 0)),
               water_total_tpd=tpd("H2O", rec["water2"].get("H2O", 0) + rec["water3"].get("H2O", 0)),
               # heat
               Q1_kW=rec["Q1"], Q2_kW=rec["Q2"], Q3_kW=rec["Q3"], Q_total_kW=rec["Q1"] + rec["Q2"] + rec["Q3"],
               closure_C_rel=closure["C"], closure_H_rel=closure["H"], closure_O_rel=closure["O"])
    if ref is not None:
        row.update(once_C_total_tpd=ref["C_total_tpd"], once_C_from_CO2_total_tpd=ref["C_from_CO2_total_tpd"],
                   delta_C_total_vs_once_tpd=row["C_total_tpd"] - ref["C_total_tpd"],
                   delta_C_from_CO2_vs_once_tpd=row["C_from_CO2_total_tpd"] - ref["C_from_CO2_total_tpd"],
                   delta_Q1_vs_once_kW=row["Q1_kW"] - ref["Q1_kW"], delta_water_vs_once_tpd=row["water_total_tpd"] - ref["water_total_tpd"],
                   delta_H2_export_vs_once_tpd=row["H2_net_exportable_tpd"] - ref["H2_net_exportable_tpd"],
                   S2_inlet_ratio_vs_once=row["S2_inlet_total_kmol_d"] / ref["S2_inlet_total_kmol_d"])
    return row

# ---------------- one worker group: fixed (mode, T3/phi, h); sweep r x p with warm start
def run_group(args):
    tier, mode, T3, phi, h, quick = args
    k2 = k2_from_phi(phi) if mode == "kinetic" else np.nan
    base = dict(tier=tier, mode=mode, T3=T3, phi=phi, k2=k2, h=h)
    ref_cfg = dict(base, p=1.0, r=0.0)
    rec, info = solve_recycle(ref_cfg); ref = make_row(ref_cfg, rec, info)
    rows = [ref]; recs = [dict(cfg=ref_cfg, rec=rec, info=info)]
    r_grid = [0.95] if quick else R_GRID
    p_grid = [0.1, 0.02] if quick else P_GRID
    for r_ch4 in r_grid:
        x0 = None
        for p in p_grid:
            cfg = dict(base, p=p, r=r_ch4); t0 = time.time()
            try:
                rec, info = solve_recycle(cfg, x0)
                row = make_row(cfg, rec, info, ref); row["wall_s"] = time.time() - t0
                x0 = rec["pw3"]; recs.append(dict(cfg=cfg, rec=rec, info=info))
            except Exception as e:
                row = dict(cfg, converged=False, error=str(e)[:200], wall_s=time.time() - t0)
            rows.append(row)
    return rows, recs

# ---------------- Tier 1: stoichiometric upper bound from the once-through design case
def tier1():
    cfg = dict(tier="T1", mode="kinetic", T3=T3_KIN, phi=0.0, k2=0.0, h=0.35, p=1.0, r=0.0)
    rec, _ = solve_recycle(cfg)
    g3 = rec["gas3"]; co, co2, ch4 = g3.get("CO", 0), g3.get("CO2", 0), g3.get("CH4", 0)
    C_out = carbon(g3); X1 = rec["C1"] / FRESH_CH4        # design-case X_CH4 = 0.873 (approach 0.90)
    h2_surplus = rec["permeate"].get("H2", 0) + g3.get("H2", 0)
    h2_meth = 3 * co + 4 * co2                              # CO + 3H2 -> CH4 + H2O ; CO2 + 4H2 -> CH4 + 2H2O
    solid_extra = C_out * X1; h2_back = 2 * solid_extra
    f2 = tracer(rec, 1.0, 0.0)[1]                # mixing tracer (Stage 3 outlet origin, upper bound)
    nc = tracer_netconv(rec, 1.0, 0.0)
    g3o = nc                                      # for the bound we need the Stage-3-outlet carbon origin: use purge (= pw3) vector
    f2 = nc["f_purge"]                            # net-conversion origin of the once-through Stage 3 outlet carbon
    rows = []
    rows.append(dict(case="once-through design case (kinetic, tau1*=8.905 s, phi=0, h=0.35, T3=650 K)",
                     C_solid_tpd=tpd("C(s)", rec["C1"] + rec["C3"]), C_from_CO2_tpd=tpd("C(s)", rec["C3"] * (nc["f_C3"][1] + nc["f_C3"][2])),
                     S3_out_CO_kmol_d=co, S3_out_CO2_kmol_d=co2, S3_out_CH4_kmol_d=ch4, S3_out_carbon_kmol_d=C_out,
                     H2_surplus_kmol_d=h2_surplus, H2_surplus_tpd=tpd("H2", h2_surplus), note="all values kinetic"))
    rows.append(dict(case="Tier-1a single-pass bound: all S3-outlet carbon -> CH4 -> Stage 1 at X_CH4 = 0.873",
                     C_solid_tpd=tpd("C(s)", rec["C1"] + solid_extra), C_solid_extra_tpd=tpd("C(s)", solid_extra),
                     C_from_CO2_tpd=tpd("C(s)", solid_extra * (f2[1] + f2[2])),
                     H2_required_for_methanation_kmol_d=h2_meth, H2_required_tpd=tpd("H2", h2_meth),
                     H2_returned_by_cracking_kmol_d=h2_back, H2_remaining_kmol_d=h2_surplus - h2_meth + h2_back,
                     H2_remaining_tpd=tpd("H2", h2_surplus - h2_meth + h2_back),
                     note="stoichiometric; S3-outlet carbon origin fractions from net-conversion tracer"))
    C_fresh = FRESH_CH4 + FRESH_CO2_BIO + FRESH_CO2_DAC
    h2_avail = 2 * FRESH_CH4 + SOLAR_H2; h2_need = 2 * (FRESH_CO2_BIO + FRESH_CO2_DAC)
    rows.append(dict(case="Tier-1b full-recycle asymptote (p->0, r->1): all fresh carbon -> solid; CO2 + 2H2 -> C + 2H2O net",
                     C_solid_tpd=tpd("C(s)", C_fresh), C_from_CO2_tpd=tpd("C(s)", FRESH_CO2_BIO + FRESH_CO2_DAC),
                     C_from_biogasCO2_tpd=tpd("C(s)", FRESH_CO2_BIO), C_from_DACCO2_tpd=tpd("C(s)", FRESH_CO2_DAC),
                     H2_available_kmol_d=h2_avail, H2_required_kmol_d=h2_need, H2_remaining_kmol_d=h2_avail - h2_need,
                     H2_remaining_tpd=tpd("H2", h2_avail - h2_need), water_tpd=tpd("H2O", 2 * (FRESH_CO2_BIO + FRESH_CO2_DAC)),
                     note="stoichiometric asymptote, no reactor model"))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    t0 = time.time()
    df1 = tier1(); df1.to_csv(OUT_T1, index=False); print(df1.to_string()); sys.stdout.flush()
    groups_eq = [("T2_eq", "equilibrium", T3, np.nan, h, a.quick) for T3 in T3_GRID_EQ for h in H_GRID]
    groups_kin = [("T3_kin", "kinetic", T3_KIN, phi, h, a.quick) for phi in PHI_GRID for h in H_GRID]
    if a.quick: groups_eq, groups_kin = groups_eq[2::4][:2], groups_kin[::4][:2]
    with Pool(min(12, cpu_count())) as pool:
        res_eq = pool.map(run_group, groups_eq); res_kin = pool.map(run_group, groups_kin)
    import pickle
    with open(os.path.join(RESULT, "P1_recycle_records.pkl"), "wb") as fh:
        pickle.dump({"eq": [rc for g in res_eq for rc in g[1]], "kin": [rc for g in res_kin for rc in g[1]]}, fh)
    res_eq = [g[0] for g in res_eq]; res_kin = [g[0] for g in res_kin]
    df_eq = pd.DataFrame([r for g in res_eq for r in g]); df_kin = pd.DataFrame([r for g in res_kin for r in g])
    df_eq.to_csv(OUT_EQ, index=False); df_kin.to_csv(OUT_KIN, index=False)
    print(f"\nTier 2 rows {len(df_eq)}, Tier 3 rows {len(df_kin)}, wall {time.time()-t0:.0f} s")
    print("unconverged eq:", int((~df_eq.converged.astype(bool)).sum()), " unconverged kin:", int((~df_kin.converged.astype(bool)).sum()))
    with open(OUT_TXT, "w") as fh:
        fh.write("P1 recycle analysis summary (see P1_recycle_summary.md)\n")
        fh.write(df1.to_string() + "\n\n")
        for name, df in [("Tier 2 EQUILIBRIUM", df_eq), ("Tier 3 KINETIC (uncalibrated)", df_kin)]:
            fh.write(f"===== {name} =====\n")
            cols = [c for c in ["T3_K", "phi", "h_H2_to_CFR", "r_CH4", "purge_p", "converged", "iterations", "method", "C_total_tpd", "C_from_CO2_total_tpd", "C_from_CO2_total_tpd_mixing_tracer_upper",
                                "fresh_CO2_carbon_fixed_fraction", "purge_carbon_tpd", "H2_net_exportable_tpd", "water_total_tpd", "Q1_kW", "Q2_kW", "Q3_kW",
                                "S1_inlet_CH4_ratio_vs_fresh", "S2_inlet_ratio_vs_once", "closure_C_rel"] if c in df.columns]
            with pd.option_context("display.width", 300, "display.max_columns", 50, "display.max_rows", 500, "display.float_format", lambda v: f"{v:.4g}"):
                fh.write(df[cols].to_string(index=False) + "\n\n")
    print("Saved:", OUT_T1, OUT_EQ, OUT_KIN, OUT_TXT)

if __name__ == "__main__":
    main()
