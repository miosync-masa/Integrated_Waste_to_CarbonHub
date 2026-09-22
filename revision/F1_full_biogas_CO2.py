# -*- coding: utf-8 -*-
"""
F1: Configuration change — treat ALL biogas CO2, no DAC (reviewers R1-1, R1-7; decision X3)
=========================================================================================
FUPROC-D-26-00269 Major Revision. Builds on P1 (recycle flowsheet, net-conversion tracer),
KIN_chain (tau1* = 8.905 s), T3/T3b (per-stage pressure, realistic pressure order n = 1).

Motivation (E9): with 48 t/d CH4 read as 60 vol% biogas, the accompanying CO2 is ~88 t/d; the
submitted flowsheet routed only 32 t/d of it to Stage 2 and added 10 t/d of DAC CO2. Here the
DAC is removed and the whole biogas CO2 is treated. Net recycle reaction CO2 + 2H2 -> C + 2H2O.

Feeds (kmol/d)
  CH4 fixed at 48 t/d = 2,992 kmol/d. Biogas CO2 = CH4 x (1-y)/y for CH4 volume fraction
  y in {0.55, 0.60, 0.65} -> 107.7 / 87.8 / 70.9 t/d CO2. DAC CO2 = 0.
  Electrolytic H2: 2.52 t/d (1,250 kmol/d, as submitted) or 0.
  Configurations: "submitted" (biogas CO2 32 + DAC 10 t/d, H2 2.52 t/d),
                  "2a" (all biogas CO2, no DAC, H2 2.52), "2b" (all biogas CO2, no DAC, no H2).

Cases (P1 topology: Stage 3 outlet -> water 0.95 -> CH4 separation r_CH4 = 0.95 -> Stage 1;
rest -> purge p = 0.05 -> Stage 2; Stage 2/3 separation water 0.95, H2 split h to CFR)
  Equilibrium (all stages Gibbs, 1 atm): once-through and recycle, h in {0.35, 0.5, 0.75}.  EQUILIBRIUM.
  Kinetic design case: Stage 1 tau1* = 8.905 s, Stage 2 two-reaction model phi in {0, 0.1, 1}
  (tau2 = 3 s), Stage 3 submitted 3-reaction model (650 K, tau3 = 3 s); pressures
  (1 atm, 1 atm, 1 atm) with native rate laws and (1 atm, 5 bar, 5 bar) with the methanation
  pressure order forced to n = 1 (T3b variant V1); once-through and recycle at h = 0.35.  KINETIC,
  all rate constants UNCALIBRATED. The "1 bar" case is run at 1 atm to stay identical to P1/KIN_chain.

Carbon origin: species-resolved net-conversion tracer of P1 (origins: waste CH4 / biogas CO2 /
DAC CO2). H2 limitation: the model cannot import H2, so a shortage shows up as unfixed CO2 leaving
in the purge with H2 export near zero; the stoichiometric H2 balance
(2 H2 per fixed CO2 vs 2 H2 per converted CH4 + electrolysis) is reported alongside.

Reproducibility:  cd <repo>/revision ; ../.venv/bin/python F1_full_biogas_CO2.py   (~10-15 min, 12 processes)
Imports Workflow_cantera.py, P1_recycle_analysis.py, T3_pressure_analysis.py, T3b_pressure_order_sensitivity.py.
Outputs: F1_cases.csv, F1_summary.txt
"""
import os, sys, time
import numpy as np, pandas as pd
import cantera as ct
from multiprocessing import Pool, cpu_count

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import P1_recycle_analysis as P1m
import T3_pressure_analysis as T3m
import T3b_pressure_order_sensitivity as T3b
from P1_recycle_analysis import FRESH_CH4, TAU1_STAR, TAU2, TAU3, WRF, tpd, carbon, add, scale, k2_from_phi, tear_error, _mix_origins, _reactor_origins, c_atoms

OUT_CSV = os.path.join(RESULT, "F1_cases.csv"); OUT_TXT = os.path.join(RESULT, "F1_summary.txt")
ATM = ct.one_atm; BAR = 1.0e5
T2 = W.T_rwgs; T3K = W.T_cfr
SOLAR_H2_SUBMITTED = W.solar_h2_kmol_per_day(W.solar_power_kW, W.electrolyzer_eff)   # 1,250 kmol/d
CO2_BIO_SUBMITTED = W.tpd_to_kmol_per_day(W.CO2_from_biogas_tpd, W.MW["CO2"])          # 727
CO2_DAC_SUBMITTED = W.tpd_to_kmol_per_day(W.CO2_from_DAC_tpd, W.MW["CO2"])             # 227
REC = dict(p=0.05, r=0.95)
TOL, MAXIT = 1e-8, 500

def feeds():
    F = {"submitted": dict(y=np.nan, co2_bio=CO2_BIO_SUBMITTED, co2_dac=CO2_DAC_SUBMITTED, h2=SOLAR_H2_SUBMITTED)}
    for y in [0.55, 0.60, 0.65]:
        co2 = FRESH_CH4 * (1 - y) / y
        F[f"2a_y{y:.2f}"] = dict(y=y, co2_bio=co2, co2_dac=0.0, h2=SOLAR_H2_SUBMITTED)
        F[f"2b_y{y:.2f}"] = dict(y=y, co2_bio=co2, co2_dac=0.0, h2=0.0)
    return F

# ---------------- robust Stage 3 Gibbs (gas + graphite) for H2-lean, CO-rich recycle compositions
def stage3_eq_robust(feed, P, T=T3K):
    """Multiphase equilibrium with fallbacks: (1) baseline settings; (2)-(3) alternative initial
    estimates; (4) initial moles from the gas-only equilibrium plus a carbon seed; (5)-(6) same at
    T +/- 0.25 K (negligible thermodynamic effect, flagged in the returned note)."""
    g = ct.Solution("gri30.yaml"); sld = ct.Solution("graphite.yaml")
    x, n = W.normalize_mole_dict(feed); g.TPX = T, P, x
    tries = [dict(solver="gibbs", rtol=1e-6, max_steps=200000, max_iter=100, estimate_equil=-1, log_level=0),
             dict(solver="gibbs", rtol=1e-6, max_steps=200000, max_iter=200, estimate_equil=0, log_level=0),
             dict(solver="gibbs", rtol=1e-6, max_steps=200000, max_iter=200, estimate_equil=1, log_level=0),
             dict(solver="gibbs", rtol=1e-5, max_steps=400000, max_iter=300, estimate_equil=-1, log_level=0)]
    ism = None
    try:
        ge = W.equilibrate_gas(feed, T, P)
        if ge["converged"]:
            v = np.zeros(g.n_species + 1)
            for sp, val in ge["gas_kmol_d"].items(): v[g.species_index(sp)] = val
            # element-conserving carbon seed (5 % of inlet carbon): 2 CO -> C(s) + CO2, then CH4 -> C(s) + 2 H2
            seed = 0.05 * carbon(feed); iCO, iCO2, iCH4, iH2 = (g.species_index(k) for k in ("CO", "CO2", "CH4", "H2"))
            s1 = min(seed, 0.5 * v[iCO]); v[iCO] -= 2 * s1; v[iCO2] += s1; v[-1] += s1
            s2 = min(seed - s1, v[iCH4]); v[iCH4] -= s2; v[iH2] += 2 * s2; v[-1] += s2
            ism = v
    except Exception: pass
    attempts = [(T, None, kw) for kw in tries] + [(T, ism, kw) for kw in tries[:2]] + [(T + 0.25, ism, tries[0]), (T - 0.25, ism, tries[0])]
    last = None
    for Tt, ism_t, kw in attempts:
        if ism_t is None and Tt != T: continue
        r = W._equilibrate_mixture([(g, n), (sld, 0)], Tt, P, g.species_names, sld.species_names[0], ism_t, [kw])
        if r["converged"]:
            fH = W.total_stream_enthalpy_J_per_day(feed, 0, T, P); pH = W.total_stream_enthalpy_J_per_day(r["gas_kmol_d"], r["Csolid_kmol_d"], T, P)
            # guard: elemental balance of the multiphase result against the feed
            cin = carbon(feed); cout = carbon(r["gas_kmol_d"]) + r["Csolid_kmol_d"]
            if abs(cout - cin) > 1e-6 * max(cin, 1e-30):
                last = f"carbon imbalance {cout-cin:.3e}"; continue
            return r["gas_kmol_d"], r["Csolid_kmol_d"], (pH - fH) / 86400e3
        last = r["error"]
    raise RuntimeError("Stage 3 Gibbs failed after fallbacks: " + str(last)[:200])

# ---------------- flowsheet with configurable fresh feed and per-stage pressure / rate-order factors
def one_pass(x, cfg):
    fd = cfg["feed"]; p, r_ch4, h, mode, k2 = cfg["p"], cfg["r"], cfg["h"], cfg["mode"], cfg["k2"]
    Pa1, Pa2, Pa3 = cfg["P1_Pa"], cfg["P2_Pa"], cfg["P3_Pa"]
    ch4_rec = r_ch4 * x.get("CH4", 0.0)
    rest = dict(x); rest["CH4"] = x.get("CH4", 0.0) - ch4_rec; rest = W.clean_species_dict(rest)
    purge, to_s2 = scale(rest, p), scale(rest, 1.0 - p)
    gas1, C1, Q1 = T3m.stage1(FRESH_CH4 + ch4_rec, mode, Pa1)
    fresh2 = {"CO2": fd["co2_bio"] + fd["co2_dac"]}
    if fd["h2"] > 0: fresh2["H2"] = fd["h2"]
    feed2 = add(gas1, fresh2, to_s2)
    if mode == "kinetic":
        f2 = T3b.order_factors(cfg["n"], cfg["variant"], Pa2); f3 = T3b.order_factors(cfg["n"], cfg["variant"], Pa3)
        gas2, Q2 = T3b.stage2_kinetic_n(feed2, k2, Pa2, f2)
    else:
        gas2, Q2 = T3m.stage2(feed2, "equilibrium", 0.0, Pa2)
    pw2, water2 = W.remove_species(gas2, "H2O", WRF)
    cfr_feed, permeate = W.membrane_split(pw2, {"H2": h, "CO2": 1.0, "CO": 1.0, "CH4": 1.0}, 1.0)
    if mode == "kinetic": gas3, C3, Q3 = T3b.stage3_kinetic_n(cfr_feed, Pa3, f3)
    else: gas3, C3, Q3 = stage3_eq_robust(cfr_feed, Pa3)
    pw3, water3 = W.remove_species(gas3, "H2O", WRF)
    return pw3, dict(ch4_rec=ch4_rec, purge=purge, to_s2=to_s2, gas1=gas1, C1=C1, Q1=Q1, feed2=feed2, gas2=gas2, Q2=Q2,
                     water2=water2, pw2=pw2, cfr_feed=cfr_feed, permeate=permeate, gas3=gas3, C3=C3, Q3=Q3, water3=water3, pw3=pw3)

def solve_recycle(cfg, x0=None):
    if cfg["r"] == 0.0 and cfg["p"] == 1.0:
        g, rec = one_pass({}, cfg); rec["purge"] = rec["pw3"]
        return rec, dict(iterations=1, method="once-through", converged=True, err=0.0)
    x = dict(x0) if x0 else {}
    method, it, err_hist, budget = "direct", 0, [], MAXIT; x_prev = g_prev = None
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

# ---------------- net-conversion tracer with configurable fresh CO2 split
def tracer_netconv(rec, fd):
    e_ch4 = np.array([1.0, 0, 0]); tot = fd["co2_bio"] + fd["co2_dac"]
    f_co2 = np.array([0.0, fd["co2_bio"], fd["co2_dac"]]) / tot
    o_rec = {sp: e_ch4.copy() for sp in rec["pw3"]}
    for it in range(50000):
        o_s1in = _mix_origins([({"CH4": FRESH_CH4}, {"CH4": e_ch4}), ({"CH4": rec["ch4_rec"]}, {"CH4": o_rec.get("CH4", e_ch4)})])
        o_gas1, f_C1 = _reactor_origins({"CH4": FRESH_CH4 + rec["ch4_rec"]}, o_s1in, rec["gas1"], rec["C1"])
        o_feed2 = _mix_origins([(rec["gas1"], o_gas1), ({"CO2": tot}, {"CO2": f_co2}), (rec["to_s2"], o_rec)])
        o_gas2, _ = _reactor_origins(rec["feed2"], o_feed2, rec["gas2"], 0.0)
        o_gas3, f_C3 = _reactor_origins(rec["cfr_feed"], o_gas2, rec["gas3"], rec["C3"])
        diff = max((np.max(np.abs(o_gas3[sp] - o_rec[sp])) for sp in o_gas3 if sp in o_rec), default=0.0)
        o_rec = o_gas3
        if diff < 1e-12 and it > 2: break
    purge = rec["purge"]; Cp = carbon(purge)
    pvec = sum(purge[sp] * c_atoms(sp) * o_rec[sp] for sp in purge if sp in o_rec and c_atoms(sp) > 0) / Cp if Cp > 0 else np.zeros(3)
    return f_C1, (f_C3 if rec["C3"] > 0 else np.zeros(3)), pvec, it + 1

def make_row(cfg, rec, info, label):
    fd = cfg["feed"]; C1, C3 = rec["C1"], rec["C3"]
    f1, f3, fp, tit = tracer_netconv(rec, fd); solid = C1 * f1 + C3 * f3
    purge = rec["purge"]; Cp = carbon(purge); co2_in = fd["co2_bio"] + fd["co2_dac"]
    fresh = {"CH4": FRESH_CH4, "CO2": co2_in}
    if fd["h2"] > 0: fresh["H2"] = fd["h2"]
    ins = W.elemental_inventory_total(fresh, 0.0)
    outs = W.elemental_inventory_total(add(rec["water2"], rec["water3"], rec["permeate"], purge), C1 + C3)
    closure = {e: (outs[e] - ins[e]) / ins[e] for e in W.ELEMENTS}
    feed2, gas2 = rec["feed2"], rec["gas2"]
    h2_out = rec["permeate"].get("H2", 0) + purge.get("H2", 0)
    X1 = 1 - rec["gas1"].get("CH4", 0) / (FRESH_CH4 + rec["ch4_rec"])
    # stoichiometric H2 budget (net: CO2 + 2H2 -> C + 2H2O ; CH4 -> C + 2H2)
    h2_stoich_avail = 2 * FRESH_CH4 * X1 + fd["h2"]; h2_stoich_need_all_co2 = 2 * co2_in
    return dict(case=label, config=cfg["config"], y_CH4_biogas=fd["y"], CO2_bio_tpd=tpd("CO2", fd["co2_bio"]), CO2_DAC_tpd=tpd("CO2", fd["co2_dac"]),
                CO2_total_tpd=tpd("CO2", co2_in), CO2_total_carbon_tpd=tpd("C(s)", co2_in), H2_electrolysis_tpd=tpd("H2", fd["h2"]),
                value_type="kinetic" if cfg["mode"] == "kinetic" else "equilibrium", mode=cfg["mode"], phi=cfg.get("phi", np.nan),
                P1_bar=cfg["P1_Pa"] / BAR, P2_bar=cfg["P2_Pa"] / BAR, P3_bar=cfg["P3_Pa"] / BAR, pressure_order=("native" if cfg.get("n") is None else cfg["n"]),
                recycle=(cfg["p"] < 1), purge_p=cfg["p"], r_CH4=cfg["r"], h_H2_to_CFR=cfg["h"],
                converged=info["converged"], iterations=info["iterations"], method=info["method"],
                S1_X_CH4=X1, S1_inlet_CH4_ratio_vs_fresh=(FRESH_CH4 + rec["ch4_rec"]) / FRESH_CH4,
                S2_inlet_H2_to_CO2_ratio=feed2.get("H2", 0) / feed2["CO2"], S2_inlet_total_kmol_d=sum(feed2.values()),
                S2_X_CO2_per_pass=1 - gas2.get("CO2", 0) / feed2["CO2"], S2_xi1_RWGS_CO_formed_kmol_d=gas2.get("CO", 0) - feed2.get("CO", 0),
                S2_xi2_CH4_formed_kmol_d=gas2.get("CH4", 0) - feed2.get("CH4", 0),
                S2_methanation_to_RWGS_ratio=(gas2.get("CH4", 0) - feed2.get("CH4", 0)) / max(gas2.get("CO", 0) - feed2.get("CO", 0), 1e-9),
                S3_CH4_formed_kmol_d=rec["gas3"].get("CH4", 0) - rec["cfr_feed"].get("CH4", 0),
                recycle_to_S2_total_kmol_d=sum(rec["to_s2"].values()),
                C_stage1_tpd=tpd("C(s)", C1), C_stage3_tpd=tpd("C(s)", C3), C_total_tpd=tpd("C(s)", C1 + C3),
                C_from_wasteCH4_tpd=tpd("C(s)", solid[0]), C_from_biogasCO2_tpd=tpd("C(s)", solid[1]), C_from_DACCO2_tpd=tpd("C(s)", solid[2]),
                C_from_CO2_total_tpd=tpd("C(s)", solid[1] + solid[2]), CO2_carbon_fixed_fraction=(solid[1] + solid[2]) / co2_in,
                CO2_fixed_as_CO2_tpd=(solid[1] + solid[2]) * W.MW["CO2"] / 1000.0,
                purge_CO2_tpd=tpd("CO2", purge.get("CO2", 0)), purge_CO_tpd=tpd("CO", purge.get("CO", 0)), purge_CH4_tpd=tpd("CH4", purge.get("CH4", 0)),
                purge_carbon_as_CO2_if_oxidised_tpd=Cp * W.MW["CO2"] / 1000.0,
                H2_net_exportable_tpd=tpd("H2", h2_out), H2_permeate_tpd=tpd("H2", rec["permeate"].get("H2", 0)),
                H2_stoich_available_tpd=tpd("H2", h2_stoich_avail), H2_stoich_needed_for_all_CO2_tpd=tpd("H2", h2_stoich_need_all_co2),
                H2_stoich_surplus_tpd=tpd("H2", h2_stoich_avail - h2_stoich_need_all_co2),
                water_total_tpd=tpd("H2O", rec["water2"].get("H2O", 0) + rec["water3"].get("H2O", 0)),
                Q1_kW=rec["Q1"], Q2_kW=rec["Q2"], Q3_kW=rec["Q3"], Q_total_kW=rec["Q1"] + rec["Q2"] + rec["Q3"],
                closure_C_rel=closure["C"], closure_H_rel=closure["H"], closure_O_rel=closure["O"], tracer_iterations=tit)

def run_group(args):
    config, fd, mode, phi, pres, n, k2 = args
    rows = []; base = dict(config=config, feed=fd, mode=mode, phi=phi, k2=k2, n=n, variant="V1",
                           P1_Pa=pres[0], P2_Pa=pres[1], P3_Pa=pres[2])
    plab = f"({pres[0]/BAR:.3g},{pres[1]/BAR:.3g},{pres[2]/BAR:.3g}) bar" + ("" if n is None else f" n={n:g}")
    def run_one(cfg, label, x0=None):
        t0 = time.time()
        try:
            rec, info = solve_recycle(cfg, x0); r = make_row(cfg, rec, info, label); r["wall_s"] = time.time() - t0; return r, rec
        except Exception as ex:
            return dict(case=label, config=config, mode=mode, phi=phi, converged=False, error=str(ex)[:300], wall_s=time.time() - t0,
                        P1_bar=cfg["P1_Pa"] / BAR, P2_bar=cfg["P2_Pa"] / BAR, P3_bar=cfg["P3_Pa"] / BAR, h_H2_to_CFR=cfg["h"], purge_p=cfg["p"], r_CH4=cfg["r"],
                        recycle=(cfg["p"] < 1), y_CH4_biogas=fd["y"], H2_electrolysis_tpd=tpd("H2", fd["h2"]), CO2_total_tpd=tpd("CO2", fd["co2_bio"] + fd["co2_dac"]),
                        value_type="kinetic" if mode == "kinetic" else "equilibrium", pressure_order=("native" if n is None else n)), None
    if mode == "equilibrium":
        for h in [0.35, 0.5, 0.75]:
            r, _ = run_one(dict(base, p=1.0, r=0.0, h=h), f"{config} eq once-through h={h:g}"); rows.append(r)
        x0 = None
        for h in [0.35, 0.5, 0.75]:
            r, rec = run_one(dict(base, h=h, **REC), f"{config} eq recycle h={h:g}", x0); rows.append(r)
            if rec is not None: x0 = rec["pw3"]
    else:
        r, _ = run_one(dict(base, p=1.0, r=0.0, h=0.35), f"{config} kin once-through phi={phi:g} {plab}"); rows.append(r)
        r, _ = run_one(dict(base, h=0.35, **REC), f"{config} kin recycle phi={phi:g} {plab}"); rows.append(r)
    return rows

def main():
    t0 = time.time(); F = feeds(); k2_map = {phi: k2_from_phi(phi) for phi in [0.0, 0.1, 1.0]}
    groups = []
    for config, fd in F.items():
        groups.append((config, fd, "equilibrium", np.nan, (ATM, ATM, ATM), None, 0.0))
        for phi in [0.0, 0.1, 1.0]:
            groups.append((config, fd, "kinetic", phi, (ATM, ATM, ATM), None, k2_map[phi]))
            groups.append((config, fd, "kinetic", phi, (ATM, 5 * BAR, 5 * BAR), 1.0, k2_map[phi]))
    with Pool(min(12, cpu_count())) as pool: res = pool.map(run_group, groups)
    df = pd.DataFrame([r for g in res for r in g]); df.to_csv(OUT_CSV, index=False)
    L = [f"F1 full-biogas-CO2 configurations (wall {time.time()-t0:.0f} s). Unconverged: {int((~df.converged.astype(bool)).sum())}; max iter {df.iterations.max()}; max |closure C| {df.closure_C_rel.abs().max():.1e}"]
    cols = ["case", "CO2_total_tpd", "H2_electrolysis_tpd", "iterations", "S1_X_CH4", "S1_inlet_CH4_ratio_vs_fresh", "S2_inlet_H2_to_CO2_ratio", "S2_X_CO2_per_pass",
            "S2_xi1_RWGS_CO_formed_kmol_d", "S2_xi2_CH4_formed_kmol_d", "S3_CH4_formed_kmol_d", "C_total_tpd", "C_from_biogasCO2_tpd", "C_from_DACCO2_tpd",
            "CO2_carbon_fixed_fraction", "CO2_fixed_as_CO2_tpd", "purge_carbon_as_CO2_if_oxidised_tpd", "H2_net_exportable_tpd", "H2_stoich_surplus_tpd",
            "water_total_tpd", "Q1_kW", "Q2_kW", "Q3_kW", "recycle_to_S2_total_kmol_d", "closure_C_rel"]
    with pd.option_context("display.width", 350, "display.max_columns", 60, "display.max_rows", 300, "display.float_format", lambda v: f"{v:.4g}"):
        L.append("\n=== EQUILIBRIUM ===\n" + df[df["mode"] == "equilibrium"][cols].to_string(index=False))
        L.append("\n=== KINETIC (uncalibrated) ===\n" + df[df["mode"] == "kinetic"][cols].to_string(index=False))
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", OUT_CSV, OUT_TXT)

if __name__ == "__main__":
    main()
