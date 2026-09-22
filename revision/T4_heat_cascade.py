# -*- coding: utf-8 -*-
"""
T4: Heat cascade / pinch analysis of the X4 configuration and of the submitted chain (R2-2)
==========================================================================================
FUPROC-D-26-00269 Major Revision. Builds on F1 (X4 flowsheet, solved here again to obtain every
stream), F1b/P3 (interim heat-recovery fractions), P3 (auxiliary power).

Step 0 - what Q1, Q2, Q3 are: in Workflow_cantera.py every stage duty is
    Q_i = H(products at T_i) - H(feed at T_i)
i.e. the ISOTHERMAL REACTION DUTY at the stage temperature. Feed preheating and product cooling
are not included anywhere in the submitted balance. Here they are added as separate streams, so
nothing is double counted: reaction duties enter as isothermal streams at T_i, sensible/latent
duties as temperature-range streams.

Stream inventory (X4 case, recycle equilibrium; temperatures in K)
  Hot (to be cooled):  Stage 1 gas 1200 -> 950 (enters Stage 2); Stage 1 solid carbon 1200 -> 313;
                       Stage 2 gas 950 -> 313 with water condensation; Stage 2 exotherm at 950 (if Q2 < 0);
                       Stage 3 exotherm at 650; Stage 3 gas 650 -> 313 with condensation; Stage 3 solid 650 -> 313.
  Cold (to be heated): fresh CH4 298 -> 1200; recycled CH4 313 -> 1200; Stage 1 endotherm at 1200;
                       fresh CO2 298 -> 950 (and fresh H2 in the submitted case); recycle-to-Stage-2 313 -> 950;
                       Stage 2 endotherm at 950 (if Q2 > 0); CFR feed 313 -> 650.
  Fresh feeds at 298 K; all streams leaving condensers/separators at 313 K. Ideal-gas enthalpies from GRI-3.0,
  graphite from graphite.yaml; water condensation below the dew point with Antoine vapour pressure and
  h_fg(T) = 2501 - 2.36 (T - 273.15) kJ/kg. Isothermal duties are represented over a 0.5 K span.

Pinch analysis: problem-table algorithm on piecewise-linear H(T) curves, dTmin = 20 and 30 K
(hot shifted by -dTmin/2, cold by +dTmin/2); composite curves, grand composite curve, minimum hot utility
Q_H,min (to be fired with H2, furnace efficiency 0.85 on LHV), minimum cold utility, pinch temperature,
process-to-process heat recovery, and a counter-current area target with U = 50 W/m2/K (gas-gas
assumption; condensing sections would be better).

Self-sufficiency update: H2 fired = Q_H,min/0.85/LHV; auxiliaries from P3 (P3_selfsufficiency_map.csv:
981 kW excl. biogas upgrading for the representative case; incl. upgrading also reported) converted at
eta_e = 0.50 (fuel cell) / 0.40 (H2 engine); final net H2 = exportable (F1) - fired - auxiliaries.
Equivalent recovery fraction in the F1b/P3 convention: f_eq = (Q1 + Q2+ - Q_H,min) / pool with
pool = -(min(Q2,0) + min(Q3,0)).

Extra checks: (i) raw biogas (CH4 + CO2) fed directly to Stage 1 at 1200 K (Gibbs, graphite) - dry-reforming
penalty vs the separation power of P3; (ii) the submitted once-through chain (Feed A) - how much of the
"55 % heat offset" (Q2 + Q3 = 1,675 kW vs Q1 = 3,031 kW) is usable once temperature levels are respected.

All values EQUILIBRIUM (Gibbs) unless stated. Reproducibility: cd <repo>/revision ;
../.venv/bin/python T4_heat_cascade.py  (~4 min, 11 processes). Baseline outputs untouched.
Outputs: T4_streams_X4_rep.csv, T4_streams_submitted_once.csv, T4_pinch_results.csv, T4_selfsufficiency.csv,
         T4_raw_biogas_stage1.csv, T4_composite_curves.png, T4_summary.txt
"""
import os, sys, time
import numpy as np, pandas as pd
import cantera as ct
from multiprocessing import Pool, cpu_count
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import F1_full_biogas_CO2 as F1m
from P1_recycle_analysis import FRESH_CH4, tpd, add

ATM = ct.one_atm; T_AMB = 298.15; T_COND = 313.15; DTMIN_LIST = [20.0, 30.0]
ETA_FURNACE = 0.85; LHV_KWH_KG = 33.32; ETA_E = {"fuel cell 0.50": 0.50, "H2 engine 0.40": 0.40}; U_TARGET = 50.0  # W/m2/K
OUT = {k: os.path.join(RESULT, v) for k, v in dict(streams_rep="T4_streams_X4_rep.csv", streams_sub="T4_streams_submitted_once.csv", pinch="T4_pinch_results.csv",
       self="T4_selfsufficiency.csv", raw="T4_raw_biogas_stage1.csv", png="T4_composite_curves.png", txt="T4_summary.txt").items()}
P3_MAP = os.path.join(RESULT, "P3_selfsufficiency_map.csv")

_GAS = None; _SOL = None
def gas():
    global _GAS, _SOL
    if _GAS is None: _GAS = ct.Solution("gri30.yaml"); _SOL = ct.Solution("graphite.yaml")
    return _GAS, _SOL
KMOLD_TO_MOLS = 1000.0 / 86400.0

def h_gas_kW(comp, T):
    g, _ = gas(); comp = W.clean_species_dict(comp); n = sum(comp.values())
    if n <= 0: return 0.0
    x, _ = W.normalize_mole_dict(comp); g.TPX = T, ATM, x
    return g.enthalpy_mole / 1000.0 * n * KMOLD_TO_MOLS / 1000.0     # J/kmol -> J/mol ; mol/s -> W -> kW
def h_solid_kW(n_c, T):
    _, s = gas(); s.TP = T, ATM; return s.enthalpy_mole / 1000.0 * n_c * KMOLD_TO_MOLS / 1000.0
def psat_water_atm(T):
    Tc = T - 273.15; return 10 ** (8.07131 - 1730.63 / (Tc + 233.426)) / 760.0
def hfg_kJ_mol(T): return (2501.0 - 2.36 * (T - 273.15)) * W.MW["H2O"] / 1000.0
def h_stream_kW(comp, T, condensing=True):
    """Enthalpy flow of a gas stream at T, with water condensation below the dew point (liquid at T)."""
    comp = W.clean_species_dict(comp); nw = comp.get("H2O", 0.0)
    if not condensing or nw <= 0: return h_gas_kW(comp, T)
    ndry = sum(v for k, v in comp.items() if k != "H2O"); ps = psat_water_atm(T)
    nvap_max = ndry * ps / max(1.0 - ps, 1e-9) if ps < 1 else nw
    nvap = min(nw, nvap_max); nliq = nw - nvap
    c = dict(comp); c["H2O"] = nvap
    if nvap <= 0: c.pop("H2O")
    H = h_gas_kW(c, T)
    if nliq > 0:
        g, _ = gas(); g.TPX = T, ATM, "H2O:1"; h_vap = g.enthalpy_mole / 1000.0 / 1000.0   # kJ/mol
        H += (h_vap - hfg_kJ_mol(T)) * nliq * KMOLD_TO_MOLS
    return H
def dewpoint(comp):
    comp = W.clean_species_dict(comp); nw = comp.get("H2O", 0.0); n = sum(comp.values())
    if nw <= 0: return None
    yw = nw / n; lo, hi = 273.16, 373.0
    if psat_water_atm(hi) < yw: return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if psat_water_atm(mid) > yw: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

def curve(kind, name, T_from, T_to, comp=None, solid=0.0, duty_kW=None):
    """Return dict with monotonic T array (ascending) and cumulative duty H(T) [kW] measured from the low end."""
    if duty_kW is not None:   # isothermal
        Tlo, Thi = min(T_from, T_to), max(T_from, T_to)
        return dict(kind=kind, name=name, T=np.array([Tlo, Thi]), H=np.array([0.0, abs(duty_kW)]), T_in=T_from, T_out=T_to, duty=abs(duty_kW))
    Tlo, Thi = min(T_from, T_to), max(T_from, T_to)
    pts = list(np.arange(Tlo, Thi, 20.0)) + [Thi]
    td = dewpoint(comp) if comp else None
    if td and Tlo < td < Thi: pts += list(np.arange(Tlo, min(td + 10, Thi), 2.0)) + [td]
    Ts = np.unique(np.round(np.array(pts), 4)); Ts = Ts[(Ts >= Tlo) & (Ts <= Thi)]
    Hs = np.array([(h_stream_kW(comp, T) if comp else 0.0) + (h_solid_kW(solid, T) if solid > 0 else 0.0) for T in Ts])
    Hs = Hs - Hs[0]
    return dict(kind=kind, name=name, T=Ts, H=Hs, T_in=T_from, T_out=T_to, duty=float(Hs[-1]))

def H_at(c, T):
    """Cumulative duty of stream c between its low end and temperature T (clipped)."""
    return float(np.interp(np.clip(T, c["T"][0], c["T"][-1]), c["T"], c["H"]))

def pinch(streams, dtmin):
    hot = [c for c in streams if c["kind"] == "hot"]; cold = [c for c in streams if c["kind"] == "cold"]
    sh = [dict(c, T=c["T"] - dtmin / 2) for c in hot]; sc = [dict(c, T=c["T"] + dtmin / 2) for c in cold]
    bounds = np.unique(np.round(np.concatenate([c["T"] for c in sh + sc]), 6))[::-1]   # descending
    R = [0.0]; nets = []
    for k in range(len(bounds) - 1):
        Th, Tl = bounds[k], bounds[k + 1]
        net = sum(H_at(c, Th) - H_at(c, Tl) for c in sh) - sum(H_at(c, Th) - H_at(c, Tl) for c in sc)
        nets.append(net); R.append(R[-1] + net)
    R = np.array(R); QH = max(0.0, -R.min()); casc = R + QH
    ip = int(np.argmin(np.abs(casc[1:] + 0)) + 1) if QH > 0 else None
    pinch_T_shift = bounds[int(np.argmin(casc))] if QH > 0 else np.nan
    QC = casc[-1]
    hot_total = sum(c["duty"] for c in hot); cold_total = sum(c["duty"] for c in cold)
    recovered = cold_total - QH
    # composite curves (actual temperatures)
    def composite(cs):
        Tb = np.unique(np.concatenate([c["T"] for c in cs]))
        Hc = np.array([sum(H_at(c, T) for c in cs) for T in Tb]); return Tb, Hc
    Th_c, Hh_c = composite(hot); Tc_c, Hc_c = composite(cold)
    Hc_c = Hc_c + QC      # cold composite shifted so that overlap is consistent with QC at the cold end
    # area target (counter-current, enthalpy intervals)
    Hgrid = np.unique(np.concatenate([Hh_c, Hc_c])); area = 0.0
    for i in range(len(Hgrid) - 1):
        h0, h1 = Hgrid[i], Hgrid[i + 1]; q = h1 - h0
        if q <= 1e-9: continue
        if h0 < Hh_c[0] - 1e-9 or h1 > Hh_c[-1] + 1e-9 or h0 < Hc_c[0] - 1e-9 or h1 > Hc_c[-1] + 1e-9: continue  # utility sections
        th0, th1 = np.interp([h0, h1], Hh_c, Th_c); tc0, tc1 = np.interp([h0, h1], Hc_c, Tc_c)
        d0, d1 = th0 - tc0, th1 - tc1
        if d0 <= 0 or d1 <= 0: continue
        lm = (d0 - d1) / np.log(d0 / d1) if abs(d0 - d1) > 1e-9 else d0
        area += q * 1000.0 / (U_TARGET * lm)
    return dict(dTmin=dtmin, Q_H_min_kW=QH, Q_C_min_kW=QC, pinch_T_shifted_K=pinch_T_shift, hot_total_kW=hot_total, cold_total_kW=cold_total,
                recovered_kW=recovered, area_target_m2=area, gcc_T=bounds, gcc_H=casc, hot_comp=(Th_c, Hh_c), cold_comp=(Tc_c, Hc_c))

def streams_from_rec(rec, cfg):
    fd = cfg["feed"]; S = []
    S.append(curve("hot", "Stage 1 gas 1200->950 K", 1200.0, 950.0, comp=rec["gas1"], solid=0.0))
    if rec["C1"] > 0: S.append(curve("hot", "Stage 1 solid C 1200->313 K", 1200.0, T_COND, comp=None, solid=rec["C1"]))
    if rec["Q2"] < 0: S.append(curve("hot", "Stage 2 exotherm @950 K", 950.0, 949.5, duty_kW=-rec["Q2"]))
    S.append(curve("hot", "Stage 2 gas 950->313 K (condensing)", 950.0, T_COND, comp=rec["gas2"]))
    if rec["Q3"] < 0: S.append(curve("hot", "Stage 3 exotherm @650 K", 650.0, 649.5, duty_kW=-rec["Q3"]))
    S.append(curve("hot", "Stage 3 gas 650->313 K (condensing)", 650.0, T_COND, comp=rec["gas3"]))
    if rec["C3"] > 0: S.append(curve("hot", "Stage 3 solid C 650->313 K", 650.0, T_COND, comp=None, solid=rec["C3"]))
    S.append(curve("cold", "fresh CH4 298->1200 K", T_AMB, 1200.0, comp={"CH4": FRESH_CH4}))
    if rec["ch4_rec"] > 0: S.append(curve("cold", "recycled CH4 313->1200 K", T_COND, 1200.0, comp={"CH4": rec["ch4_rec"]}))
    S.append(curve("cold", "Stage 1 endotherm @1200 K", 1200.0, 1200.5, duty_kW=rec["Q1"]))
    fresh2 = {"CO2": fd["co2_bio"] + fd["co2_dac"]}
    S.append(curve("cold", "fresh CO2 298->950 K", T_AMB, 950.0, comp=fresh2))
    if fd["h2"] > 0: S.append(curve("cold", "electrolytic H2 298->950 K", T_AMB, 950.0, comp={"H2": fd["h2"]}))
    if sum(rec["to_s2"].values()) > 0: S.append(curve("cold", "recycle to Stage 2 313->950 K", T_COND, 950.0, comp=rec["to_s2"]))
    if rec["Q2"] > 0: S.append(curve("cold", "Stage 2 endotherm @950 K", 950.0, 950.5, duty_kW=rec["Q2"]))
    S.append(curve("cold", "CFR feed 313->650 K", T_COND, 650.0, comp=rec["cfr_feed"]))
    return S

def stream_table(S):
    rows = []
    for c in S:
        rows.append(dict(kind=c["kind"], stream=c["name"], T_in_K=c["T_in"], T_out_K=c["T_out"], duty_kW=c["duty"]))
    return pd.DataFrame(rows)

def run_case(args):
    label, cfg = args; t0 = time.time()
    rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label)
    S = streams_from_rec(rec, cfg); res = {}
    for dt in DTMIN_LIST: res[dt] = pinch(S, dt)
    return dict(label=label, row=row, streams=stream_table(S), pinch=res, rec_Q=(rec["Q1"], rec["Q2"], rec["Q3"]), C1=rec["C1"], C3=rec["C3"])

def raw_biogas_stage1(y=0.60):
    co2 = FRESH_CH4 * (1 - y) / y; feed = {"CH4": FRESH_CH4, "CO2": co2}
    r = W.equilibrate_gas_plus_graphite(feed, 1200.0, ATM); g = r["gas_kmol_d"]
    fH = W.total_stream_enthalpy_J_per_day(feed, 0, 1200.0, ATM); pH = W.total_stream_enthalpy_J_per_day(g, r["Csolid_kmol_d"], 1200.0, ATM)
    ref = W.run_stage1_pyrolysis(W.CH4_tpd, 1200.0, ATM)
    return pd.DataFrame([
        dict(case="pure CH4 to Stage 1 (baseline)", CH4_in=FRESH_CH4, CO2_in=0.0, X_CH4=1 - ref["result"]["gas_kmol_d"].get("CH4", 0) / FRESH_CH4,
             C_solid_tpd=tpd("C(s)", ref["result"]["Csolid_kmol_d"]), CO_out=ref["result"]["gas_kmol_d"].get("CO", 0), H2_out=ref["result"]["gas_kmol_d"].get("H2", 0),
             H2O_out=0.0, Q1_kW=ref["Q_kW"], Q1_per_t_C_kWh=ref["Q_kW"] * 24 / tpd("C(s)", ref["result"]["Csolid_kmol_d"])),
        dict(case=f"raw biogas (y={y}) to Stage 1, no separation", CH4_in=FRESH_CH4, CO2_in=co2, X_CH4=1 - g.get("CH4", 0) / FRESH_CH4,
             C_solid_tpd=tpd("C(s)", r["Csolid_kmol_d"]), CO_out=g.get("CO", 0), H2_out=g.get("H2", 0), H2O_out=g.get("H2O", 0),
             Q1_kW=(pH - fH) / 86400e3, Q1_per_t_C_kWh=(pH - fH) / 86400e3 * 24 / max(tpd("C(s)", r["Csolid_kmol_d"]), 1e-9))])

def main():
    t0 = time.time(); F = F1m.feeds(); cases = []
    for cname in ["2b_y0.55", "2b_y0.60", "2b_y0.65"]:
        for h in [0.35, 0.50, 0.75]:
            cases.append((f"{cname} eq recycle h={h:g}", dict(config=cname, feed=F[cname], mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1",
                                                              P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=h, p=0.05, r=0.95)))
    sub = dict(config="submitted", feed=F["submitted"], mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1", P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.35)
    cases.append(("submitted eq recycle h=0.35", dict(sub, p=0.05, r=0.95)))
    cases.append(("submitted eq once-through h=0.35", dict(sub, p=1.0, r=0.0)))
    with Pool(min(len(cases), cpu_count())) as pool: res = pool.map(run_case, cases)
    R = {r["label"]: r for r in res}
    # tables
    rep = R["2b_y0.60 eq recycle h=0.5"]; rep["streams"].to_csv(OUT["streams_rep"], index=False)
    R["submitted eq once-through h=0.35"]["streams"].to_csv(OUT["streams_sub"], index=False)
    prow = []
    for r in res:
        Q1, Q2, Q3 = r["rec_Q"]; pool_ = -(min(Q2, 0) + min(Q3, 0)); demand = Q1 + max(Q2, 0)
        for dt, p in r["pinch"].items():
            prow.append(dict(case=r["label"], config=r["row"]["config"], y_CH4_biogas=r["row"]["y_CH4_biogas"], h_H2_to_CFR=r["row"]["h_H2_to_CFR"], recycle=r["row"]["recycle"],
                             dTmin_K=dt, Q1_kW=Q1, Q2_kW=Q2, Q3_kW=Q3, reaction_demand_Q1_plus_Q2pos_kW=demand, exothermic_pool_kW=pool_,
                             hot_streams_total_kW=p["hot_total_kW"], cold_streams_total_kW=p["cold_total_kW"], Q_H_min_kW=p["Q_H_min_kW"], Q_C_min_kW=p["Q_C_min_kW"],
                             process_heat_recovered_kW=p["recovered_kW"], pinch_T_shifted_K=p["pinch_T_shifted_K"], area_target_m2_U50=p["area_target_m2"],
                             hot_utility_without_integration_kW=p["cold_total_kW"], equivalent_recovery_fraction_F1b_convention=(demand - p["Q_H_min_kW"]) / pool_ if pool_ > 0 else np.nan,
                             share_of_Q1_covered_by_process_heat=max(0.0, (Q1 - p["Q_H_min_kW"]) / Q1)))
    pdf = pd.DataFrame(prow); pdf.to_csv(OUT["pinch"], index=False)
    # self-sufficiency
    p3 = pd.read_csv(P3_MAP); p3 = p3[p3.heat_recovery_fraction == 0.0]
    srows = []
    for r in res:
        if not r["row"]["recycle"]: continue
        lab = r["label"]; h2_exp = r["row"]["H2_net_exportable_tpd"]
        for dt, p in r["pinch"].items():
            h2_fired = p["Q_H_min_kW"] / ETA_FURNACE * 24 / LHV_KWH_KG / 1000.0
            for bnd in ["excl. upgrading (submitted boundary)", "incl. upgrading (base 0.25 kWh/Nm3)"]:
                for ename, eta in ETA_E.items():
                    m = p3[(p3.case == lab) & (p3.boundary == bnd) & (p3.power_source == ename)]
                    aux_kW = float(m.aux_power_kW.iloc[0]) if len(m) else np.nan
                    h2_aux = aux_kW * 24 / (LHV_KWH_KG * eta) / 1000.0
                    Q3 = r["rec_Q"][2]; q_bott_src = min(-Q3 if Q3 < 0 else 0.0, p["Q_C_min_kW"])   # Stage 3 exotherm at 650 K, rejected to cold utility
                    d = dict(case=lab, config=r["row"]["config"], y_CH4_biogas=r["row"]["y_CH4_biogas"], h_H2_to_CFR=r["row"]["h_H2_to_CFR"], dTmin_K=dt,
                             CO2_carbon_fixed_fraction=r["row"]["CO2_carbon_fixed_fraction"], C_total_tpd=r["row"]["C_total_tpd"],
                             Q_H_min_kW=p["Q_H_min_kW"], H2_exportable_tpd=h2_exp, H2_fired_tpd=h2_fired, H2_after_heat_tpd=h2_exp - h2_fired,
                             boundary=bnd, power_source=ename, aux_power_kW=aux_kW, H2_for_aux_tpd=h2_aux, H2_net_final_tpd=h2_exp - h2_fired - h2_aux,
                             self_sufficient=(h2_exp - h2_fired - h2_aux) >= 0, bottoming_heat_source_650K_kW=q_bott_src)
                    for eb in (0.20, 0.25):   # bottoming power cycle (steam/ORC) on the 650 K Stage 3 exotherm
                        P_b = eb * q_bott_src; aux_res = max(0.0, aux_kW - P_b); h2_res = aux_res * 24 / (LHV_KWH_KG * eta) / 1000.0
                        d[f"bottoming_power_eta{eb:.2f}_kW"] = P_b; d[f"aux_residual_eta{eb:.2f}_kW"] = aux_res
                        d[f"H2_net_final_with_bottoming_eta{eb:.2f}_tpd"] = h2_exp - h2_fired - h2_res
                    srows.append(d)
    sdf = pd.DataFrame(srows); sdf.to_csv(OUT["self"], index=False)
    rawdf = raw_biogas_stage1(0.60); rawdf.to_csv(OUT["raw"], index=False)
    # figure
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for j, (lab, ttl) in enumerate([("2b_y0.60 eq recycle h=0.5", "X4 (2b, y=0.60, h=0.50), recycle equilibrium"), ("submitted eq once-through h=0.35", "Submitted chain (Feed A), once-through equilibrium")]):
        p = R[lab]["pinch"][20.0]
        Th, Hh = p["hot_comp"]; Tc, Hc = p["cold_comp"]
        ax[0, j].plot(Hh / 1000, Th, "r-", label="hot composite"); ax[0, j].plot(Hc / 1000, Tc, "b-", label="cold composite (shifted by Q_C,min)")
        ax[0, j].axhline(650, color="gray", ls=":", lw=0.8); ax[0, j].axhline(1200, color="gray", ls=":", lw=0.8)
        ax[0, j].set_xlabel("enthalpy flow [MW]"); ax[0, j].set_ylabel("T [K]"); ax[0, j].set_title(f"{ttl}\ncomposite curves, dTmin = 20 K: Q_H,min = {p['Q_H_min_kW']/1000:.2f} MW, Q_C,min = {p['Q_C_min_kW']/1000:.2f} MW")
        ax[0, j].legend(fontsize=8); ax[0, j].grid(alpha=.3)
        for dt, col in [(20.0, "k"), (30.0, "gray")]:
            q = R[lab]["pinch"][dt]; ax[1, j].plot(q["gcc_H"] / 1000, q["gcc_T"], color=col, label=f"GCC dTmin = {dt:g} K (Q_H,min {q['Q_H_min_kW']/1000:.2f} MW)")
        ax[1, j].set_xlabel("net heat flow [MW]"); ax[1, j].set_ylabel("shifted T [K]"); ax[1, j].set_title("grand composite curve"); ax[1, j].legend(fontsize=8); ax[1, j].grid(alpha=.3)
    fig.suptitle("T4 heat cascade (equilibrium values; reaction duties isothermal at stage temperature, feed preheat and product cooling as separate streams)")
    fig.tight_layout(); fig.savefig(OUT["png"], dpi=160); plt.close(fig)
    # summary
    L = [f"T4 heat cascade (wall {time.time()-t0:.0f} s). EQUILIBRIUM values."]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 200, "display.float_format", lambda v: f"{v:,.1f}"):
        L.append("\n--- Streams, X4 representative (2b, y=0.60, h=0.50) ---\n" + rep["streams"].to_string(index=False))
        L.append("\n--- Streams, submitted once-through ---\n" + R["submitted eq once-through h=0.35"]["streams"].to_string(index=False))
        L.append("\n--- Pinch results ---\n" + pdf[["case", "dTmin_K", "Q1_kW", "Q2_kW", "Q3_kW", "hot_streams_total_kW", "cold_streams_total_kW", "Q_H_min_kW", "Q_C_min_kW", "process_heat_recovered_kW", "pinch_T_shifted_K", "area_target_m2_U50", "equivalent_recovery_fraction_F1b_convention", "share_of_Q1_covered_by_process_heat"]].to_string(index=False))
        L.append("\n--- Raw biogas to Stage 1 ---\n" + rawdf.to_string(index=False))
    for bnd in sdf.boundary.unique():
        for ename in ETA_E:
            sel = sdf[(sdf.boundary == bnd) & (sdf.power_source == ename) & (sdf.config != "submitted")]
            piv = sel.pivot_table(index=["y_CH4_biogas", "h_H2_to_CFR"], columns="dTmin_K", values="H2_net_final_tpd")
            L.append(f"\n--- Final net H2 [t/d] with pinch-based hot utility | {bnd} | {ename} ---\n" + piv.to_string(float_format=lambda v: f"{v:+.2f}"))
            for eb in (0.20, 0.25):
                pv2 = sel.pivot_table(index=["y_CH4_biogas", "h_H2_to_CFR"], columns="dTmin_K", values=f"H2_net_final_with_bottoming_eta{eb:.2f}_tpd")
                L.append(f"    with bottoming power cycle on the 650 K exotherm, eta_el = {eb:.2f}:\n" + pv2.to_string(float_format=lambda v: f"{v:+.2f}"))
    with open(OUT["txt"], "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", *OUT.values())

if __name__ == "__main__":
    main()
