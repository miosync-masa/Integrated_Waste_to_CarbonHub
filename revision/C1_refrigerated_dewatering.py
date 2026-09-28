"""
C1_refrigerated_dewatering.py -- make the heat side consistent with the 95 % water removal of the mass balance.

The flowsheet removes 95 % of the water after Stage 2 and after Stage 3 (W.remove_species, composition only); the
heat cascade (T4) and the auxiliary table (P3) assumed condensation at 313.15 K, which at 1 atm condenses only
~70 % (Stage 2 outlet) and ~89 % (Stage 3 outlet). This script keeps the 95 % removal (so every mass balance,
yield and fixation number is unchanged) and adds the refrigerated dewatering it implies:
  1. condenser outlet temperature T_req at which the saturated vapour left in the gas equals 5 % of the water
     (Antoine vapour pressure as in T4), per case: 2b y in {0.55, 0.60, 0.65} x h in {0.35, 0.50, 0.75} equilibrium
     recycle, X4 once-through, and the kinetic design case loop;
  2. refrigeration duty = enthalpy removed below 313.15 K (sensible + latent, T4 stream model) and the chiller power
     with COP = eta_II * T_evap / (T_cond - T_evap), T_evap = T_req - 5 K, T_cond = 318.15 K (air-cooled condenser,
     35 C ambient + 10 K approach), eta_II = 0.45 (0.35 / 0.55 sensitivity) -- second-law efficiency typical of
     vapour-compression chillers (assumption; e.g. Gordon & Ng, Cool Thermodynamics, 2000; ASHRAE Handbook -- to be
     confirmed by the authors, 要確認);
  3. pinch analysis (T4 problem table) with the condensing hot streams extended to T_req and the recycle/CFR cold
     streams starting at T_req; the cold utility is split at a hot-side temperature of 313.15 K into cooling
     water/air (above) and refrigeration (below: the sub-313 K duty of the two condensers, which no process cold stream can absorb);
     for comparison the T4 convention (313 K) is recomputed for every case;
  4. auxiliary table (P3 items recomputed on the same converged recycle) + chiller compressor + fans for the chiller
     heat rejection; self-sufficiency map (T4 convention: H2 fired for Q_H,min / 0.85, aux via fuel cell 0.50 or engine
     0.40, bottoming 0 / 0.20 / 0.25 on the 650 K exotherm) and the T4b options (membrane / amine upgrading);
  5. a compact table for the TEA (E7): Q_H,min, HEN area target, air-cooler duty, chiller duty and power, aux load.
EQUILIBRIUM values unless the case name says kinetic. Refrigerant properties are not modelled (Carnot-based COP).

Outputs (revision/Result/): C1_condensers.csv, C1_streams_X4_rep.csv, C1_streams_X4_once.csv, C1_pinch_results.csv,
  C1_aux_items.csv, C1_aux_totals.csv, C1_selfsufficiency.csv, C1_surplus_heat_use.csv, C1_for_TEA.csv,
  C1_composite_curves.png, C1_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python C1_refrigerated_dewatering.py   (~4 min, parallel)
"""
import os, time
import numpy as np, pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W
import F1_full_biogas_CO2 as F1m
import T4_heat_cascade as T4m
import T4b_surplus_heat_use as T4b
import P3_auxiliary_power as P3m
from P1_recycle_analysis import FRESH_CH4, tpd, add

ATM = F1m.ATM; T_CW = T4m.T_COND; T_AMB = T4m.T_AMB; DTS = T4m.DTMIN_LIST; WRF = W.water_remove_frac
ETA_FURNACE, LHV, ETA_E, U = T4m.ETA_FURNACE, T4m.LHV_KWH_KG, T4m.ETA_E, T4m.U_TARGET
NM3_PER_KMOL = 22.414
COP = dict(T_cond_K=318.15, dT_evap_K=5.0, eta_II=0.45, eta_II_range=(0.35, 0.55))
FAN_FRACTION = P3m.FAN_FRACTION

def t_required(comp, removal=WRF):
    """Outlet temperature at which the saturated vapour equals (1 - removal) of the inlet water; and the removal reached at 313.15 K."""
    comp = W.clean_species_dict(comp); nw = comp.get("H2O", 0.0); ndry = sum(v for k, v in comp.items() if k != "H2O")
    if nw <= 0: return np.nan, np.nan, np.nan
    nvap_t = (1 - removal) * nw; ps_t = nvap_t / (ndry + nvap_t)
    lo, hi = 253.15, 373.15
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if T4m.psat_water_atm(mid) > ps_t: hi = mid
        else: lo = mid
    T_req = 0.5 * (lo + hi)
    ps313 = T4m.psat_water_atm(T_CW); nvap313 = min(nw, ndry * ps313 / (1 - ps313)); rem313 = 1 - nvap313 / nw
    return T_req, rem313, ps_t

def cop(T_req, eta=COP["eta_II"]):
    Te = T_req - COP["dT_evap_K"]; return eta * Te / (COP["T_cond_K"] - Te)

def refrig_duty(comp, T_in, T_req):
    """(duty above 313 K = cooling water/air, duty below 313 K = refrigeration) for a condensing stream cooled T_in -> T_req."""
    h_in, h_cw, h_out = (T4m.h_stream_kW(comp, T) for T in (T_in, T_CW, T_req))
    return h_in - h_cw, h_cw - h_out

def streams_refrig(rec, cfg, T2out, T3out):
    fd = cfg["feed"]; S = []
    S.append(T4m.curve("hot", "Stage 1 gas 1200->950 K", 1200.0, 950.0, comp=rec["gas1"]))
    if rec["C1"] > 0: S.append(T4m.curve("hot", "Stage 1 solid C 1200->313 K", 1200.0, T_CW, solid=rec["C1"]))
    if rec["Q2"] < 0: S.append(T4m.curve("hot", "Stage 2 exotherm @950 K", 950.0, 949.5, duty_kW=-rec["Q2"]))
    S.append(T4m.curve("hot", f"Stage 2 gas 950->{T2out:.0f} K (condensing, refrigerated below 313 K)", 950.0, T2out, comp=rec["gas2"]))
    if rec["Q3"] < 0: S.append(T4m.curve("hot", "Stage 3 exotherm @650 K", 650.0, 649.5, duty_kW=-rec["Q3"]))
    S.append(T4m.curve("hot", f"Stage 3 gas 650->{T3out:.0f} K (condensing, refrigerated below 313 K)", 650.0, T3out, comp=rec["gas3"]))
    if rec["C3"] > 0: S.append(T4m.curve("hot", "Stage 3 solid C 650->313 K", 650.0, T_CW, solid=rec["C3"]))
    S.append(T4m.curve("cold", "fresh CH4 298->1200 K", T_AMB, 1200.0, comp={"CH4": FRESH_CH4}))
    if rec["ch4_rec"] > 0: S.append(T4m.curve("cold", f"recycled CH4 {T3out:.0f}->1200 K", T3out, 1200.0, comp={"CH4": rec["ch4_rec"]}))
    S.append(T4m.curve("cold", "Stage 1 endotherm @1200 K", 1200.0, 1200.5, duty_kW=rec["Q1"]))
    S.append(T4m.curve("cold", "fresh CO2 298->950 K", T_AMB, 950.0, comp={"CO2": fd["co2_bio"] + fd["co2_dac"]}))
    if fd["h2"] > 0: S.append(T4m.curve("cold", "electrolytic H2 298->950 K", T_AMB, 950.0, comp={"H2": fd["h2"]}))
    if sum(rec["to_s2"].values()) > 0: S.append(T4m.curve("cold", f"recycle to Stage 2 {T3out:.0f}->950 K", T3out, 950.0, comp=rec["to_s2"]))
    if rec["Q2"] > 0: S.append(T4m.curve("cold", "Stage 2 endotherm @950 K", 950.0, 950.5, duty_kW=rec["Q2"]))
    S.append(T4m.curve("cold", f"CFR feed {T2out:.0f}->650 K", T2out, 650.0, comp=rec["cfr_feed"]))
    return S

def aux_with_chiller(rec, cfg, q_ref2, q_ref3, cop2, cop3):
    df, tot = P3m.aux_items(rec, cfg)
    P2, P3_ = q_ref2 / cop2, q_ref3 / cop3; rej = (q_ref2 + P2) + (q_ref3 + P3_)
    new = [dict(item="H1_chiller_S2_condenser", name=f"chiller, Stage 2 condenser to {cfg['_T2out']:.1f} K (COP {cop2:.2f})", kW=P2, basis=f"duty {q_ref2:,.0f} kW below 313 K", source="Carnot x eta_II 0.45, T_cond 318 K (assumption)"),
           dict(item="H2_chiller_S3_condenser", name=f"chiller, Stage 3 condenser to {cfg['_T3out']:.1f} K (COP {cop3:.2f})", kW=P3_, basis=f"duty {q_ref3:,.0f} kW below 313 K", source="Carnot x eta_II 0.45, T_cond 318 K (assumption)"),
           dict(item="H3_chiller_heat_rejection_fans", name=f"air-cooled chiller condensers, {FAN_FRACTION*100:.1f} % of {rej:,.0f} kW rejected", kW=rej * FAN_FRACTION, basis="duty + compressor work", source="assumption 1-2 % of duty")]
    df = pd.concat([df[df.item != "G_other"], pd.DataFrame(new)], ignore_index=True)
    base_codes = ["B_H2membrane_base", "C_CH4separation_base", "D_recycle_blower", "D_feed_blower", "E_condenser_fans", "F_solids_handling", "H1_chiller_S2_condenser", "H2_chiller_S3_condenser", "H3_chiller_heat_rejection_fans"]
    sub = df[df.item.isin(base_codes)].kW.sum(); df = pd.concat([df, pd.DataFrame([dict(item="G_other", name=f"other ({P3m.OTHER_FRAC*100:.0f} % of B-H)", kW=sub * P3m.OTHER_FRAC, basis="", source="assumption")])], ignore_index=True)
    excl = sub * (1 + P3m.OTHER_FRAC); up = {k: float(df[df.item == f"A_upgrading_{k}"].kW.iloc[0]) for k in ("low", "base", "high")}
    return df, dict(chiller_kW=P2 + P3_, chiller_fans_kW=rej * FAN_FRACTION, chiller_heat_rejected_kW=rej, aux_excl_upgrading_kW=excl, aux_incl_upgrading_kW=excl + up["base"],
                    aux_incl_upgrading_low_kW=excl + up["low"], aux_incl_upgrading_high_kW=excl + up["high"], aux_excl_upgrading_P3_original_kW=tot["aux_excl_upgrading_kW"], duty_cooling_to_313K_kW=tot["duty_cooling_kW"])

def run(args):
    label, cfg = args; t0 = time.time()
    rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label)
    T2o, rem2_313, _ = t_required(rec["gas2"]); T3o, rem3_313, _ = t_required(rec["gas3"])
    cw2, rf2 = refrig_duty(rec["gas2"], W.T_rwgs, T2o); cw3, rf3 = refrig_duty(rec["gas3"], W.T_cfr, T3o)
    cop2, cop3 = cop(T2o), cop(T3o); cfg["_T2out"], cfg["_T3out"] = T2o, T3o
    cond = dict(case=label, config=cfg["config"], y_CH4_biogas=cfg["feed"]["y"], h_H2_to_CFR=cfg["h"], recycle=cfg["p"] < 1, value_type=row["value_type"],
                S2_out_water_kmol_d=rec["gas2"].get("H2O", 0), S2_out_dry_kmol_d=sum(v for k, v in rec["gas2"].items() if k != "H2O"), S2_T_required_K=T2o, S2_removal_at_313K=rem2_313,
                S2_cooling_950_to_313_kW=cw2, S2_refrigeration_below_313_kW=rf2, S2_COP=cop2, S2_chiller_kW=rf2 / cop2,
                S3_out_water_kmol_d=rec["gas3"].get("H2O", 0), S3_out_dry_kmol_d=sum(v for k, v in rec["gas3"].items() if k != "H2O"), S3_T_required_K=T3o, S3_removal_at_313K=rem3_313,
                S3_cooling_650_to_313_kW=cw3, S3_refrigeration_below_313_kW=rf3, S3_COP=cop3, S3_chiller_kW=rf3 / cop3,
                refrigeration_total_kW=rf2 + rf3, chiller_power_total_kW=rf2 / cop2 + rf3 / cop3,
                chiller_power_etaII_035_kW=rf2 / cop(T2o, 0.35) + rf3 / cop(T3o, 0.35), chiller_power_etaII_055_kW=rf2 / cop(T2o, 0.55) + rf3 / cop(T3o, 0.55),
                water_recovered_tpd=row["water_total_tpd"], below_freezing=(min(T2o, T3o) < 273.15))
    aux_df, aux_tot = aux_with_chiller(rec, cfg, rf2, rf3, cop2, cop3); aux_df.insert(0, "case", label)
    out = dict(label=label, row=row, cond=cond, aux_df=aux_df, aux_tot=aux_tot, Q=(rec["Q1"], rec["Q2"], rec["Q3"]), pw3=sum(rec["pw3"].values()), wall=time.time() - t0)
    if cfg["mode"] == "equilibrium":
        S_new = streams_refrig(rec, cfg, T2o, T3o); S_old = T4m.streams_from_rec(rec, cfg)
        out["streams"] = T4m.stream_table(S_new); out["pinch"] = {dt: T4m.pinch(S_new, dt) for dt in DTS}; out["pinch_313"] = {dt: T4m.pinch(S_old, dt) for dt in DTS}
        for dt in DTS:
            # cold-utility placement: cooling water/air takes every hot-side duty at >= 313.15 K; refrigeration takes only the heat the
            # condensing streams release below 313.15 K (no process cold stream can absorb it: the fresh feeds at 298 K sit above the
            # shifted hot temperature of that heat for dTmin >= 15 K), i.e. exactly the sub-313 K duties of the two condensers
            p = out["pinch"][dt]; p["Q_refrig_utility_kW"] = rf2 + rf3; p["Q_cw_utility_kW"] = p["Q_C_min_kW"] - p["Q_refrig_utility_kW"]
    return out

def main():
    t0 = time.time(); F = F1m.feeds(); cases = []
    for cname in ["2b_y0.55", "2b_y0.60", "2b_y0.65"]:
        for h in [0.35, 0.50, 0.75]:
            cases.append((f"{cname} eq recycle h={h:g}", dict(config=cname, feed=F[cname], mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1", P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=h, p=0.05, r=0.95)))
    cases.append(("2b_y0.60 eq once-through h=0.5", dict(config="2b_y0.60", feed=F["2b_y0.60"], mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1", P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.5, p=1.0, r=0.0)))
    cases.append(("2b_y0.60 kin recycle phi=1 h=0.5", dict(config="2b_y0.60", feed=F["2b_y0.60"], mode="kinetic", phi=1.0, k2=F1m.k2_from_phi(1.0), n=None, variant="V1", P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.5, p=0.05, r=0.95)))
    with Pool(min(len(cases), cpu_count())) as pool: res = pool.map(run, cases)
    R = {r["label"]: r for r in res}
    COND = pd.DataFrame([r["cond"] for r in res]); COND.to_csv(os.path.join(RESULT, "C1_condensers.csv"), index=False)
    AUX = pd.concat([r["aux_df"] for r in res], ignore_index=True); AUX.to_csv(os.path.join(RESULT, "C1_aux_items.csv"), index=False)
    AT = pd.DataFrame([dict(case=r["label"], **r["aux_tot"]) for r in res]); AT.to_csv(os.path.join(RESULT, "C1_aux_totals.csv"), index=False)
    R["2b_y0.60 eq recycle h=0.5"]["streams"].to_csv(os.path.join(RESULT, "C1_streams_X4_rep.csv"), index=False)
    R["2b_y0.60 eq once-through h=0.5"]["streams"].to_csv(os.path.join(RESULT, "C1_streams_X4_once.csv"), index=False)
    prow = []
    for r in res:
        if "pinch" not in r: continue
        Q1, Q2, Q3 = r["Q"]; pool_ = -(min(Q2, 0) + min(Q3, 0)); demand = Q1 + max(Q2, 0)
        for conv, PP in (("refrigerated to T_req (this work)", r["pinch"]), ("T4 convention (313 K)", r["pinch_313"])):
            for dt, p in PP.items():
                prow.append(dict(case=r["label"], convention=conv, y_CH4_biogas=r["row"]["y_CH4_biogas"], h_H2_to_CFR=r["row"]["h_H2_to_CFR"], recycle=r["row"]["recycle"], dTmin_K=dt,
                                 Q1_kW=Q1, Q2_kW=Q2, Q3_kW=Q3, hot_streams_total_kW=p["hot_total_kW"], cold_streams_total_kW=p["cold_total_kW"], Q_H_min_kW=p["Q_H_min_kW"], Q_C_min_kW=p["Q_C_min_kW"],
                                 Q_cw_utility_kW=p.get("Q_cw_utility_kW", p["Q_C_min_kW"]), Q_refrig_utility_kW=p.get("Q_refrig_utility_kW", 0.0), process_heat_recovered_kW=p["recovered_kW"],
                                 pinch_T_shifted_K=p["pinch_T_shifted_K"], area_target_m2_U50=p["area_target_m2"], equivalent_recovery_fraction_F1b_convention=(demand - p["Q_H_min_kW"]) / pool_ if pool_ > 0 else np.nan))
    PIN = pd.DataFrame(prow); PIN.to_csv(os.path.join(RESULT, "C1_pinch_results.csv"), index=False)
    # self-sufficiency (T4 convention of the accounting, with chiller in the auxiliary load)
    srows = []
    for r in res:
        if "pinch" not in r or not r["row"]["recycle"]: continue
        lab = r["label"]; h2_exp = r["row"]["H2_net_exportable_tpd"]; Q3 = r["Q"][2]
        for dt, p in r["pinch"].items():
            h2_fired = p["Q_H_min_kW"] / ETA_FURNACE * 24 / LHV / 1000.0
            for bnd, aux_kW in [("excl. upgrading (submitted boundary)", r["aux_tot"]["aux_excl_upgrading_kW"]), ("incl. upgrading (base 0.25 kWh/Nm3)", r["aux_tot"]["aux_incl_upgrading_kW"])]:
                for ename, eta in ETA_E.items():
                    h2_aux = aux_kW * 24 / (LHV * eta) / 1000.0; q_b = min(-Q3 if Q3 < 0 else 0.0, p["Q_C_min_kW"])
                    d = dict(case=lab, config=r["row"]["config"], y_CH4_biogas=r["row"]["y_CH4_biogas"], h_H2_to_CFR=r["row"]["h_H2_to_CFR"], dTmin_K=dt, CO2_carbon_fixed_fraction=r["row"]["CO2_carbon_fixed_fraction"], C_total_tpd=r["row"]["C_total_tpd"],
                             Q_H_min_kW=p["Q_H_min_kW"], Q_refrig_utility_kW=p["Q_refrig_utility_kW"], chiller_kW=r["aux_tot"]["chiller_kW"], H2_exportable_tpd=h2_exp, H2_fired_tpd=h2_fired, H2_after_heat_tpd=h2_exp - h2_fired,
                             boundary=bnd, power_source=ename, aux_power_kW=aux_kW, H2_for_aux_tpd=h2_aux, H2_net_final_tpd=h2_exp - h2_fired - h2_aux, self_sufficient=(h2_exp - h2_fired - h2_aux) >= 0, bottoming_heat_source_650K_kW=q_b)
                    for eb in (0.20, 0.25):
                        P_b = eb * q_b; res_kW = max(0.0, aux_kW - P_b); d[f"bottoming_power_eta{eb:.2f}_kW"] = P_b; d[f"aux_residual_eta{eb:.2f}_kW"] = res_kW
                        d[f"H2_net_final_with_bottoming_eta{eb:.2f}_tpd"] = h2_exp - h2_fired - res_kW * 24 / (LHV * eta) / 1000.0
                    srows.append(d)
    SELF = pd.DataFrame(srows); SELF.to_csv(os.path.join(RESULT, "C1_selfsufficiency.csv"), index=False)
    # T4b options for the three h = 0.50 cases (dTmin 20 K), chiller included in aux_excl
    brows = []
    for r in res:
        if "pinch" not in r or r["row"]["h_H2_to_CFR"] != 0.5 or not r["row"]["recycle"]: continue
        y = r["row"]["y_CH4_biogas"]; raw = FRESH_CH4 * NM3_PER_KMOL / y; p = r["pinch"][T4b.DT]; QH, QC, Q3 = p["Q_H_min_kW"], p["Q_C_min_kW"], r["Q"][2]
        h2_exp = r["row"]["H2_net_exportable_tpd"]; h2_fired = QH / ETA_FURNACE * 24 / LHV / 1000.0; h2_heat = h2_exp - h2_fired; aux_excl = r["aux_tot"]["aux_excl_upgrading_kW"]
        q_plateau = min(-Q3, QC); up_m = raw * T4b.E_MEMBRANE / 24.0; up_ae = raw * T4b.E_AMINE_EL / 24.0; up_ah = raw * T4b.Q_AMINE_HEAT / 24.0
        base = dict(case=r["label"], y_CH4_biogas=y, Q_H_min_kW=QH, Q_C_min_kW=QC, Q3_kW=Q3, plateau_650K_kW=q_plateau, H2_exportable_tpd=h2_exp, H2_fired_tpd=h2_fired, H2_after_heat_tpd=h2_heat, aux_excl_upgrading_kW=aux_excl, chiller_kW=r["aux_tot"]["chiller_kW"])
        for ename, eta in ETA_E.items():
            h2_el = lambda kW: kW * 24 / (LHV * eta) / 1000.0
            for eb in [None] + T4b.ETA_B:
                P_b = 0.0 if eb is None else eb * q_plateau
                brows.append(dict(base, option="ref: upgrading outside boundary" + ("" if eb is None else f", bottoming eta {eb:.2f}"), power_source=ename, electricity_demand_kW=aux_excl, bottoming_power_kW=P_b, H2_net_final_tpd=h2_heat - h2_el(max(0.0, aux_excl - P_b))))
                dem = aux_excl + up_m
                brows.append(dict(base, option="(i) membrane upgrading" + (", no bottoming" if eb is None else f", all plateau to power eta {eb:.2f}"), power_source=ename, electricity_demand_kW=dem, bottoming_power_kW=P_b, H2_net_final_tpd=h2_heat - h2_el(max(0.0, dem - P_b))))
            for rlab, Treb in T4b.T_REB.items():
                avail = T4b.gcc_at(p, Treb + T4b.DT / 2); q_am = min(up_ah, avail); short = up_ah - q_am; el_am = up_ae + (raw * 0.05 / 24.0 if Treb < 370 else 0.0); dem = aux_excl + el_am
                brows.append(dict(base, option=f"(ii) amine upgrading, reboiler {rlab}, no bottoming", power_source=ename, electricity_demand_kW=dem, bottoming_power_kW=0.0, heat_to_amine_kW=q_am, heat_shortfall_kW=short, H2_net_final_tpd=h2_heat - h2_el(dem) - short / ETA_FURNACE * 24 / LHV / 1000.0))
                q_pow = max(0.0, min(q_plateau, avail - q_am))
                for eb in T4b.ETA_B:
                    P_b = eb * q_pow
                    brows.append(dict(base, option=f"(iii) amine upgrading, reboiler {rlab}, bottoming eta {eb:.2f} on remaining heat", power_source=ename, electricity_demand_kW=dem, bottoming_power_kW=P_b, heat_to_amine_kW=q_am, heat_shortfall_kW=short, H2_net_final_tpd=h2_heat - h2_el(max(0.0, dem - P_b)) - short / ETA_FURNACE * 24 / LHV / 1000.0))
    SURP = pd.DataFrame(brows); SURP.to_csv(os.path.join(RESULT, "C1_surplus_heat_use.csv"), index=False)
    # table for the TEA
    trows = []
    for r in res:
        c, a = r["cond"], r["aux_tot"]; p = r["pinch"][20.0] if "pinch" in r else None
        trows.append(dict(case=r["label"], value_type=r["row"]["value_type"], y_CH4_biogas=r["row"]["y_CH4_biogas"], h_H2_to_CFR=r["row"]["h_H2_to_CFR"], recycle=r["row"]["recycle"],
                          Q1_kW=r["Q"][0], Q2_kW=r["Q"][1], Q3_kW=r["Q"][2], Q_H_min_kW_dT20=(p["Q_H_min_kW"] if p else r["Q"][0] + max(r["Q"][1], 0.0)),
                          H2_fired_tpd=(p["Q_H_min_kW"] if p else r["Q"][0] + max(r["Q"][1], 0.0)) / ETA_FURNACE * 24 / LHV / 1000.0,
                          hen_area_m2_U50=(p["area_target_m2"] if p else np.nan), Q_C_min_kW=(p["Q_C_min_kW"] if p else np.nan), Q_cw_utility_kW=(p["Q_cw_utility_kW"] if p else np.nan), Q_refrig_utility_kW=(p["Q_refrig_utility_kW"] if p else c["refrigeration_total_kW"]),
                          refrigeration_duty_streams_kW=c["refrigeration_total_kW"], chiller_kW=a["chiller_kW"], chiller_heat_rejected_kW=a["chiller_heat_rejected_kW"],
                          air_cooler_duty_kW=a["duty_cooling_to_313K_kW"] + a["chiller_heat_rejected_kW"], aux_excl_upgrading_kW=a["aux_excl_upgrading_kW"], aux_incl_upgrading_kW=a["aux_incl_upgrading_kW"],
                          aux_excl_upgrading_P3_original_kW=a["aux_excl_upgrading_P3_original_kW"], S2_T_required_K=c["S2_T_required_K"], S3_T_required_K=c["S3_T_required_K"],
                          H2_net_exportable_tpd=r["row"]["H2_net_exportable_tpd"], C_total_tpd=r["row"]["C_total_tpd"], water_total_tpd=r["row"]["water_total_tpd"], CO2_fixed_as_CO2_tpd=r["row"]["CO2_fixed_as_CO2_tpd"],
                          S2_inlet_total_kmol_d=r["row"]["S2_inlet_total_kmol_d"], pw3_kmol_d=r["pw3"], C_stage3_tpd=r["row"]["C_stage3_tpd"]))
    TEA = pd.DataFrame(trows); TEA.to_csv(os.path.join(RESULT, "C1_for_TEA.csv"), index=False)
    # figure: X4 recycle and X4 once-through, refrigerated convention, dTmin 20 K
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for j, (lab, ttl) in enumerate([("2b_y0.60 eq recycle h=0.5", "Base case (recycle, equilibrium)"), ("2b_y0.60 eq once-through h=0.5", "Same feed, once-through (equilibrium)")]):
        p = R[lab]["pinch"][20.0]; Th, Hh = p["hot_comp"]; Tc, Hc = p["cold_comp"]
        ax[0, j].plot(Hh / 1000, Th, "r-", label="hot composite"); ax[0, j].plot(Hc / 1000, Tc, "b-", label="cold composite (shifted by Q_C,min)")
        for T in (313.15, 650, 1200): ax[0, j].axhline(T, color="gray", ls=":", lw=0.8)
        ax[0, j].set_xlabel("enthalpy flow [MW]"); ax[0, j].set_ylabel("T [K]")
        ax[0, j].set_title(f"{ttl}\ndTmin = 20 K: Q_H,min = {p['Q_H_min_kW']/1000:.2f} MW, Q_C,min = {p['Q_C_min_kW']/1000:.2f} MW (refrigeration {p['Q_refrig_utility_kW']/1000:.2f} MW)", fontsize=9)
        ax[0, j].legend(fontsize=8); ax[0, j].grid(alpha=.3)
        for dt, col in [(20.0, "k"), (30.0, "gray")]:
            q = R[lab]["pinch"][dt]; ax[1, j].plot(q["gcc_H"] / 1000, q["gcc_T"], color=col, label=f"GCC dTmin = {dt:g} K (Q_H,min {q['Q_H_min_kW']/1000:.2f} MW)")
        ax[1, j].axhline(313.15 - 10, color="c", ls="--", lw=0.8, label="cooling-water limit (hot side 313 K, shifted)")
        ax[1, j].set_xlabel("net heat flow [MW]"); ax[1, j].set_ylabel("shifted T [K]"); ax[1, j].set_title("grand composite curve"); ax[1, j].legend(fontsize=8); ax[1, j].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULT, "C1_composite_curves.png"), dpi=160); plt.close(fig)
    # summary
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300); ff = lambda v: f"{v:,.1f}" if abs(v) >= 10 else f"{v:.3f}"
    rep = R["2b_y0.60 eq recycle h=0.5"]
    L = [f"C1 refrigerated dewatering (wall {time.time()-t0:.0f} s). Water removal 95 % kept (mass balance unchanged). COP = {COP['eta_II']} x T_evap/(T_cond - T_evap), T_evap = T_req - {COP['dT_evap_K']:.0f} K, T_cond = {COP['T_cond_K']} K.",
         "", "--- condenser outlet temperatures and refrigeration ---",
         COND[["case", "S2_T_required_K", "S2_removal_at_313K", "S2_refrigeration_below_313_kW", "S2_COP", "S2_chiller_kW", "S3_T_required_K", "S3_removal_at_313K", "S3_refrigeration_below_313_kW", "S3_COP", "S3_chiller_kW", "chiller_power_total_kW", "chiller_power_etaII_035_kW", "chiller_power_etaII_055_kW", "below_freezing"]].to_string(index=False, float_format=ff),
         "", "--- streams, X4 representative, refrigerated ---", rep["streams"].to_string(index=False, float_format=ff),
         "", "--- pinch results (both conventions) ---",
         PIN[["case", "convention", "dTmin_K", "Q_H_min_kW", "Q_C_min_kW", "Q_cw_utility_kW", "Q_refrig_utility_kW", "process_heat_recovered_kW", "pinch_T_shifted_K", "area_target_m2_U50"]].to_string(index=False, float_format=ff),
         "", "--- auxiliary totals (P3 items + chiller) ---", AT.to_string(index=False, float_format=ff),
         "", "--- items, X4 representative ---", AUX[AUX.case == "2b_y0.60 eq recycle h=0.5"][["item", "name", "kW", "basis"]].to_string(index=False, float_format=ff),
         "", "--- table for the TEA ---", TEA.to_string(index=False, float_format=ff)]
    for bnd in SELF.boundary.unique():
        for ename in ETA_E:
            sel = SELF[(SELF.boundary == bnd) & (SELF.power_source == ename)]
            L.append(f"\n--- Final net H2 [t/d] | {bnd} | {ename} | no bottoming ---\n" + sel.pivot_table(index=["y_CH4_biogas", "h_H2_to_CFR"], columns="dTmin_K", values="H2_net_final_tpd").to_string(float_format=lambda v: f"{v:+.2f}"))
            for eb in (0.20, 0.25): L.append(f"    with bottoming eta {eb:.2f}:\n" + sel.pivot_table(index=["y_CH4_biogas", "h_H2_to_CFR"], columns="dTmin_K", values=f"H2_net_final_with_bottoming_eta{eb:.2f}_tpd").to_string(float_format=lambda v: f"{v:+.2f}"))
    L.append("\n--- T4b options with chiller (dTmin 20 K, h = 0.50) ---\n" + SURP[["case", "option", "power_source", "electricity_demand_kW", "bottoming_power_kW", "H2_net_final_tpd"]].to_string(index=False, float_format=ff))
    open(os.path.join(RESULT, "C1_summary.txt"), "w").write("\n".join(L)); print("\n".join(L[:12]))

if __name__ == "__main__":
    main()
