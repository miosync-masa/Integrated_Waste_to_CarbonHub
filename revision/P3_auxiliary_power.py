# -*- coding: utf-8 -*-
"""
P3: Auxiliary power of the X4 configuration and final hydrogen self-sufficiency (R1-5, R2-5)
=============================================================================================
FUPROC-D-26-00269 Major Revision. Builds on F1 (X4 flowsheet), F1b (heat/H2 balance) and T3
(compression model). X4 = configuration 2b (all biogas CO2, no DAC, no electrolysis), recycle
r_CH4 = 0.95, p = 0.05, all stages at 1 atm; representative y = 0.60, h = 0.50; sensitivity
y in {0.55, 0.60, 0.65}, h in {0.35, 0.50, 0.75}; the submitted configuration (h = 0.35) is
carried along for comparison. EQUILIBRIUM recycle values (F1 flowsheet re-solved here to obtain
every stream).

Auxiliary items (assumptions and sources)
  A  Biogas upgrading (CH4/CO2 separation upstream of Stage 1/2). Raw biogas = 66,964 Nm3/d CH4 / y.
     Specific electricity e_up = 0.25 kWh/Nm3 raw biogas (base), range 0.20-0.45:
       - Bauer, Hulteberg, Persson, Tamm, "Biogas upgrading - Review of commercial technologies",
         SGC Rapport 2013:270: membranes 0.20-0.30 kWh/Nm3 raw biogas (manufacturer guarantee, p.34),
         PSA 0.15-0.30 (p.28, 0.2-0.3 p.52), water scrubbing 0.2-0.3 (p.43-45), amine 0.12-0.14 electric
         + 0.55 heat (p.21); methane slip PSA 1.8-2 %, water 1 %, membranes ~0.5 %, amine 0.1 % (p.53).
         http://vav.griffel.net/filer/c_sgc2013-270.pdf (copy: P3_source_SGC2013-270_biogas_upgrading.pdf)
       - Kamusoko & Mukumba, Bioengineering 13(5):543 (2026), Table 3: membranes 0.25-0.43, PSA 0.45,
         water scrubbing 0.45, cryogenic 0.51 kWh/Nm3. doi:10.3390/bioengineering13050543
     The submitted paper placed AD and upgrading OUTSIDE the boundary; results are given both
     excluding (boundary as submitted) and including this item.
  B  H2 membrane between Stage 2 and 3 (split h). Driving force options, both computed:
       B1 feed compression of the whole post-condensation Stage 2 stream 1 atm -> 4 bar
          (multistage, intercooled 313 K, eta_s 0.75; T3 model);
       B2 permeate vacuum: permeate H2 drawn at 0.25 bar and recompressed to 1 atm (eta_s 0.75).
     Base = the cheaper option (B2 in all cases). Assumption: polymeric membrane, H2-selective.
  C  CH4 separation at the Stage 3 outlet (recovery 0.95): CH4/CO2/CO/H2 separation treated like
     biogas upgrading, e_sep = 0.25 kWh/Nm3 of feed gas (assumption, same sources as A); the
     alternative "compress feed to 4 bar" is also reported.
  D  Recycle blower: recycle streams (to Stage 2 and CH4 to Stage 1) and fresh biogas feeds,
     pressure rise 0.3 bar (assumed total loop pressure drop), eta_s 0.70, single stage.
  E  Condenser cooling (air-cooled): duty = latent heat of condensed water (2,257 kJ/kg) + sensible
     cooling of the Stage 2 outlet (950 -> 313 K) and Stage 3 outlet (650 -> 313 K, Cantera enthalpies);
     fan power = 1.5 % of duty (assumption; air coolers typically 1-2 % of duty).
  F  Solid carbon handling (lock hoppers, conveyors, cooling): 10 kWh per t C (assumption, 5-20).
  G  Other (instrument air, controls, water pumps): 5 % of A-F subtotal (assumption).
Power supply: (i) surplus H2 to electricity at eta_e = 0.40 (H2 engine) or 0.50 (fuel cell), H2 LHV
33.3 kWh/kg; (ii) grid electricity priced as in the TEA (0.04 / 0.06 / 0.08 USD/kWh, 330 d).
Final balance: H2 net after heat (F1b logic, demand Q1 + Q2+, furnace 0.85, recovery f) minus H2
for auxiliaries. Required heat-recovery fraction for zero net H2 is solved analytically.

Note on T3 section 4: the split-pressure compression (0.74-1.05 MW) is NOT part of X4 (all stages
at 1 atm); the only compression here is the membrane driving force (B) and the loop blower (D).

Reproducibility:  cd <repo>/revision ; ../.venv/bin/python P3_auxiliary_power.py   (~3-4 min, 10 processes)
Outputs: P3_aux_power_items.csv, P3_selfsufficiency_map.csv, P3_summary.txt
"""
import os, sys, time
import numpy as np, pandas as pd
import cantera as ct
from multiprocessing import Pool, cpu_count

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import F1_full_biogas_CO2 as F1m
import T3_pressure_analysis as T3m
from P1_recycle_analysis import FRESH_CH4, tpd, carbon, add

OUT_ITEMS = os.path.join(RESULT, "P3_aux_power_items.csv"); OUT_MAP = os.path.join(RESULT, "P3_selfsufficiency_map.csv"); OUT_TXT = os.path.join(RESULT, "P3_summary.txt")
ATM = ct.one_atm; BAR = 1e5
E_UP = {"low": 0.20, "base": 0.25, "high": 0.45}      # kWh/Nm3 raw biogas
E_SEP_CH4 = 0.25                                      # kWh/Nm3 feed gas at Stage 3 outlet
DP_LOOP_BAR = 0.3; ETA_BLOWER = 0.70
FAN_FRACTION = 0.015; SOLIDS_KWH_T = 10.0; OTHER_FRAC = 0.05
ETA_FURNACE = 0.85; LHV_KWH_KG = 33.32; ETA_E = {"H2 engine 0.40": 0.40, "fuel cell 0.50": 0.50}
F_GRID = [0.0, 0.25, 0.50]
NM3_PER_KMOL = 22.414
CH4_NM3_D = FRESH_CH4 * NM3_PER_KMOL     # 66,964 Nm3/d (at 0 C, 1 atm)

def stream_enthalpy_kW(gas, T):
    return W.total_stream_enthalpy_J_per_day(gas, 0, T, ATM) / 86400e3

def aux_items(rec, cfg):
    fd = cfg["feed"]; y = fd["y"]; items = []
    def it(code, name, kW, basis, source): items.append(dict(item=code, name=name, kW=kW, basis=basis, source=source))
    # A biogas upgrading
    raw = CH4_NM3_D / y if y == y else CH4_NM3_D / 0.60   # submitted config: assume y = 0.60 for the raw-biogas volume
    for k, e in E_UP.items():
        it(f"A_upgrading_{k}", f"biogas CH4/CO2 separation ({k}: {e} kWh/Nm3)", raw * e / 24.0, f"raw biogas {raw:,.0f} Nm3/d", "SGC 2013:270 p.28/34/52; Kamusoko & Mukumba 2026 Tab.3")
    # B H2 membrane
    pw2 = rec["pw2"]; perm = rec["permeate"]
    b1, _, _ = T3m.compress_kW(pw2, ATM, 4 * BAR); b2, _, _ = T3m.compress_kW(perm, 0.25 * BAR, ATM)
    it("B1_H2membrane_feed_compression", "H2 membrane, feed 1 atm -> 4 bar (whole Stage 2 stream)", b1, f"feed {sum(pw2.values()):,.0f} kmol/d", "T3 compression model; assumption 4 bar feed")
    it("B2_H2membrane_permeate_vacuum", "H2 membrane, permeate vacuum 0.25 bar -> 1 atm", b2, f"permeate {sum(perm.values()):,.0f} kmol/d", "assumption 0.25 bar permeate")
    it("B_H2membrane_base", "H2 membrane driving force (cheaper of B1/B2)", min(b1, b2), "B2" if b2 <= b1 else "B1", "")
    # C CH4 separation at Stage 3 outlet
    pw3 = rec["pw3"]; n3 = sum(pw3.values())
    c_spec = n3 * NM3_PER_KMOL * E_SEP_CH4 / 24.0; c_comp, _, _ = T3m.compress_kW(pw3, ATM, 4 * BAR)
    it("C_CH4separation_base", f"CH4 separation at Stage 3 outlet ({E_SEP_CH4} kWh/Nm3 feed)", c_spec, f"feed {n3:,.0f} kmol/d = {n3*NM3_PER_KMOL:,.0f} Nm3/d", "assumption, same sources as A")
    it("C_alt_feed_compression_4bar", "CH4 separation alt.: feed 1 atm -> 4 bar", c_comp, "", "T3 compression model")
    # D blowers
    loop = add(rec["to_s2"], {"CH4": rec["ch4_rec"]}); fresh = {"CH4": FRESH_CH4, "CO2": fd["co2_bio"] + fd["co2_dac"]}
    d1, _, _ = T3m.compress_kW(loop, ATM, ATM + DP_LOOP_BAR * BAR, eta_s=ETA_BLOWER, max_ratio=10.0)
    d2, _, _ = T3m.compress_kW(fresh, ATM, ATM + DP_LOOP_BAR * BAR, eta_s=ETA_BLOWER, max_ratio=10.0)
    it("D_recycle_blower", f"recycle blower, dP {DP_LOOP_BAR} bar", d1, f"{sum(loop.values()):,.0f} kmol/d", "assumption dP, eta 0.70")
    it("D_feed_blower", f"fresh biogas/CO2 blower, dP {DP_LOOP_BAR} bar", d2, f"{sum(fresh.values()):,.0f} kmol/d", "assumption")
    # E condenser fans
    lat = (rec["water2"].get("H2O", 0) + rec["water3"].get("H2O", 0)) * W.MW["H2O"] * 2257.0 / 86400.0   # kW (kmol/d * kg/kmol * kJ/kg / s)
    sens2 = stream_enthalpy_kW(rec["gas2"], W.T_rwgs) - stream_enthalpy_kW(rec["gas2"], 313.15)
    sens3 = stream_enthalpy_kW(rec["gas3"], W.T_cfr) - stream_enthalpy_kW(rec["gas3"], 313.15)
    duty = lat + sens2 + sens3
    it("E_condenser_fans", f"air-cooled condensers, {FAN_FRACTION*100:.1f} % of {duty:,.0f} kW duty", duty * FAN_FRACTION, f"latent {lat:,.0f} + sensible {sens2+sens3:,.0f} kW", "assumption 1-2 % of duty")
    # F solids
    C_tpd = tpd("C(s)", rec["C1"] + rec["C3"])
    it("F_solids_handling", f"solid carbon handling, {SOLIDS_KWH_T} kWh/t", C_tpd * SOLIDS_KWH_T / 24.0, f"{C_tpd:.1f} t/d", "assumption 5-20 kWh/t")
    df = pd.DataFrame(items)
    base_codes = ["B_H2membrane_base", "C_CH4separation_base", "D_recycle_blower", "D_feed_blower", "E_condenser_fans", "F_solids_handling"]
    sub = df[df.item.isin(base_codes)].kW.sum()
    it("G_other", f"other ({OTHER_FRAC*100:.0f} % of B-F)", sub * OTHER_FRAC, "", "assumption")
    df = pd.DataFrame(items)
    tot_excl = sub * (1 + OTHER_FRAC)
    tot_incl = tot_excl + float(df[df.item == "A_upgrading_base"].kW.iloc[0])
    return df, dict(duty_cooling_kW=duty, aux_excl_upgrading_kW=tot_excl, aux_incl_upgrading_kW=tot_incl,
                    aux_incl_upgrading_low_kW=tot_excl + float(df[df.item == "A_upgrading_low"].kW.iloc[0]),
                    aux_incl_upgrading_high_kW=tot_excl + float(df[df.item == "A_upgrading_high"].kW.iloc[0]))

def kW_to_H2_tpd_elec(kW, eta_e): return kW * 24.0 / (LHV_KWH_KG * eta_e) / 1000.0

def run_case(args):
    label, cfg = args; t0 = time.time()
    rec, info = F1m.solve_recycle(cfg)
    row = F1m.make_row(cfg, rec, info, label)
    items, tot = aux_items(rec, cfg)
    items.insert(0, "case", label); items.insert(1, "config", cfg["config"]); items.insert(2, "h_H2_to_CFR", cfg["h"])
    row.update(tot); row["wall_s"] = time.time() - t0
    return row, items

def main():
    t0 = time.time(); F = F1m.feeds()
    cases = []
    for cname in ["2b_y0.55", "2b_y0.60", "2b_y0.65"]:
        for h in [0.35, 0.50, 0.75]:
            cases.append((f"{cname} eq recycle h={h:g}", dict(config=cname, feed=F[cname], mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1",
                                                              P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=h, p=0.05, r=0.95)))
    cases.append(("submitted eq recycle h=0.35", dict(config="submitted", feed=F["submitted"], mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1",
                                                       P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.35, p=0.05, r=0.95)))
    with Pool(min(len(cases), cpu_count())) as pool: res = pool.map(run_case, cases)
    rows = [r for r, _ in res]; items = pd.concat([i for _, i in res], ignore_index=True)
    items.to_csv(OUT_ITEMS, index=False)
    # self-sufficiency map
    maps = []
    for r in rows:
        Q1, Q2, Q3 = r["Q1_kW"], r["Q2_kW"], r["Q3_kW"]; pool_ = -(min(Q2, 0) + min(Q3, 0)); demand = Q1 + max(Q2, 0)
        for f in F_GRID:
            h2_burn = max(0.0, demand - f * pool_) / ETA_FURNACE * 24 / LHV_KWH_KG / 1000.0
            h2_after_heat = r["H2_net_exportable_tpd"] - h2_burn
            for bnd, aux_kW in [("excl. upgrading (submitted boundary)", r["aux_excl_upgrading_kW"]), ("incl. upgrading (base 0.25 kWh/Nm3)", r["aux_incl_upgrading_kW"]),
                                ("incl. upgrading (high 0.45)", r["aux_incl_upgrading_high_kW"])]:
                for ename, eta_e in ETA_E.items():
                    h2_aux = kW_to_H2_tpd_elec(aux_kW, eta_e); net = h2_after_heat - h2_aux
                    E_exp_kW = (r["H2_net_exportable_tpd"] - h2_aux) * 1000 * LHV_KWH_KG / 24.0
                    f_req = (demand - ETA_FURNACE * E_exp_kW) / pool_ if pool_ > 0 else np.inf
                    maps.append(dict(case=r["case"], config=r["config"], y_CH4_biogas=r["y_CH4_biogas"], h_H2_to_CFR=r["h_H2_to_CFR"],
                                     CO2_carbon_fixed_fraction=r["CO2_carbon_fixed_fraction"], C_total_tpd=r["C_total_tpd"],
                                     heat_recovery_fraction=f, boundary=bnd, power_source=ename, aux_power_kW=aux_kW,
                                     H2_exportable_tpd=r["H2_net_exportable_tpd"], H2_burned_for_heat_tpd=h2_burn, H2_after_heat_tpd=h2_after_heat,
                                     H2_for_auxiliaries_tpd=h2_aux, H2_net_final_tpd=net, self_sufficient=(net >= 0),
                                     heat_recovery_required_for_zero_net=max(0.0, f_req),
                                     grid_cost_usd_y_at_0p06=aux_kW * 24 * 330 * 0.06, grid_cost_usd_y_at_0p04=aux_kW * 24 * 330 * 0.04, grid_cost_usd_y_at_0p08=aux_kW * 24 * 330 * 0.08))
    mp = pd.DataFrame(maps); mp.to_csv(OUT_MAP, index=False)
    L = [f"P3 auxiliary power (wall {time.time()-t0:.0f} s). EQUILIBRIUM recycle, 1 atm."]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 400, "display.float_format", lambda v: f"{v:,.1f}"):
        rep = items[items.case == "2b_y0.60 eq recycle h=0.5"][["item", "name", "kW", "basis", "source"]]
        L.append("\n--- Items, representative X4 case (2b, y = 0.60, h = 0.50) ---\n" + rep.to_string(index=False))
        tot = pd.DataFrame(rows)[["case", "aux_excl_upgrading_kW", "aux_incl_upgrading_kW", "aux_incl_upgrading_high_kW", "duty_cooling_kW", "H2_net_exportable_tpd", "Q1_kW", "Q2_kW", "Q3_kW", "C_total_tpd", "CO2_carbon_fixed_fraction"]]
        L.append("\n--- Totals per case ---\n" + tot.to_string(index=False))
    for bnd in mp.boundary.unique():
        for ename in ETA_E:
            sel = mp[(mp.boundary == bnd) & (mp.power_source == ename) & (mp.config != "submitted")]
            piv = sel.pivot_table(index=["y_CH4_biogas", "h_H2_to_CFR"], columns="heat_recovery_fraction", values="H2_net_final_tpd")
            L.append(f"\n--- Final net H2 [t/d] after heat (Q1+Q2+, furnace 0.85) and auxiliaries | {bnd} | {ename} ---\n" + piv.to_string(float_format=lambda v: f"{v:+.2f}"))
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", OUT_ITEMS, OUT_MAP, OUT_TXT)

if __name__ == "__main__":
    main()
