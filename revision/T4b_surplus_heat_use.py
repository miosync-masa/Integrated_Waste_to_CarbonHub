# -*- coding: utf-8 -*-
"""
T4b: Use of the 650 K surplus heat — power vs amine regeneration (follow-up to T4)
==================================================================================
FUPROC-D-26-00269 Major Revision. Re-solves the X4 cases (2b, h = 0.50, y in {0.55, 0.60, 0.65})
at recycle equilibrium with the F1 flowsheet, rebuilds the T4 pinch cascade, and compares three
uses of the heat that lies below the pinch (the 650 K Stage 3 exotherm and the product cooling):

  (i)   all surplus heat above the power-cycle source temperature -> bottoming power, eta_el 0.20 / 0.25
        (T4 convention: source = Stage 3 exotherm plateau, min(-Q3, Q_C,min));
  (ii)  biogas upgrading switched from membranes to amine scrubbing whose reboiler heat is taken from
        the cascade; electricity of the amine unit from H2 power; no bottoming power;
  (iii) as (ii) plus bottoming power from the heat that remains above the reboiler temperature.

Amine data (SGC Rapport 2013:270, Bauer et al., p.21-22): electricity 0.12-0.14 kWh/Nm3 raw biogas
(0.13 used), reboiler heat ~0.55 kWh/Nm3 raw biogas, reboiler operable down to 90 C with a vacuum
option (+0.05 kWh/Nm3 electricity); standard reboiler temperature taken as 120 C (base) and 160 C
(conservative). Heat available to the reboiler is read from the grand composite curve at the shifted
reboiler temperature (T_reb + dTmin/2); the 650 K plateau supplies it with a driving force far above
dTmin = 20 K. Membrane upgrading (P3): 0.25 kWh/Nm3 electricity, no heat.
Bottoming cycle efficiency 0.20-0.25 is an assumption for a ~650 K source (Carnot 0.52 to 313 K, i.e.
40-50 % second-law efficiency); see references_T4b_additions.bib for ORC/steam reviews.

Boundary: biogas upgrading INSIDE the boundary in all three options (that is the question); the
"excl. upgrading" case of T4 is repeated for reference. Auxiliaries other than upgrading from P3
(P3_selfsufficiency_map.csv, 'excl. upgrading' rows). H2 accounting as in T4: exportable H2 (F1)
minus H2 fired for Q_H,min/0.85 minus H2 for residual electricity at eta_e 0.50 (fuel cell) / 0.40
(H2 engine). EQUILIBRIUM values.

Reproducibility: cd <repo>/revision ; ../.venv/bin/python T4b_surplus_heat_use.py  (~1 min)
Outputs: T4b_surplus_heat_use.csv, T4b_summary.txt
"""
import os, sys, time
import numpy as np, pandas as pd
import cantera as ct
from multiprocessing import Pool

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import F1_full_biogas_CO2 as F1m
import T4_heat_cascade as T4m
from P1_recycle_analysis import FRESH_CH4

ATM = ct.one_atm; DT = 20.0; ETA_FURNACE = 0.85; LHV = 33.32
ETA_E = {"fuel cell 0.50": 0.50, "H2 engine 0.40": 0.40}; ETA_B = [0.20, 0.25]
E_MEMBRANE = 0.25; E_AMINE_EL = 0.13; Q_AMINE_HEAT = 0.55           # kWh/Nm3 raw biogas
T_REB = {"120 C (base)": 393.15, "160 C (conservative)": 433.15, "90 C (vacuum option, +0.05 kWh/Nm3)": 363.15}
NM3_PER_KMOL = 22.414
OUT = os.path.join(RESULT, "T4b_surplus_heat_use.csv"); OUT_TXT = os.path.join(RESULT, "T4b_summary.txt")
P3 = pd.read_csv(os.path.join(RESULT, "P3_selfsufficiency_map.csv")); P3 = P3[(P3.heat_recovery_fraction == 0.0) & P3.boundary.str.startswith("excl")]

def gcc_at(p, T_shift):
    """Net heat cascading through shifted temperature T_shift [kW] (from the grand composite curve)."""
    T = p["gcc_T"][::-1]; H = p["gcc_H"][::-1]        # ascending T
    return float(np.interp(T_shift, T, H))

def run(args):
    label, cfg = args
    rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label)
    S = T4m.streams_from_rec(rec, cfg); p = T4m.pinch(S, DT)
    return dict(label=label, row=row, pinch=p, Q=(rec["Q1"], rec["Q2"], rec["Q3"]))

def main():
    t0 = time.time(); F = F1m.feeds(); cases = []
    for cname in ["2b_y0.55", "2b_y0.60", "2b_y0.65"]:
        cases.append((f"{cname} eq recycle h=0.5", dict(config=cname, feed=F[cname], mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1",
                                                        P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.5, p=0.05, r=0.95)))
    with Pool(3) as pool: res = pool.map(run, cases)
    rows = []
    for r in res:
        y = r["row"]["y_CH4_biogas"]; raw = FRESH_CH4 * NM3_PER_KMOL / y            # Nm3/d raw biogas
        p = r["pinch"]; QH = p["Q_H_min_kW"]; QC = p["Q_C_min_kW"]; Q3 = r["Q"][2]
        h2_exp = r["row"]["H2_net_exportable_tpd"]; h2_fired = QH / ETA_FURNACE * 24 / LHV / 1000.0; h2_heat = h2_exp - h2_fired
        aux_excl = float(P3[(P3.case == r["label"]) & (P3.power_source == "fuel cell 0.50")].aux_power_kW.iloc[0])
        q_plateau = min(-Q3, QC)
        up_membrane_kW = raw * E_MEMBRANE / 24.0; up_amine_el_kW = raw * E_AMINE_EL / 24.0; up_amine_heat_kW = raw * Q_AMINE_HEAT / 24.0
        base = dict(case=r["label"], y_CH4_biogas=y, raw_biogas_Nm3_d=raw, Q_H_min_kW=QH, Q_C_min_kW=QC, Q3_kW=Q3, plateau_650K_kW=q_plateau,
                    H2_exportable_tpd=h2_exp, H2_fired_tpd=h2_fired, H2_after_heat_tpd=h2_heat, aux_excl_upgrading_kW=aux_excl,
                    upgrading_membrane_el_kW=up_membrane_kW, upgrading_amine_el_kW=up_amine_el_kW, upgrading_amine_heat_kW=up_amine_heat_kW,
                    CO2_carbon_fixed_fraction=r["row"]["CO2_carbon_fixed_fraction"], C_total_tpd=r["row"]["C_total_tpd"])
        for ename, eta in ETA_E.items():
            def h2_el(kW): return kW * 24 / (LHV * eta) / 1000.0
            # reference: excl. upgrading, no bottoming / with bottoming
            for eb in [None] + ETA_B:
                P_b = 0.0 if eb is None else eb * q_plateau
                rows.append(dict(base, option="ref: upgrading outside boundary" + ("" if eb is None else f", bottoming eta {eb:.2f}"), power_source=ename,
                                 reboiler=None, heat_to_amine_kW=0.0, heat_available_at_reboiler_kW=np.nan, heat_to_power_kW=q_plateau if eb else 0.0, bottoming_power_kW=P_b,
                                 electricity_demand_kW=aux_excl, residual_electricity_kW=max(0.0, aux_excl - P_b), H2_net_final_tpd=h2_heat - h2_el(max(0.0, aux_excl - P_b))))
            # (i) membrane upgrading inside boundary, all plateau heat to power
            for eb in [None] + ETA_B:
                P_b = 0.0 if eb is None else eb * q_plateau; dem = aux_excl + up_membrane_kW
                rows.append(dict(base, option="(i) membrane upgrading" + (" , no bottoming" if eb is None else f", all plateau to power eta {eb:.2f}"), power_source=ename,
                                 reboiler=None, heat_to_amine_kW=0.0, heat_available_at_reboiler_kW=np.nan, heat_to_power_kW=q_plateau if eb else 0.0, bottoming_power_kW=P_b,
                                 electricity_demand_kW=dem, residual_electricity_kW=max(0.0, dem - P_b), H2_net_final_tpd=h2_heat - h2_el(max(0.0, dem - P_b))))
            # (ii)/(iii) amine upgrading with reboiler heat from the cascade
            for rlab, Treb in T_REB.items():
                avail = gcc_at(p, Treb + DT / 2)                      # heat cascading through the shifted reboiler temperature
                q_am = min(up_amine_heat_kW, avail); short = up_amine_heat_kW - q_am
                el_am = up_amine_el_kW + (raw * 0.05 / 24.0 if Treb < 370 else 0.0)
                dem = aux_excl + el_am
                # (ii) no power
                rows.append(dict(base, option=f"(ii) amine upgrading, reboiler {rlab}, no bottoming", power_source=ename, reboiler=rlab, heat_to_amine_kW=q_am,
                                 heat_available_at_reboiler_kW=avail, heat_shortfall_kW=short, heat_to_power_kW=0.0, bottoming_power_kW=0.0,
                                 electricity_demand_kW=dem, residual_electricity_kW=dem, H2_net_final_tpd=h2_heat - h2_el(dem) - short / ETA_FURNACE * 24 / LHV / 1000.0))
                # (iii) power from the heat remaining above the reboiler level (plateau minus what the amine takes from it)
                q_pow = max(0.0, min(q_plateau, avail - q_am))
                for eb in ETA_B:
                    P_b = eb * q_pow
                    rows.append(dict(base, option=f"(iii) amine upgrading, reboiler {rlab}, bottoming eta {eb:.2f} on remaining heat", power_source=ename, reboiler=rlab,
                                     heat_to_amine_kW=q_am, heat_available_at_reboiler_kW=avail, heat_shortfall_kW=short, heat_to_power_kW=q_pow, bottoming_power_kW=P_b,
                                     electricity_demand_kW=dem, residual_electricity_kW=max(0.0, dem - P_b),
                                     H2_net_final_tpd=h2_heat - h2_el(max(0.0, dem - P_b)) - short / ETA_FURNACE * 24 / LHV / 1000.0))
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    L = [f"T4b surplus-heat use (wall {time.time()-t0:.0f} s). EQUILIBRIUM. dTmin {DT} K."]
    with pd.option_context("display.width", 320, "display.max_columns", 40, "display.max_rows", 400, "display.float_format", lambda v: f"{v:,.2f}"):
        for lab in df.case.unique():
            d = df[(df.case == lab) & (df.power_source == "fuel cell 0.50")]
            L.append(f"\n=== {lab}: raw biogas {d.raw_biogas_Nm3_d.iloc[0]:,.0f} Nm3/d; Q_H,min {d.Q_H_min_kW.iloc[0]:,.0f} kW; plateau {d.plateau_650K_kW.iloc[0]:,.0f} kW; H2 after heat {d.H2_after_heat_tpd.iloc[0]:.2f} t/d; aux excl. {d.aux_excl_upgrading_kW.iloc[0]:,.0f} kW; membrane {d.upgrading_membrane_el_kW.iloc[0]:,.0f} kW el; amine {d.upgrading_amine_el_kW.iloc[0]:,.0f} kW el + {d.upgrading_amine_heat_kW.iloc[0]:,.0f} kW heat")
            L.append(d[["option", "heat_available_at_reboiler_kW", "heat_to_amine_kW", "heat_to_power_kW", "bottoming_power_kW", "electricity_demand_kW", "residual_electricity_kW", "H2_net_final_tpd"]].to_string(index=False))
        piv = df.pivot_table(index="option", columns=["y_CH4_biogas", "power_source"], values="H2_net_final_tpd")
        L.append("\n=== Final net H2 [t/d] by option x (y, power source) ===\n" + piv.to_string(float_format=lambda v: f"{v:+.2f}"))
    with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", OUT, OUT_TXT)

if __name__ == "__main__":
    main()
