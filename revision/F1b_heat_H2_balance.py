# -*- coding: utf-8 -*-
"""
F1b: Heat-and-hydrogen balance on top of F1 (Stage 1 fired with surplus H2)
===========================================================================
FUPROC-D-26-00269 Major Revision. Reads F1_cases.csv (F1_full_biogas_CO2.py) and adds, for
every case and for three heat-recovery levels, the hydrogen that must be burned to supply the
high-temperature heat demand, and the hydrogen that is still exportable afterwards.

Assumptions (interim until T4 fixes the heat cascade)
  - Stage 1 (1200 K) duty Q1 is supplied by burning system hydrogen in a furnace with 85 %
    efficiency on the lower heating value (LHV H2 from GRI-3.0 at 298 K, = 119.96 MJ/kg).
  - Stage 2 (950 K) is endothermic in the H2-lean configurations (Q2 > 0); that duty must also be
    supplied at high temperature. Two demand definitions are reported:
        "Q1 only"  : demand = Q1                      (as specified in the task)
        "Q1 + Q2+" : demand = Q1 + max(Q2, 0)         (stricter, recommended)
  - The exotherms of Stage 3 (650 K) and of Stage 2 when Q2 < 0 cannot be cascaded to 1200 K;
    a fraction f in {0, 0.25, 0.50} of the exothermic pool -(min(Q2,0) + min(Q3,0)) is credited
    as feed preheat, reducing the fired duty:   Q_fired = max(0, demand - f * pool) / 0.85.
  - H2 burned = Q_fired / LHV; H2 net after heat = H2 exportable (F1) - H2 burned.
  - Self-sufficiency test for configuration 2b (no electrolysis): H2 net after heat >= 0.
    Auxiliary electricity (compression, separation, P3) is NOT included here.
  - Stoichiometric reference: overall CH4 + rho CO2 -> (1+rho) C + (2-2rho) H2 + 2rho H2O with
    rho = (1-y)/y; ideal surplus H2 and standard reaction enthalpy at 298 K (GRI-3.0 + graphite),
    and the ideal furnace demand if the whole process ran at its minimum enthalpy.

All values equilibrium/kinetic exactly as in F1 (column value_type); the added quantities are
derived, not simulated.

Reproducibility:  cd <repo>/revision ; ../.venv/bin/python F1b_heat_H2_balance.py   (seconds)
Outputs: F1b_heat_H2_balance.csv, F1b_stoichiometric_reference.csv, F1b_summary.txt
"""
import os, sys
import numpy as np, pandas as pd
import cantera as ct

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W

IN = os.path.join(RESULT, "F1_cases.csv")
OUT = os.path.join(RESULT, "F1b_heat_H2_balance.csv"); OUT_ST = os.path.join(RESULT, "F1b_stoichiometric_reference.csv"); OUT_TXT = os.path.join(RESULT, "F1b_summary.txt")
ETA_FURNACE = 0.85; F_GRID = [0.0, 0.25, 0.50]

g = ct.Solution("gri30.yaml"); g.TP = 298.15, ct.one_atm
Hf = {sp: g.standard_enthalpies_RT[g.species_index(sp)] * ct.gas_constant * 298.15 / 1e6 for sp in ["CH4", "CO2", "H2", "H2O", "O2"]}   # kJ/mol
LHV_H2_kJ_mol = -(Hf["H2O"] - Hf["H2"] - 0.5 * Hf["O2"])          # 241.8 kJ/mol
LHV_H2_kWh_kg = LHV_H2_kJ_mol / W.MW["H2"] / 3.6                    # 33.3 kWh/kg
def kW_to_H2_tpd(kW): return kW * 24.0 / LHV_H2_kWh_kg / 1000.0

df = pd.read_csv(IN)
rows = []
for _, r in df.iterrows():
    if not bool(r.converged): continue
    Q1, Q2, Q3 = r.Q1_kW, r.Q2_kW, r.Q3_kW
    pool = -(min(Q2, 0.0) + min(Q3, 0.0))
    for f in F_GRID:
        for dem_label, demand in [("Q1 only", Q1), ("Q1 + Q2+", Q1 + max(Q2, 0.0))]:
            q_fired = max(0.0, demand - f * pool) / ETA_FURNACE
            h2_burn = kW_to_H2_tpd(q_fired)
            h2_net = r.H2_net_exportable_tpd - h2_burn
            rows.append(dict(case=r.case, config=r.config, y_CH4_biogas=r.y_CH4_biogas, H2_electrolysis_tpd=r.H2_electrolysis_tpd, value_type=r.value_type,
                             mode=r["mode"], phi=r.phi, P2_bar=r.P2_bar, pressure_order=r.pressure_order, recycle=r.recycle, h_H2_to_CFR=r.h_H2_to_CFR,
                             heat_recovery_fraction=f, demand_definition=dem_label,
                             Q1_kW=Q1, Q2_kW=Q2, Q3_kW=Q3, exothermic_pool_kW=pool, high_T_demand_kW=demand, preheat_credit_kW=f * pool,
                             Q_fired_kW=q_fired, H2_burned_tpd=h2_burn, H2_exportable_before_heat_tpd=r.H2_net_exportable_tpd,
                             H2_net_after_heat_tpd=h2_net, self_sufficient_H2=(h2_net >= 0.0),
                             H2_burned_share_of_exportable=(h2_burn / r.H2_net_exportable_tpd if r.H2_net_exportable_tpd > 0 else np.inf),
                             C_total_tpd=r.C_total_tpd, CO2_carbon_fixed_fraction=r.CO2_carbon_fixed_fraction, CO2_fixed_as_CO2_tpd=r.CO2_fixed_as_CO2_tpd))
out = pd.DataFrame(rows); out.to_csv(OUT, index=False)

# ---------------- stoichiometric reference
st = []
n_ch4 = W.tpd_to_kmol_per_day(W.CH4_tpd, W.MW["CH4"])
for y in [0.55, 0.60, 0.65]:
    rho = (1 - y) / y; co2 = n_ch4 * rho
    h2_surplus = n_ch4 * (2 - 2 * rho)                           # kmol/d, X_CH4 = 1, all CO2 fixed
    dH = (2 * rho * Hf["H2O"]) - Hf["CH4"] - rho * Hf["CO2"]      # kJ per mol CH4 (C(gr) = 0)
    q_net_kW = dH * n_ch4 * 1000 / 86400.0                         # + endothermic
    h2_energy_kW = h2_surplus * LHV_H2_kJ_mol * 1000 / 86400.0
    st.append(dict(y_CH4_biogas=y, rho_CO2_per_CH4=rho, CO2_tpd=W.kmol_per_day_to_tpd(co2, W.MW["CO2"]),
                   overall_reaction=f"CH4 + {rho:.3f} CO2 -> {1+rho:.3f} C + {2-2*rho:.3f} H2 + {2*rho:.3f} H2O",
                   solid_C_ideal_tpd=W.kmol_per_day_to_tpd(n_ch4 * (1 + rho), W.MW_C_SOLID),
                   ideal_surplus_H2_tpd=W.kmol_per_day_to_tpd(h2_surplus, W.MW["H2"]),
                   ideal_surplus_H2_LHV_kW=h2_energy_kW,
                   dH_overall_298K_kJ_per_mol_CH4=dH, net_heat_demand_298K_kW=q_net_kW,
                   ideal_fired_H2_for_net_heat_tpd=kW_to_H2_tpd(max(q_net_kW, 0) / ETA_FURNACE),
                   ideal_H2_after_net_heat_tpd=W.kmol_per_day_to_tpd(h2_surplus, W.MW["H2"]) - kW_to_H2_tpd(max(q_net_kW, 0) / ETA_FURNACE),
                   water_ideal_tpd=W.kmol_per_day_to_tpd(2 * rho * n_ch4, W.MW["H2O"])))
st = pd.DataFrame(st); st.to_csv(OUT_ST, index=False)

# ---------------- summary text
L = [f"F1b heat/H2 balance. LHV H2 = {LHV_H2_kJ_mol:.1f} kJ/mol = {LHV_H2_kWh_kg:.2f} kWh/kg; furnace efficiency {ETA_FURNACE}."]
L.append("\n--- Stoichiometric reference (X_CH4 = 1, all CO2 fixed, 298 K) ---\n" + st.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
sel = out[(out["mode"] == "equilibrium") & out.recycle & (out.demand_definition == "Q1 + Q2+")]
piv = sel.pivot_table(index=["config", "h_H2_to_CFR"], columns="heat_recovery_fraction", values="H2_net_after_heat_tpd")
L.append("\n--- EQUILIBRIUM recycle: H2 net after heat [t/d], demand = Q1 + Q2(+), by heat-recovery fraction ---\n" + piv.to_string(float_format=lambda v: f"{v:.2f}"))
sel1 = out[(out["mode"] == "equilibrium") & out.recycle & (out.demand_definition == "Q1 only")]
piv1 = sel1.pivot_table(index=["config", "h_H2_to_CFR"], columns="heat_recovery_fraction", values="H2_net_after_heat_tpd")
L.append("\n--- EQUILIBRIUM recycle: H2 net after heat [t/d], demand = Q1 only ---\n" + piv1.to_string(float_format=lambda v: f"{v:.2f}"))
selk = out[(out["mode"] == "kinetic") & out.recycle & (out.demand_definition == "Q1 + Q2+") & (out.phi == 1.0)]
pivk = selk.pivot_table(index=["config", "P2_bar"], columns="heat_recovery_fraction", values="H2_net_after_heat_tpd")
L.append("\n--- KINETIC recycle (phi = 1, h = 0.35): H2 net after heat [t/d], demand = Q1 + Q2(+) ---\n" + pivk.to_string(float_format=lambda v: f"{v:.2f}"))
with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
print("\n".join(L)); print("Saved:", OUT, OUT_ST, OUT_TXT)
