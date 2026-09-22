# -*- coding: utf-8 -*-
"""
E9: Gate-fee basis in the screening TEA (reviewers R1-6, R2-6)
================================================================
FUPROC-D-26-00269 Major Revision.

Inconsistency: the process is introduced with 800 t/d of organic waste (-> 48 t/d CH4 and
32 t/d biogas CO2 via anaerobic digestion), but validation/screening_tea.py charges the gate
fee on waste_tpd = 100.0 t/d in all three scenarios (lines 105/120/135; the field is commented
"Throughput anchor for gate fee"; no derivation is given in the code). Scaling to 800 t/d
multiplies the gate revenue by 8, so the chargeable basis and the fee level must be made
explicit rather than silently changed.

This script
  1. reproduces the submitted TEA numbers with a re-implementation of its annual-value formula
     (exact same inputs), to make sure the E9 scenarios are computed on the same engine;
  2. checks the feedstock boundary 800 t/d -> 48 t/d CH4 + 32 t/d CO2 for internal consistency
     (implied biogas yield and CH4 fraction);
  3. evaluates gate-fee scenarios: chargeable basis x fee level, for all three submitted
     scenarios and both paths, and reports annual gate revenue, annual value and simple payback;
  4. finds break-even fees and identifies which items carry the Base case when the fee is zero.

Fee-level evidence (East Africa; all URLs recorded, copies of the PDFs kept as E9_source_*):
  - African Clean Cities Platform (JICA / UNEP / UN-Habitat), Nairobi city profile (2022):
    "Tipping fee of KES 100/ton is charged"; "Amount to be spent per ton of waste: KES 1,850/ton";
    USD 1 = KES 99.56 (March 2019)  -> tipping fee ~USD 1.0/t, cost ~USD 18.6/t.
    https://www.africancleancities.org/sites/default/files/2022/07/nairobi_en.pdf
  - Kasozi & von Blottnitz (Univ. Cape Town for the City Council of Nairobi / UNEP), "Solid Waste
    Management in Nairobi: A Situation Analysis", draft 17 Feb 2010: CCN disposal cost to Dandora
    1,020 KShs/t (April 2009), estimated 4,089 KShs/t at the planned Ruai landfill; proposed
    weight-based collection+disposal charges 3.2 KShs/kg (residential) and 2.0 KShs/kg
    (non-domestic).  https://www.ecopost.co.ke/assets/pdf/nairobi_solid_waste.pdf
    Converted with the World Bank official exchange rate 2009 (77.35 KES/USD).
  - KCCA, "Kampala Waste Treatment and Disposal PPP - Project Teaser" (Oct 2017): >= 1,000 t/d MSW
    guaranteed, operator remunerated through "gate fees charged to the KCCA" (amount not stated).
    https://www.kcca.go.ug/uDocs/kampala-waste-treatment-and-disposal-ppp.pdf
  - World Bank, What a Waste 2.0 (2018): integrated waste-management operating cost about USD 35/t
    in low-income countries vs > USD 100/t in high-income countries (report value; the report PDF
    could not be opened from this machine - TO BE VERIFIED against
    https://openknowledge.worldbank.org/handle/10986/30317 ).
  - Exchange rates: World Bank WDI PA.NUS.FCRF (KES/USD 2009: 77.35, 2019: 101.99, 2024: 134.82)
    https://api.worldbank.org/v2/country/KE/indicator/PA.NUS.FCRF?date=2009:2024&format=json
  These are disposal COSTS or collection charges, not necessarily fees a private facility could
  collect; the actual tipping fee levied at Dandora (~USD 1/t) is far below cost.

Reproducibility:  cd <repo>/revision ; ../.venv/bin/python E9_gate_fee_scenarios.py  (seconds)
Baseline outputs in validation/Result/ are NOT touched.
Outputs: E9_gate_fee_scenarios.csv, E9_gate_fee_breakeven.csv, E9_feedstock_check.csv, E9_gate_fee_summary.txt
"""
import os, sys, itertools
import numpy as np, pandas as pd

from _paths import HERE, REPO, RESULT, SOURCES, NOTES  # revision/_paths.py
from _paths import BASE as BASE_DIR   # (this script defines its own BASE dict for the submitted TEA inputs)
OUT = os.path.join(RESULT, "E9_gate_fee_scenarios.csv"); OUT_BE = os.path.join(RESULT, "E9_gate_fee_breakeven.csv")
OUT_FS = os.path.join(RESULT, "E9_feedstock_check.csv"); OUT_TXT = os.path.join(RESULT, "E9_gate_fee_summary.txt")

# ---------------- submitted TEA inputs (copied verbatim from validation/screening_tea.py)
DAYS = 330
BASE = dict(CH4_feed_tpd=48.0, CO2_feed_tpd=42.0, solar_added_H2_tpd=2.52,
            solid_carbon_stage1_tpd=34.85150348488499, solid_carbon_stage3_tpd=1.4760297024500548,
            recovered_water_tpd=16.67253924345945, Q1=3030.9360665085123, Q2=-196.44502096811965, Q3=-1478.5055973735964)
TECH = dict(solid_C=BASE["solid_carbon_stage1_tpd"] + BASE["solid_carbon_stage3_tpd"], water=BASE["recovered_water_tpd"],
            heat_kW=max(0.0, BASE["Q1"] + BASE["Q2"] + BASE["Q3"]), elec_kW=5000.0)
SCEN = {
    "Conservative": dict(fee=8.0, cprice=180.0, water=0.30, credit=0.0, carb_val=15.0, carb_tpd=0.0, elec=0.08, heat=0.025, opex=1.8e6, capex=14.0e6, waste_tpd=100.0, cred_frac=0.0),
    "Base":         dict(fee=15.0, cprice=350.0, water=0.75, credit=35.0, carb_val=25.0, carb_tpd=0.0, elec=0.06, heat=0.018, opex=1.5e6, capex=12.0e6, waste_tpd=100.0, cred_frac=0.0),
    "Opportunity":  dict(fee=25.0, cprice=700.0, water=1.50, credit=80.0, carb_val=40.0, carb_tpd=10.0, elec=0.04, heat=0.010, opex=1.3e6, capex=10.0e6, waste_tpd=100.0, cred_frac=0.50),
}
def evaluate(s, path, waste_tpd=None, fee=None):
    waste_tpd = s["waste_tpd"] if waste_tpd is None else waste_tpd; fee = s["fee"] if fee is None else fee
    solid = TECH["solid_C"]
    if path == "Carbon Fixation Path":
        carb_tpd = s["carb_tpd"]; cred = s["cred_frac"] * (solid * 44.01 / 12.01 + carb_tpd)
    else: carb_tpd = 0.0; cred = 0.0
    rev_gate = waste_tpd * DAYS * fee; rev_c = solid * DAYS * s["cprice"]; rev_w = TECH["water"] * DAYS * s["water"]
    rev_carb = carb_tpd * DAYS * s["carb_val"]; rev_cred = cred * DAYS * s["credit"]
    c_el = TECH["elec_kW"] * 24 * DAYS * s["elec"]; c_heat = TECH["heat_kW"] * 24 * DAYS * s["heat"]; c_opex = s["opex"]
    val = rev_gate + rev_c + rev_w + rev_carb + rev_cred - c_el - c_heat - c_opex
    return dict(gate_revenue=rev_gate, carbon_revenue=rev_c, water_value=rev_w, carbonate_value=rev_carb, credit_value=rev_cred,
                electricity_cost=c_el, heat_cost=c_heat, fixed_opex=c_opex, annual_value=val, payback_y=(s["capex"] / val if val > 0 else np.nan))

# ---------------- 1. reproduce submitted results
ref = pd.read_csv(os.path.join(BASE_DIR, "validation", "Result", "screening_tea_summary.csv"))
chk = []
for _, r in ref.iterrows():
    e = evaluate(SCEN[r.Scenario], r.Case)
    chk.append(dict(Scenario=r.Scenario, Case=r.Case, submitted_annual_value=r["Annual value [$ / y]"], reproduced=e["annual_value"],
                    diff=e["annual_value"] - r["Annual value [$ / y]"], submitted_gate_revenue=r["Waste gate revenue [$ / y]"], reproduced_gate=e["gate_revenue"]))
chk = pd.DataFrame(chk); assert chk["diff"].abs().max() < 1e-6, chk

# ---------------- 2. feedstock boundary check
rho_CH4, rho_CO2 = 0.7168, 1.9768   # kg/Nm3 at 0 C, 1 atm
ch4_nm3 = 48.0e3 / rho_CH4; co2_nm3 = 32.0e3 / rho_CO2
fs = [dict(item="organic waste intake", value=800.0, unit="t/d", note="manuscript boundary (not in code)"),
      dict(item="CH4 to Stage 1", value=48.0, unit="t/d", note=f"= {ch4_nm3:,.0f} Nm3/d = 60 kg CH4 per t waste"),
      dict(item="biogas CO2 to Stage 2", value=32.0, unit="t/d", note=f"= {co2_nm3:,.0f} Nm3/d = 40 kg CO2 per t waste"),
      dict(item="CH4 yield per t waste", value=ch4_nm3 / 800, unit="Nm3 CH4/t waste", note="83.7; plausible for food-rich organic waste (order 50-100)"),
      dict(item="CH4 fraction if 48+32 t/d were the whole biogas", value=ch4_nm3 / (ch4_nm3 + co2_nm3), unit="vol fraction", note="0.80 - higher than typical raw biogas (0.55-0.65)"),
      dict(item="biogas CO2 at 60 vol% CH4", value=ch4_nm3 * 0.4 / 0.6 * rho_CO2 / 1000, unit="t/d", note="88 t/d -> the 32 t/d routed to Stage 2 is ~36 % of biogas CO2; the rest must be vented/used elsewhere - manuscript should say so"),
      dict(item="total biogas at 60 vol% CH4", value=ch4_nm3 / 0.6 / 800, unit="Nm3 biogas/t waste", note="139 Nm3/t; high end for mixed organic MSW, typical for food waste")]
fs = pd.DataFrame(fs); fs.to_csv(OUT_FS, index=False)

# ---------------- 3. scenarios: chargeable basis x fee level
bases = [("submitted (100 t/d, undocumented)", 100.0), ("no gate fee (0 t/d)", 0.0),
         ("municipal share 25 % of 800 t/d", 200.0), ("municipal share 50 % of 800 t/d", 400.0), ("municipal share 75 % of 800 t/d", 600.0), ("full intake 800 t/d", 800.0)]
fees = [("0 (no fee)", 0.0), ("1.0 (Nairobi tipping fee KES 100/t, ACC 2022)", 1.0), ("8 (submitted Conservative)", 8.0),
        ("13.2 (Nairobi CCN disposal cost 2009, 1,020 KShs/t)", 13.2), ("15 (submitted Base)", 15.0), ("18.6 (Nairobi cost KES 1,850/t, ACC 2022)", 18.6),
        ("25 (submitted Opportunity)", 25.0), ("35 (WaW 2.0 low-income operating cost, to be verified)", 35.0), ("52.9 (Nairobi Ruai landfill estimate 2009)", 52.9)]
rows = []
for sname, s in SCEN.items():
    for path in ["Carbon Product Path", "Carbon Fixation Path"]:
        for (blab, tpd), (flab, fee) in itertools.product(bases, fees):
            e = evaluate(s, path, tpd, fee)
            rows.append(dict(scenario=sname, path=path, charge_basis=blab, chargeable_tpd=tpd, fee_usd_per_t=fee, fee_label=flab,
                             gate_revenue_usd_y=e["gate_revenue"], annual_value_usd_y=e["annual_value"], simple_payback_y=e["payback_y"],
                             gate_share_of_gross_revenue=e["gate_revenue"] / (e["gate_revenue"] + e["carbon_revenue"] + e["water_value"] + e["carbonate_value"] + e["credit_value"])))
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)

# ---------------- 4. break-even and drivers (Base, product path)
sB = SCEN["Base"]; e0 = evaluate(sB, "Carbon Product Path", 0.0, 0.0)
be = []
for blab, tpd in bases[2:]:
    fee_val0 = -e0["annual_value"] / (tpd * DAYS) if e0["annual_value"] < 0 else 0.0
    fee_pb10 = (sB["capex"] / 10.0 - e0["annual_value"]) / (tpd * DAYS)
    fee_pb5 = (sB["capex"] / 5.0 - e0["annual_value"]) / (tpd * DAYS)
    be.append(dict(charge_basis=blab, chargeable_tpd=tpd, fee_for_zero_annual_value=fee_val0, fee_for_payback_10y=fee_pb10, fee_for_payback_5y=fee_pb5))
be = pd.DataFrame(be); be.to_csv(OUT_BE, index=False)
cprice_be_gate0 = (e0["electricity_cost"] + e0["heat_cost"] + e0["fixed_opex"] - e0["water_value"]) / (TECH["solid_C"] * DAYS)
cprice_pb10_gate0 = (sB["capex"] / 10.0 + e0["electricity_cost"] + e0["heat_cost"] + e0["fixed_opex"] - e0["water_value"]) / (TECH["solid_C"] * DAYS)

L = []
L.append("E9 gate-fee basis — summary (screening TEA engine reproduced exactly; max |diff| = %.2e USD/y)" % chk["diff"].abs().max())
L.append("\n--- Reproduction of submitted TEA ---\n" + chk.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
L.append("\n--- Feedstock boundary check ---\n" + fs.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
piv = df[(df.path == "Carbon Product Path")].pivot_table(index=["scenario", "charge_basis"], columns="fee_usd_per_t", values="annual_value_usd_y")
L.append("\n--- Annual value [USD/y], Carbon Product Path, by chargeable basis x fee ---\n" + piv.to_string(float_format=lambda v: f"{v:,.0f}"))
pivp = df[(df.path == "Carbon Product Path") & (df.scenario == "Base")].pivot_table(index="charge_basis", columns="fee_usd_per_t", values="simple_payback_y")
L.append("\n--- Simple payback [y], Base, Carbon Product Path ---\n" + pivp.to_string(float_format=lambda v: f"{v:,.1f}"))
L.append("\n--- Base, product path, gate fee = 0: components [USD/y] ---")
for k in ["carbon_revenue", "water_value", "electricity_cost", "heat_cost", "fixed_opex", "annual_value"]: L.append(f"  {k:18s} {e0[k]:>14,.0f}")
L.append(f"  payback at gate 0: {e0['payback_y']:.1f} y; carbon price for zero annual value at gate 0: {cprice_be_gate0:,.0f} USD/t; for 10-y payback: {cprice_pb10_gate0:,.0f} USD/t")
L.append("\n--- Break-even gate fee [USD/t], Base, product path ---\n" + be.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
with open(OUT_TXT, "w") as fh: fh.write("\n".join(L) + "\n")
print("\n".join(L)); print("Saved:", OUT, OUT_BE, OUT_FS, OUT_TXT)
