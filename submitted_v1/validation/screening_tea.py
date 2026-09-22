# -*- coding: utf-8 -*-
# ============================================================
# Screening-Level TEA for:
# Waste / methane / CO2 -> solid carbon + recovered water
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# ------------------------------------------------------------
# 1) Baseline technical outputs from your thermodynamic workflow
#    (replace here if simulation updates)
# ------------------------------------------------------------
OPERATING_DAYS_PER_YEAR = 330  # screening assumption

BASELINE = {
    "CH4_feed_tpd": 48.0,
    "CO2_feed_tpd": 42.0,
    "solar_added_H2_tpd": 2.52,
    "solid_carbon_stage1_tpd": 34.85150348488499,
    "solid_carbon_stage3_tpd": 1.4760297024500548,
    "recovered_water_tpd": 16.67253924345945,
    "Q_stage1_kW": 3030.9360665085123,   # heat in
    "Q_stage2_kW": -196.44502096811965,  # heat out
    "Q_stage3_kW": -1478.5055973735964,  # heat out
}

# Optional: if you want to explore a carbonate/mineralized-carbon route
# This is NOT from the current thermodynamic baseline. It is a scenario variable.
DEFAULT_CARBONATE_FIXED_CO2_TPD = 0.0

# ------------------------------------------------------------
# 2) Derived technical quantities
# ------------------------------------------------------------
def build_technical_baseline(baseline: dict) -> dict:
    tech = dict(baseline)

    tech["solid_carbon_total_tpd"] = (
        baseline["solid_carbon_stage1_tpd"] + baseline["solid_carbon_stage3_tpd"]
    )

    # Net external heat requirement: absorb positive heat demand, allow internal exotherm credit
    tech["net_external_heat_kW"] = max(
        0.0,
        baseline["Q_stage1_kW"] + baseline["Q_stage2_kW"] + baseline["Q_stage3_kW"]
    )

    # If solar electricity is onsite, you can still assign an LCOE-like cost.
    # Use this as the electricity consumption proxy for added H2 pathway, auxiliaries, etc.
    tech["onsite_electric_load_kW"] = 5000.0

    return tech

TECH = build_technical_baseline(BASELINE)

# ------------------------------------------------------------
# 3) Economic scenario definitions
# ------------------------------------------------------------
@dataclass
class EconomicScenario:
    name: str

    # Revenue-side
    waste_gate_fee_usd_per_t_waste: float
    carbon_product_price_usd_per_t: float
    recovered_water_value_usd_per_t: float
    carbon_credit_usd_per_tCO2e: float

    # Carbon-fixation path only
    carbonate_value_usd_per_tCO2_fixed: float
    carbonate_fixed_co2_tpd: float

    # Cost-side
    electricity_cost_usd_per_kWh: float
    heat_cost_usd_per_kWhth: float
    fixed_opex_usd_per_year: float

    # Capital
    capex_usd: float

    # Throughput anchor for gate fee
    waste_tpd: float

    # Creditability factor for removals [0-1]
    creditable_fraction_of_fixed_CO2e: float = 0.0

# NOTE:
# These are placeholder screening values.
# Replace them with your own local assumptions as needed.
SCENARIOS = [
    EconomicScenario(
        name="Conservative",
        waste_gate_fee_usd_per_t_waste=8.0,
        carbon_product_price_usd_per_t=180.0,
        recovered_water_value_usd_per_t=0.30,
        carbon_credit_usd_per_tCO2e=0.0,
        carbonate_value_usd_per_tCO2_fixed=15.0,
        carbonate_fixed_co2_tpd=0.0,
        electricity_cost_usd_per_kWh=0.08,
        heat_cost_usd_per_kWhth=0.025,
        fixed_opex_usd_per_year=1.8e6,
        capex_usd=14.0e6,
        waste_tpd=100.0,
        creditable_fraction_of_fixed_CO2e=0.0,
    ),
    EconomicScenario(
        name="Base",
        waste_gate_fee_usd_per_t_waste=15.0,
        carbon_product_price_usd_per_t=350.0,
        recovered_water_value_usd_per_t=0.75,
        carbon_credit_usd_per_tCO2e=35.0,
        carbonate_value_usd_per_tCO2_fixed=25.0,
        carbonate_fixed_co2_tpd=0.0,
        electricity_cost_usd_per_kWh=0.06,
        heat_cost_usd_per_kWhth=0.018,
        fixed_opex_usd_per_year=1.5e6,
        capex_usd=12.0e6,
        waste_tpd=100.0,
        creditable_fraction_of_fixed_CO2e=0.0,
    ),
    EconomicScenario(
        name="Opportunity",
        waste_gate_fee_usd_per_t_waste=25.0,
        carbon_product_price_usd_per_t=700.0,
        recovered_water_value_usd_per_t=1.50,
        carbon_credit_usd_per_tCO2e=80.0,
        carbonate_value_usd_per_tCO2_fixed=40.0,
        carbonate_fixed_co2_tpd=10.0,
        electricity_cost_usd_per_kWh=0.04,
        heat_cost_usd_per_kWhth=0.010,
        fixed_opex_usd_per_year=1.3e6,
        capex_usd=10.0e6,
        waste_tpd=100.0,
        creditable_fraction_of_fixed_CO2e=0.50,
    ),
]

# ------------------------------------------------------------
# 4) TEA engine
# ------------------------------------------------------------
def tC_to_tCO2e(tC: float) -> float:
    return tC * (44.01 / 12.01)

def annualize_tpd(tpd: float, days: float = OPERATING_DAYS_PER_YEAR) -> float:
    return tpd * days

def evaluate_case(tech: dict, econ: EconomicScenario, case_name: str) -> dict:
    # Product and fixation choices
    if case_name == "Carbon Product Path":
        solid_carbon_tpd = tech["solid_carbon_total_tpd"]
        carbonate_fixed_co2_tpd = 0.0
        creditable_co2e_tpd = 0.0

    elif case_name == "Carbon Fixation Path":
        solid_carbon_tpd = tech["solid_carbon_total_tpd"]
        carbonate_fixed_co2_tpd = econ.carbonate_fixed_co2_tpd

        # Creditable removal can come from durable fixed carbon streams.
        # Screening assumption: only a fraction is counted.
        fixed_from_solid_carbon_tCO2e_tpd = tC_to_tCO2e(solid_carbon_tpd)
        fixed_from_carbonate_tCO2e_tpd = carbonate_fixed_co2_tpd

        creditable_co2e_tpd = econ.creditable_fraction_of_fixed_CO2e * (
            fixed_from_solid_carbon_tCO2e_tpd + fixed_from_carbonate_tCO2e_tpd
        )

    else:
        raise ValueError(f"Unknown case_name: {case_name}")

    # Annual revenues
    annual_waste_revenue = annualize_tpd(econ.waste_tpd) * econ.waste_gate_fee_usd_per_t_waste
    annual_carbon_revenue = annualize_tpd(solid_carbon_tpd) * econ.carbon_product_price_usd_per_t
    annual_water_value = annualize_tpd(tech["recovered_water_tpd"]) * econ.recovered_water_value_usd_per_t
    annual_carbonate_value = annualize_tpd(carbonate_fixed_co2_tpd) * econ.carbonate_value_usd_per_tCO2_fixed
    annual_credit_value = annualize_tpd(creditable_co2e_tpd) * econ.carbon_credit_usd_per_tCO2e

    # Annual costs
    annual_electricity_cost = (
        tech["onsite_electric_load_kW"] * 24.0 * OPERATING_DAYS_PER_YEAR * econ.electricity_cost_usd_per_kWh
    )
    annual_heat_cost = (
        tech["net_external_heat_kW"] * 24.0 * OPERATING_DAYS_PER_YEAR * econ.heat_cost_usd_per_kWhth
    )
    annual_fixed_opex = econ.fixed_opex_usd_per_year

    annual_value = (
        annual_waste_revenue
        + annual_carbon_revenue
        + annual_water_value
        + annual_carbonate_value
        + annual_credit_value
        - annual_electricity_cost
        - annual_heat_cost
        - annual_fixed_opex
    )

    simple_payback_years = np.nan
    if annual_value > 0:
        simple_payback_years = econ.capex_usd / annual_value

    return {
        "Scenario": econ.name,
        "Case": case_name,

        "Waste gate revenue [$ / y]": annual_waste_revenue,
        "Carbon product revenue [$ / y]": annual_carbon_revenue,
        "Recovered water value [$ / y]": annual_water_value,
        "Carbonate/fixed-carbon value [$ / y]": annual_carbonate_value,
        "Carbon credit value [$ / y]": annual_credit_value,

        "Electricity cost [$ / y]": annual_electricity_cost,
        "Heat cost [$ / y]": annual_heat_cost,
        "Fixed O&M [$ / y]": annual_fixed_opex,

        "Annual value [$ / y]": annual_value,
        "Simple payback [y]": simple_payback_years,

        "Solid carbon product [t / d]": solid_carbon_tpd,
        "Recovered water [t / d]": tech["recovered_water_tpd"],
        "Carbonate-fixed CO2 [t / d]": carbonate_fixed_co2_tpd,
        "Creditable CO2e [t / d]": creditable_co2e_tpd,
        "Net external heat [kW]": tech["net_external_heat_kW"],
        "Onsite electric load [kW]": tech["onsite_electric_load_kW"],
    }

def evaluate_all(tech: dict, scenarios: list[EconomicScenario]) -> pd.DataFrame:
    rows = []
    for econ in scenarios:
        for case_name in ["Carbon Product Path", "Carbon Fixation Path"]:
            rows.append(evaluate_case(tech, econ, case_name))
    return pd.DataFrame(rows)

tea_df = evaluate_all(TECH, SCENARIOS)

# ------------------------------------------------------------
# 5) Display core result table
# ------------------------------------------------------------
display_cols = [
    "Scenario", "Case",
    "Solid carbon product [t / d]",
    "Recovered water [t / d]",
    "Carbonate-fixed CO2 [t / d]",
    "Annual value [$ / y]",
    "Simple payback [y]",
]
print("\n===== SCREENING-LEVEL TEA SUMMARY =====")
print(tea_df[display_cols].to_string(index=False))

# Save full result table
tea_df.to_csv("screening_tea_summary.csv", index=False)

# ------------------------------------------------------------
# 6) Breakdown plot
# ------------------------------------------------------------
def plot_breakdown(df: pd.DataFrame, scenario_name: str, case_name: str, filename: str):
    row = df[(df["Scenario"] == scenario_name) & (df["Case"] == case_name)].iloc[0]

    labels = [
        "Waste fee",
        "Carbon product",
        "Water",
        "Carbonate/fixed",
        "Carbon credit",
        "Electricity",
        "Heat",
        "Fixed O&M",
    ]
    values = [
        row["Waste gate revenue [$ / y]"],
        row["Carbon product revenue [$ / y]"],
        row["Recovered water value [$ / y]"],
        row["Carbonate/fixed-carbon value [$ / y]"],
        row["Carbon credit value [$ / y]"],
        -row["Electricity cost [$ / y]"],
        -row["Heat cost [$ / y]"],
        -row["Fixed O&M [$ / y]"],
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("USD / year")
    plt.title(f"Annual value breakdown: {scenario_name} | {case_name}")
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()

plot_breakdown(tea_df, "Base", "Carbon Product Path", "tea_breakdown_base_carbon_product.png")
plot_breakdown(tea_df, "Base", "Carbon Fixation Path", "tea_breakdown_base_carbon_fixation.png")

# ------------------------------------------------------------
# 7) Sensitivity analysis
#    Base scenario only, one-at-a-time +/- 30%
# ------------------------------------------------------------
def sensitivity_analysis(tech: dict, base_scenario: EconomicScenario, case_name: str,
                         delta: float = 0.30) -> pd.DataFrame:
    params = {
        "waste_gate_fee_usd_per_t_waste": "Waste gate fee",
        "carbon_product_price_usd_per_t": "Carbon product price",
        "recovered_water_value_usd_per_t": "Recovered water value",
        "carbon_credit_usd_per_tCO2e": "Carbon credit value",
        "carbonate_value_usd_per_tCO2_fixed": "Carbonate value",
        "electricity_cost_usd_per_kWh": "Electricity cost",
        "heat_cost_usd_per_kWhth": "Heat cost",
        "fixed_opex_usd_per_year": "Fixed O&M",
        "capex_usd": "CAPEX",
    }

    base_result = evaluate_case(tech, base_scenario, case_name)
    base_annual_value = base_result["Annual value [$ / y]"]

    rows = []
    for attr, label in params.items():
        original = getattr(base_scenario, attr)

        if original == 0:
            low = 0.0
            high = 0.0
        else:
            low = original * (1 - delta)
            high = original * (1 + delta)

        scenario_low = EconomicScenario(**asdict(base_scenario))
        setattr(scenario_low, attr, low)
        result_low = evaluate_case(tech, scenario_low, case_name)

        scenario_high = EconomicScenario(**asdict(base_scenario))
        setattr(scenario_high, attr, high)
        result_high = evaluate_case(tech, scenario_high, case_name)

        rows.append({
            "Parameter": label,
            "Base annual value [$ / y]": base_annual_value,
            "Low annual value [$ / y]": result_low["Annual value [$ / y]"],
            "High annual value [$ / y]": result_high["Annual value [$ / y]"],
            "Delta low [$ / y]": result_low["Annual value [$ / y]"] - base_annual_value,
            "Delta high [$ / y]": result_high["Annual value [$ / y]"] - base_annual_value,
            "Impact abs max [$ / y]": max(
                abs(result_low["Annual value [$ / y]"] - base_annual_value),
                abs(result_high["Annual value [$ / y]"] - base_annual_value)
            )
        })

    sens_df = pd.DataFrame(rows).sort_values("Impact abs max [$ / y]", ascending=False)
    return sens_df

base_scenario = [s for s in SCENARIOS if s.name == "Base"][0]

sens_product = sensitivity_analysis(TECH, base_scenario, "Carbon Product Path")
sens_fixation = sensitivity_analysis(TECH, base_scenario, "Carbon Fixation Path")

sens_product.to_csv("sensitivity_carbon_product.csv", index=False)
sens_fixation.to_csv("sensitivity_carbon_fixation.csv", index=False)

print("\n===== SENSITIVITY | BASE | Carbon Product Path =====")
print(sens_product[["Parameter", "Delta low [$ / y]", "Delta high [$ / y]"]].to_string(index=False))

print("\n===== SENSITIVITY | BASE | Carbon Fixation Path =====")
print(sens_fixation[["Parameter", "Delta low [$ / y]", "Delta high [$ / y]"]].to_string(index=False))

# ------------------------------------------------------------
# 8) Tornado-like plots
# ------------------------------------------------------------
def plot_tornado(sens_df: pd.DataFrame, title: str, filename: str):
    df = sens_df.copy().iloc[::-1]

    y = np.arange(len(df))
    low = df["Delta low [$ / y]"].values
    high = df["Delta high [$ / y]"].values

    plt.figure(figsize=(9, 6))
    for i in range(len(df)):
        plt.plot([low[i], high[i]], [y[i], y[i]], marker="o")
    plt.axvline(0.0, linestyle="--")
    plt.yticks(y, df["Parameter"])
    plt.xlabel("Change in annual value [USD / year]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()

plot_tornado(sens_product, "Sensitivity: Base | Carbon Product Path", "tornado_base_carbon_product.png")
plot_tornado(sens_fixation, "Sensitivity: Base | Carbon Fixation Path", "tornado_base_carbon_fixation.png")

# ------------------------------------------------------------
# 9) Optional quick view of assumptions
# ------------------------------------------------------------
assumptions_df = pd.DataFrame([asdict(s) for s in SCENARIOS])
assumptions_df.to_csv("screening_tea_assumptions.csv", index=False)

print("\nSaved files:")
print(" - screening_tea_summary.csv")
print(" - screening_tea_assumptions.csv")
print(" - sensitivity_carbon_product.csv")
print(" - sensitivity_carbon_fixation.csv")
print(" - tea_breakdown_base_carbon_product.png")
print(" - tea_breakdown_base_carbon_fixation.png")
print(" - tornado_base_carbon_product.png")
print(" - tornado_base_carbon_fixation.png")
