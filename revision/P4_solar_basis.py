# -*- coding: utf-8 -*-
"""
P4: Basis of the "5 MW" solar-electrolysis figure (reviewer R2-3)
==================================================================
FUPROC-D-26-00269 Major Revision.

The submitted model produces the solar hydrogen as
    n_H2 = power_kW * 1000 * 86400 * eff / LHV / M_H2,   LHV = 120e6 J/kg, eff = 0.70, power = 5000 kW
(Workflow_cantera.py, solar_h2_kmol_per_day), i.e. 5 MW is the CONTINUOUS AVERAGE ELECTRIC
POWER delivered to the electrolyser, with a 70 % LHV-based conversion efficiency. It is not
a PV nameplate capacity. This script (1) recomputes the power on LHV and HHV bases, (2)
converts the continuous power into PV capacity and land for the equatorial dry belt using
the World Bank / Solargis Global PV Potential dataset (PVOUT Level 1, long-term
kWh/kWp/day), (3) sizes the electrolyser and the H2 buffer for PV-following operation and
compares with a baseload (geothermal / grid) alternative, and (4) lists every place where
"5 MW / 5,000 kW" appears in the submitted code and revision files.

External data (all URLs recorded in the outputs)
  - H2 heating values: US DOE AFDC Fuel Properties Comparison, https://afdc.energy.gov/fuels/properties
    LHV 51,585 Btu/lb (= 33.3 kWh/kg, 120.0 MJ/kg); HHV 61,013 Btu/lb (= 39.4 kWh/kg, 141.9 MJ/kg).
  - PVOUT by country: Solargis for the World Bank, "Global Photovoltaic Power Potential by Country"
    (ESMAP, June 2020), country-ranking workbook
    https://datacatalogfiles.worldbank.org/ddh-published/0038379/1/DR0046831/solargis_pvpotential_countryranking_2020_data.xlsx
    (dataset page https://energydata.info/dataset/global-photovoltaic-power-potential-by-country ;
    report https://documents1.worldbank.org/curated/en/466331592817725242/pdf/Global-Photovoltaic-Power-Potential-by-Country.pdf ).
    PVOUT = specific yield of a utility-scale, fixed-mounted, optimally tilted monofacial c-Si system,
    long-term average, including 3.5 % soiling and 7.5 % other losses. A copy of the workbook is kept as
    P4_source_solargis_pvpotential_countryranking_2020_data.xlsx.
  - Electrolyser efficiency context: IRENA, Green Hydrogen Cost Reduction (2020),
    https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2020/Dec/IRENA_Green_hydrogen_cost_2020.pdf
    "Efficiency at nominal capacity is 65 % (LHV, 51.2 kWh/kg H2) in 2020 and 76 % (43.8 kWh/kg H2) in 2050".
  - Land use: the 1-2 ha/MWp range used below is an ASSUMPTION for fixed-tilt utility PV in the tropics
    (dense fixed-tilt layouts ~1 ha/MWp, generous layouts ~2 ha/MWp). Reference methodology and US
    empirical data: Ong, Campbell, Denholm, Margolis, Heath, "Land-Use Requirements for Solar Power
    Plants in the United States", NREL/TP-6A20-56290 (2013), doi:10.2172/1086349 (bibliography verified via
    Crossref; the PDF at nrel.gov was not reachable from this machine, so no figure is quoted from it), and
    Bolinger & Bolinger, "Land Requirements for Utility-Scale PV: An Empirical Update on Power and Energy
    Density", IEEE J. Photovoltaics (2022), doi:10.1109/JPHOTOV.2021.3136805.
  - Geothermal baseload: IRENA, Renewable Capacity Statistics 2025 (Abu Dhabi, 2025), table "Geothermal
    energy, CAP (MW)", p. 57: Kenya 863 (2020/2021), 950 (2022), 985 (2023), 940 (2024) MW; Africa 947 MW (2024).
    https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Mar/IRENA_DAT_RE_Capacity_Statistics_2025.pdf
    (copy kept as P4_source_IRENA_Renewable_Capacity_Statistics_2025.pdf). National statistics: EPRA (Kenya),
    Energy & Petroleum Statistics Report FY 2023/2024 (installed geothermal 943.7 MW as at 30 June 2024 according
    to the report summary; the PDF at epra.go.ke could not be opened from this machine and must be checked):
    https://www.epra.go.ke/sites/default/files/2024-10/EPRA%20Energy%20and%20Petroleum%20Statistics%20Report%20FY%202023-2024_2.pdf
  - Annual-energy convention: the process model is a continuous daily balance (kW, t/d); the screening TEA uses
    330 operating days per year (OPERATING_DAYS_PER_YEAR = 330 in validation/screening_tea.py). Annual figures are
    therefore reported below on both bases and the manuscript should state which one it uses.

Reproducibility:  cd <repo>/revision ; ../.venv/bin/python P4_solar_basis.py   (seconds; needs openpyxl)
Outputs: P4_solar_basis.csv (all derived numbers), P4_pvout_countries.csv, P4_solar_basis_summary.txt
"""
import os, sys, re, glob
import numpy as np
import pandas as pd

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W

OUT = os.path.join(RESULT, "P4_solar_basis.csv"); OUT_PV = os.path.join(RESULT, "P4_pvout_countries.csv"); OUT_TXT = os.path.join(RESULT, "P4_solar_basis_summary.txt")
XLSX = os.path.join(SOURCES, "P4_source_solargis_pvpotential_countryranking_2020_data.xlsx")

# ---------------- 1. basis
LHV_kWh_kg = 51585 * 2.326 / 3600.0 * 1000 / 1000   # Btu/lb -> kJ/kg -> kWh/kg  (51,585 Btu/lb * 2.326 kJ/kg per Btu/lb = 119,987 kJ/kg)
HHV_kWh_kg = 61013 * 2.326 / 3600.0
eff = W.electrolyzer_eff; P_kW = W.solar_power_kW
h2_kmol_d = W.solar_h2_kmol_per_day(P_kW, eff); h2_t_d = h2_kmol_d * W.MW["H2"] / 1000.0; h2_kg_h = h2_t_d * 1000 / 24
rows = []
def add(section, item, value, unit, note=""): rows.append(dict(section=section, item=item, value=value, unit=unit, note=note))
add("1 basis", "code LHV used (Workflow_cantera.solar_h2_kmol_per_day)", 120e6 / 3.6e6, "kWh/kg", "LHV=120e6 J/kg in code")
add("1 basis", "AFDC LHV", LHV_kWh_kg, "kWh/kg", "51,585 Btu/lb"); add("1 basis", "AFDC HHV", HHV_kWh_kg, "kWh/kg", "61,013 Btu/lb")
add("1 basis", "H2 produced in model", h2_t_d, "t/d", f"{h2_kmol_d:.1f} kmol/d, {h2_kg_h:.1f} kg/h")
add("1 basis", "specific consumption implied (LHV/eff)", LHV_kWh_kg / eff, "kWh/kg H2", "eff 0.70 LHV")
P_LHV = h2_kg_h * LHV_kWh_kg / eff; P_HHV = h2_kg_h * HHV_kWh_kg / eff
add("1 basis", "continuous power, eff 0.70 on LHV", P_LHV, "kW", "= submitted 5,000 kW")
add("1 basis", "continuous power, eff 0.70 on HHV", P_HHV, "kW", "if 0.70 were HHV-based, the same H2 needs this power")
add("1 basis", "LHV-equivalent efficiency if 0.70 is HHV", eff * HHV_kWh_kg / LHV_kWh_kg, "-", "0.70 HHV = 0.83 LHV (implausible today)")
add("1 basis", "HHV-equivalent efficiency of 0.70 LHV", eff * LHV_kWh_kg / HHV_kWh_kg, "-", "")
add("1 basis", "IRENA 2020 system efficiency (LHV)", 0.65, "-", "51.2 kWh/kg"); add("1 basis", "IRENA 2050 target (LHV)", 0.76, "-", "43.8 kWh/kg")
add("1 basis", "power at IRENA 2020 efficiency (LHV) for same H2", h2_kg_h * 51.2, "kW", "")
add("1 basis", "power at IRENA 2050 efficiency (LHV) for same H2", h2_kg_h * 43.8, "kW", "")
E_day_LHV = P_LHV * 24 / 1000.0   # MWh/d
E_day_HHV = P_HHV * 24 / 1000.0
add("1 basis", "daily electricity, LHV basis", E_day_LHV, "MWh/d", ""); add("1 basis", "daily electricity, HHV basis", E_day_HHV, "MWh/d", "")

# ---------------- 2. PV capacity from PVOUT
ss = pd.read_excel(XLSX, "Summary statistics", header=None); hdr = list(ss.iloc[1]); ss = ss.iloc[2:].copy()
ss.columns = [f"{h}_{i}" if i >= 4 else h for i, h in enumerate(hdr)]
pv = ss.rename(columns={"Average_15": "PVOUT_avg", "10th percentile_13": "PVOUT_p10", "90th percentile_18": "PVOUT_p90",
                        "Minimum_12": "PVOUT_min", "Maximum_19": "PVOUT_max", "Average_7": "GHI_avg"})
iso = {"KEN": "Kenya", "SOM": "Somalia", "ETH": "Ethiopia", "DJI": "Djibouti", "ERI": "Eritrea", "SDN": "Sudan", "TCD": "Chad", "NER": "Niger",
       "MLI": "Mali", "MRT": "Mauritania", "NAM": "Namibia", "EGY": "Egypt", "SAU": "Saudi Arabia", "AUS": "Australia", "JPN": "Japan", "DEU": "Germany"}
pv = pv[pv.ISO_A3.isin(iso)][["ISO_A3", "Country or region", "GHI_avg", "PVOUT_min", "PVOUT_p10", "PVOUT_avg", "PVOUT_p90", "PVOUT_max"]].copy()
for c in ["GHI_avg", "PVOUT_min", "PVOUT_p10", "PVOUT_avg", "PVOUT_p90", "PVOUT_max"]: pv[c] = pv[c].astype(float)
pv["PVOUT_avg_kWh_kWp_yr"] = pv.PVOUT_avg * 365; pv["PVOUT_p90_kWh_kWp_yr"] = pv.PVOUT_p90 * 365
pv["capacity_factor_avg"] = pv.PVOUT_avg / 24.0; pv["capacity_factor_p90"] = pv.PVOUT_p90 / 24.0
pv["PV_MWp_for_5MW_cont_LHV_avg"] = E_day_LHV * 1000 / pv.PVOUT_avg / 1000.0
pv["PV_MWp_for_5MW_cont_LHV_p90site"] = E_day_LHV * 1000 / pv.PVOUT_p90 / 1000.0
pv["PV_MWp_for_HHV_basis_avg"] = E_day_HHV * 1000 / pv.PVOUT_avg / 1000.0
pv["land_ha_at_1ha_per_MWp"] = pv.PV_MWp_for_5MW_cont_LHV_avg * 1.0; pv["land_ha_at_2ha_per_MWp"] = pv.PV_MWp_for_5MW_cont_LHV_avg * 2.0
pv = pv.sort_values("PVOUT_avg", ascending=False); pv.to_csv(OUT_PV, index=False)
belt = pv[pv.ISO_A3.isin(["KEN", "SOM", "ETH", "DJI", "ERI", "SDN", "TCD", "NER", "MLI", "MRT"])]
add("2 PV", "equatorial dry belt PVOUT avg range (country averages)", f"{belt.PVOUT_avg.min():.2f}-{belt.PVOUT_avg.max():.2f}", "kWh/kWp/day", ", ".join(belt["Country or region"]))
add("2 PV", "equatorial dry belt PVOUT 90th-percentile sites", f"{belt.PVOUT_p90.min():.2f}-{belt.PVOUT_p90.max():.2f}", "kWh/kWp/day", "best decile of each country's area")
add("2 PV", "equatorial dry belt specific yield", f"{belt.PVOUT_avg.min()*365:.0f}-{belt.PVOUT_p90.max()*365:.0f}", "kWh/kWp/yr", "avg to p90")
add("2 PV", "equatorial dry belt capacity factor", f"{belt.PVOUT_avg.min()/24:.3f}-{belt.PVOUT_p90.max()/24:.3f}", "-", "PVOUT/24 h")
add("2 PV", "PV capacity for 5 MW continuous (LHV), belt", f"{pv.loc[pv.ISO_A3.isin(belt.ISO_A3),'PV_MWp_for_5MW_cont_LHV_p90site'].min():.1f}-{belt.PV_MWp_for_5MW_cont_LHV_avg.max():.1f}", "MWp", "120 MWh/d / PVOUT")
kmin = float(pv[pv.ISO_A3 == "KEN"].PV_MWp_for_5MW_cont_LHV_p90site.iloc[0]); kavg = float(pv[pv.ISO_A3 == "KEN"].PV_MWp_for_5MW_cont_LHV_avg.iloc[0])
add("2 PV", "PV capacity, Kenya (avg / best-decile sites e.g. Turkana-Marsabit)", f"{kavg:.1f} / {kmin:.1f}", "MWp", "")
add("2 PV", "PV capacity, HHV basis, belt", f"{(E_day_HHV*1000/belt.PVOUT_p90.max()/1000):.1f}-{(E_day_HHV*1000/belt.PVOUT_avg.min()/1000):.1f}", "MWp", "")
add("2 PV", "land at 1-2 ha/MWp, belt", f"{kmin*1.0:.0f}-{belt.PV_MWp_for_5MW_cont_LHV_avg.max()*2.0:.0f}", "ha", "0.2-0.5 km2")
add("2 PV", "ratio PV nameplate / continuous power", f"{kmin/ (P_LHV/1000):.1f}-{belt.PV_MWp_for_5MW_cont_LHV_avg.max()/(P_LHV/1000):.1f}", "-", "= 1/CF")

# ---------------- 3. operating modes
# (a) PV-following: electrolyser sized to PV AC peak; H2 buffer from intra-day mismatch. Two daily production profiles
def buffer_kg(profile, hours=24, dt=1/60):
    t = np.arange(0, hours, dt); p = profile(t); p = p / (p.sum() * dt) * (h2_kg_h * 24)   # kg/h, normalised to daily H2
    cum = np.cumsum((p - h2_kg_h) * dt); return cum.max() - min(0.0, cum.min()), p.max() / h2_kg_h
prof_sin2 = lambda t: np.where((t >= 6) & (t <= 18), np.sin(np.pi * (t - 6) / 12) ** 2, 0.0)   # clear-sky bell, 6 peak-sun-hours
prof_half = lambda t: np.where((t >= 6) & (t <= 18), np.sin(np.pi * (t - 6) / 12), 0.0)        # broader half-sine, 7.6 PSH (optimistic)
prof_rect = lambda t: np.where((t >= 9) & (t <= 15), 1.0, 0.0)                                  # 6 h block (pessimistic peakiness)
for name, pf in [("sin2 bell (6 PSH, 12 h daylight)", prof_sin2), ("half-sine (7.6 PSH)", prof_half), ("6 h rectangular", prof_rect)]:
    b, pk = buffer_kg(pf)
    add("3a PV-following", f"intra-day H2 buffer, {name}", b, "kg", f"peak/mean production {pk:.2f}")
    add("3a PV-following", f"electrolyser peak / continuous, {name}", pk * P_LHV, "kW", "electrolyser capacity if it follows PV")
add("3a PV-following", "cloudy-day autonomy, 1 day", h2_t_d * 1000, "kg", "adds one day of Stage 2 demand")
add("3a PV-following", "electrolyser capacity factor if sized to PV peak (sin2)", 0.25, "-", "= 6 PSH / 24 h")
add("3a PV-following", "H2 buffer volume at 30 bar / 200 bar (2 t)", f"{2000/2.4:.0f} / {2000/14.5:.0f}", "m3", "densities ~2.4 and ~14.5 kg/m3 at 300 K")
# (b) baseload
add("3b baseload", "electrolyser capacity", P_LHV / 0.95, "kW", "5 MW continuous at 95 % availability")
add("3b baseload", "Kenya geothermal installed capacity 2023 (IRENA 2025)", 985.0, "MW", "IRENA Renewable Capacity Statistics 2025, p.57")
add("3b baseload", "Kenya geothermal installed capacity 2024 (IRENA 2025)", 940.0, "MW", "same table; EPRA FY2023/24 reports 943.7 MW as at June 2024 (to be checked)")
add("3b baseload", "5 MW as share of Kenya geothermal (2024)", 5.0 / 940.0, "-", "")
add("4 annual", "annual electricity, 365 d continuous (LHV basis)", E_day_LHV * 365 / 1000.0, "GWh/yr", "process-model basis")
add("4 annual", "annual electricity, 330 operating days (LHV basis)", E_day_LHV * 330 / 1000.0, "GWh/yr", "= screening TEA basis (5,000 kW x 24 h x 330 d)")
add("4 annual", "annual H2, 365 d / 330 d", f"{h2_t_d*365:.0f} / {h2_t_d*330:.0f}", "t/yr", "")
add("4 annual", "annual electricity, 330 d, HHV basis", E_day_HHV * 330 / 1000.0, "GWh/yr", "")

# ---------------- 4. inventory of "5 MW" occurrences
inv = []
for path in [os.path.join(BASE, "Workflow_cantera.py"), os.path.join(BASE, "validation", "screening_tea.py"), os.path.join(BASE, "README.md")] + sorted(glob.glob(os.path.join(HERE, "*.py"))) + sorted(glob.glob(os.path.join(NOTES, "*.md"))):
    if os.path.basename(path).startswith("P4_"): continue
    try: lines = open(path, encoding="utf-8").read().splitlines()
    except Exception: continue
    for i, ln in enumerate(lines, 1):
        if re.search(r"5000\.?0?\b|5,000 kW|5 MW|solar_power_kW|share_of_5MW|onsite_electric_load", ln) and not re.search(r"max_steps|rtol|n_steps", ln):
            inv.append(dict(file=os.path.relpath(path, REPO), line=i, text=ln.strip()[:140]))
inv = pd.DataFrame(inv)

df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
with open(OUT_TXT, "w") as fh:
    fh.write("P4 solar basis — derived numbers\n"); fh.write(df.to_string(index=False) + "\n\n")
    fh.write("PVOUT by country (Solargis/World Bank 2020, Level 1, kWh/kWp/day)\n")
    fh.write(pv[["ISO_A3", "Country or region", "GHI_avg", "PVOUT_p10", "PVOUT_avg", "PVOUT_p90", "PVOUT_avg_kWh_kWp_yr", "capacity_factor_avg", "PV_MWp_for_5MW_cont_LHV_avg", "PV_MWp_for_5MW_cont_LHV_p90site"]].to_string(index=False, float_format=lambda v: f"{v:.3g}") + "\n\n")
    fh.write("Occurrences of 5 MW / 5,000 kW / solar_power_kW in code and notes\n"); fh.write(inv.to_string(index=False) + "\n")
print(open(OUT_TXT).read()); print("Saved:", OUT, OUT_PV, OUT_TXT)
