# Revision analyses (Major Revision of FUPROC-D-26-00269)

This folder contains the additional calculations made for the Major Revision of
*Mass-Balance-Consistent Carbon Management: Thermodynamic and Kinetic Design of an Integrated Waste-to-Solid-Carbon System*.
The submitted code and its outputs are kept unchanged in [`../submitted_v1/`](../submitted_v1/); every script here imports
`Workflow_cantera.py` from there (through `_paths.py`) and writes its outputs to [`Result/`](Result/).

**Value types.** Every table states whether a number is an **equilibrium** value (Gibbs minimisation, GRI-3.0 + graphite)
or a **kinetic** value (reduced models with **uncalibrated** rate constants). The two are never mixed.

## Layout

| Path | Content |
|---|---|
| `*.py` | 17 self-contained scripts (flat, same style as `submitted_v1/validation/`); each docstring states purpose, assumptions, sources and outputs |
| `_paths.py` | shared paths (`BASE` = `submitted_v1/`, `RESULT` = `Result/`) |
| `run_all.sh` | re-runs everything in dependency order (about 80 min on a 12-core laptop) |
| `Result/` | all CSV / TXT / PNG outputs, plus one `.log` per script from the last `run_all.sh` |
| `SOURCES.md`, `SOURCES.sha256` | third-party documents used (URLs, checksums); the files themselves are not committed |

## Scripts, reviewer comments and outputs

| ID | Script | Reviewer comment(s) | Question answered | Main outputs |
|---|---|---|---|---|
| K3 | `K3_stage2_extent_decomposition.py`, `K3_stage2_two_reaction_kinetic.py` | R1-3, R2-4, R4-2 | Contribution of CO2 methanation in Stage 2; apportionment of the −196 kW duty; two-reaction Stage 2 model | `K3_*.csv`, `K3_stage2_2rxn_tau_sweep.png` |
| KIN | `KIN_chain_design_case.py` | K5, R1-5, m2, m8 | Stage 1 design residence time (approach ≥ 0.90 → τ* = 8.9 s) and the full kinetic chain | `KIN_chain_*.csv` |
| P1/P2 | `P1_recycle_analysis.py`, `P1_recycle_plot.py` | R1-5 | Converged recycle with species-resolved carbon-origin tracking; H2 split | `P1_recycle_*.csv`, `P1_recycle_co2_carbon.png` |
| T3 | `T3_pressure_analysis.py`, `T3b_pressure_order_sensitivity.py` | R1-2 | Pressure effect per stage; compression duty; pressure-order sensitivity of the kinetic result | `T3_*.csv`, `T3b_*.csv` |
| T1 | `T1_carbon_form_sensitivity.py` | R1-2 | Gibbs-energy offset of the solid-carbon phase (0–20 kJ/mol) | `T1_carbon_form_sensitivity.csv` |
| P4 | `P4_solar_basis.py` | R2-3 | Basis of the 5 MW electrolysis figure; PV capacity and land for the equatorial dry belt | `P4_*.csv` |
| E9 | `E9_gate_fee_scenarios.py` | R1-6, R2-6 | Gate-fee basis (chargeable share × fee), East-African fee evidence | `E9_*.csv` |
| F1 | `F1_full_biogas_CO2.py`, `F1b_heat_H2_balance.py` | R1-1, R1-7 | All biogas CO2 treated, no DAC, with/without electrolysis; hydrogen and heat self-sufficiency | `F1_cases.csv`, `F1b_*.csv` |
| P3 | `P3_auxiliary_power.py` | R1-5, R2-5 | Auxiliary power (membranes, separation, blowers, cooling, solids) and self-sufficiency map | `P3_*.csv` |
| T4 | `T4_heat_cascade.py`, `T4b_surplus_heat_use.py` | R2-2 | Pinch analysis (composite and grand composite curves); use of the 650 K surplus heat | `T4_*.csv`, `T4_composite_curves.png`, `T4b_*.csv` |
| E7/E11 | `E7_TEA_X4.py` | R1-6, R2-7, R4-4 | Screening TEA (AACE Class 5) of the revised configuration: CAPEX build-up, NPV/IRR, tornado, break-even | `E7_*.csv`, `E7_tornado.png` |
| K2 | `K2_validation_design.py` | R2-1 | Identifiability (Fisher information) and a minimal multi-temperature validation design | `K2_*.csv` |

Dependency order: K3 → KIN → P1 → T3 → T3b → T1 → P4 → E9 → F1 → F1b → P3 → T4 → T4b → E7 → K2 (as in `run_all.sh`).

## Environment

Tested with Python 3.12, Cantera 3.2.0, NumPy 2.5, SciPy 1.18, pandas, Matplotlib, openpyxl (see `../Requirements.txt`).

```bash
cd <repo>
python -m venv .venv && .venv/bin/pip install -r Requirements.txt
bash revision/run_all.sh          # or: cd revision && ../.venv/bin/python <script>.py
```

`P4_solar_basis.py` needs the Solargis workbook listed in `SOURCES.md` in `revision/sources/`.
