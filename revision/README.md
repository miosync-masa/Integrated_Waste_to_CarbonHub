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
| `*.py` | 26 self-contained scripts (flat, same style as `submitted_v1/validation/`); each docstring states purpose, assumptions, sources and outputs |
| `_paths.py` | shared paths (`BASE` = `submitted_v1/`, `RESULT` = `Result/`) |
| `run_all.sh` | re-runs everything in dependency order (about 90 min on a 12-core laptop; the kinetic recycle solves dominate) |
| `Result/` | all CSV / TXT / PNG outputs, plus one `.log` per script from the last `run_all.sh` |
| `SOURCES.md`, `SOURCES.sha256` | third-party documents used (URLs, checksums); the files themselves are not committed |

## Base configuration used in the revision (X4)

Scripts whose names start with `X4_`, and the TEA (`E7_TEA_X4.py`), refer to the revised base configuration fixed during the revision:

| Item | Value |
|---|---|
| Feed | waste CH4 48 t/d (as submitted) + **all** CO2 of a 60 vol% CH4 biogas (87.8 t/d CO2, 24.0 t C/d); no DAC, no electrolysis (configuration "2b", y_CH4 = 0.60) |
| Stage temperatures / pressure | 1200 / 950 / 650 K, 1 atm (Stage 3 = carbon recovery stage: methanation with recycle, or direct deposition, depending on h) |
| Recycle | Stage 3 outlet: water removal 0.95 → CH4 separation r_CH4 = 0.95 back to Stage 1 → purge p = 0.05 → rest to Stage 2; membrane H2 split to Stage 3 h = 0.50 |
| Kinetic design case | Stage 1 τ1* = 8.905 s (approach 0.90), Stage 2 two-reaction model φ = 1 (τ2 = 3 s), Stage 3 reduced 3-reaction model (τ3 = 3 s); all rate constants uncalibrated |
| Representative numbers (equilibrium) | solid carbon 58.6 t/d (Stage 1 37.7, Stage 3 21.0), 94.8 % of the biogas-CO2 carbon fixed, Q1/Q2/Q3 = +3,275 / +842 / −3,171 kW, purge CO2 4.5 t/d (`X4_fig1_summary.txt`) |

The equilibrium loop is the design basis; the kinetic loop "as modelled" circulates 40,700 kmol/d of CO-rich gas and is reported for information only (see Notes).

## Scripts, reviewer comments and outputs

Rows follow the execution order of `run_all.sh`.

| ID | Script | Reviewer comment(s) | Question answered | Main outputs |
|---|---|---|---|---|
| K3 | `K3_stage2_extent_decomposition.py`, `K3_stage2_two_reaction_kinetic.py` | R1-3, R2-4, R4-2 | Contribution of CO2 methanation in Stage 2; apportionment of the −196 kW duty; two-reaction Stage 2 model | `K3_*.csv`, `K3_stage2_2rxn_tau_sweep.png` |
| KIN | `KIN_chain_design_case.py` | K5, R1-5, m2, m8 | Stage 1 design residence time (approach ≥ 0.90 → τ* = 8.9 s) and the full kinetic chain | `KIN_chain_*.csv` |
| R4 Major 3 | `KIN_cfr_window_origin.py` | R4-3 | Origin (waste CH4 vs CO2) of the Stage 3 solid carbon in the submitted kinetic chain across the 750–850 K window, species-resolved tracer and reaction-resolved attribution, tau1 = 3 s and tau1* | `KIN_cfr_window_origin.csv`, `KIN_cfr_window_origin_summary.txt` |
| P1/P2 | `P1_recycle_analysis.py`, `P1_recycle_plot.py` | R1-5 | Converged recycle with species-resolved carbon-origin tracking; H2 split | `P1_recycle_*.csv`, `P1_recycle_co2_carbon.png` |
| T3 | `T3_pressure_analysis.py`, `T3b_pressure_order_sensitivity.py` | R1-2 | Pressure effect per stage; compression duty; pressure-order sensitivity of the kinetic result | `T3_*.csv`, `T3b_*.csv` |
| T1 | `T1_carbon_form_sensitivity.py` | R1-2 | Gibbs-energy offset of the solid-carbon phase (0–20 kJ/mol) | `T1_carbon_form_sensitivity.csv` |
| P4 | `P4_solar_basis.py` | R2-3 | Basis of the 5 MW electrolysis figure; PV capacity and land for the equatorial dry belt | `P4_*.csv` |
| E9 | `E9_gate_fee_scenarios.py` | R1-6, R2-6 | Gate-fee basis (chargeable share × fee), East-African fee evidence | `E9_*.csv` |
| F1 | `F1_full_biogas_CO2.py`, `F1b_heat_H2_balance.py` | R1-1, R1-7 | All biogas CO2 treated, no DAC, with/without electrolysis; hydrogen and heat self-sufficiency | `F1_cases.csv`, `F1b_*.csv` |
| Fig. 1 / §2 | `X4_fig1_streams.py` | R4 m8, X4 | Every inter-stage stream, duty, solid, water and purge for the X4 base configuration, equilibrium and kinetic design case side by side | `X4_fig1_*.csv`, `X4_fig1_summary.txt` |
| §3 (T1/T3 on X4) | `X4_sensitivity_dG_pressure.py` | R1-2, X4 | Once-through reference for the X4 feed, carbon-phase Gibbs offset (0–20 kJ/mol) and uniform pressure (1 atm–20 bar) with converged recycle | `X4S_*.csv`, `X4S_summary.txt` |
| §3 (T3 temperature) | `X4_T3_sensitivity.py` | R4-3, X4 | Stage 3 temperature 650–850 K with converged recycle, equilibrium and kinetic design case: solids, CO2-carbon fixation (tracer), duties, purge | `X4_T3_sensitivity.csv`, `X4_T3_sensitivity_summary.txt` |
| Fig. 2 | `X4_fig2_fixation.py` | R1-5, R4 m8 | Fixation of biogas-CO2 carbon vs membrane split h: once-through and recycle at equilibrium, kinetic design case at 1 atm and 5 bar (uncalibrated) | `X4_fig2_fixation.png/.pdf`, `X4_fig2_data.csv` |
| P3 | `P3_auxiliary_power.py` | R1-5, R2-5 | Auxiliary power (membranes, separation, blowers, cooling, solids) and self-sufficiency map | `P3_*.csv` |
| T4 | `T4_heat_cascade.py`, `T4b_surplus_heat_use.py` | R2-2 | Pinch analysis (composite and grand composite curves); use of the 650 K surplus heat | `T4_*.csv`, `T4_composite_curves.png`, `T4b_*.csv` |
| §5.2 | `X4_exergy_heat.py` | R2-2 | Heat exergy (T0 = 298.15 K) of the 13 T4 streams of the X4 case, log-mean and curve-integral values, condensing streams split sensible/latent, ORC exergy efficiency | `X4_exergy_heat_*.csv`, `X4_exergy_heat_summary.txt` |
| E7/E11 | `E7_TEA_X4.py` | R1-6, R2-7, R4-4 | Screening TEA (AACE Class 5) of the revised configuration: CAPEX build-up, NPV/IRR, tornado, break-even | `E7_*.csv`, `E7_tornado.png` |
| K2 | `K2_validation_design.py` | R2-1 | Identifiability (Fisher information) and a minimal multi-temperature validation design | `K2_*.csv` |
| K2b | `K2b_stage3_design_600_850.py` | R2-1 | Intermediate step: Stage 3 validation temperatures chosen by exhaustive D-optimal selection over a 600–850 K candidate set, submitted-case inlet (K2 parameter set and the full three-reaction set) | `K2b_*.csv`, `K2b_summary.txt` |
| K2c | `K2c_design_X4_inlets.py` | R2-1 | Intermediate step: K2b repeated with the X4 Stage 3 inlets (kinetic loop and equilibrium loop) and D-optimal Stage 2 temperatures (850–1050 K) with the X4 Stage 2 inlet; shows that the process inlet compositions are unsuitable as calibration feeds | `K2c_*.csv`, `K2c_summary.txt` |
| K2d | `K2d_design_composition.py` | R2-1 | **Recommended validation programme (supersedes K2b/K2c).** Feed composition as a design variable: two synthetic-feed series per stage, D-optimal and minimax over temperatures and H2/CO2 (Stage 3 also CO/CO2) | `K2d_*.csv`, `K2d_summary.txt` |

Dependency order (as in `run_all.sh`): K3 → KIN → KIN_cfr_window_origin → P1 → T3 → T3b → T1 → P4 → E9 → F1 → F1b → X4_fig1 → X4_sensitivity_dG_pressure → X4_T3_sensitivity → X4_fig2 → P3 → T4 → T4b → X4_exergy_heat → E7 → K2 → K2b → K2c → K2d.
Scripts that need a previous result read it from `Result/` (for example `X4_exergy_heat.py` uses the streams written by `X4_fig1_streams.py`; `K2c`/`K2d` import `K2`/`K2b`). Two scripts cache expensive kinetic solves (`E7_kinetic_X4.csv`, `X4_fig2_kinetic_cases.csv`); delete the cache file to force a re-solve.

## Notes

* **Kinetic loop "as modelled".** With the uncalibrated reduced kinetics the converged recycle of the kinetic design case circulates about 40,700 kmol/d of CO-rich gas (Stage 3 inlet H2/CO2 = 0.42). Its yields and duties are reported (`X4_fig1_*`, `X4_T3_sensitivity.csv`), but the TEA sizes the plant on the equilibrium loop and `K2c` shows that this composition is unsuitable as a calibration feed.
* **CO-free feeds freeze the reduced Stage 3 model.** `run_stage3_cfr_kinetic` in the submitted code floors every activity at `TRACE`, so with exactly zero CO the CO-hydrogenation step has a small positive rate while the CO inventory is zero; the negativity limiter then sets the whole integration step to zero and no reaction proceeds. `K2d` therefore uses a 1 % CO trace for its CO-lean series. This does not affect any process case (CO is always present after Stage 2).
* **Stage 3 temperature.** At equilibrium, fixation decreases as T3 is raised above 650 K; with the uncalibrated reduced kinetics, fixation rises to 53–76 % at 800–850 K (`X4_T3_sensitivity.csv`). The design temperature cannot be chosen until the kinetics are calibrated, which is why the validation temperatures were re-selected over 600–850 K (`K2b`–`K2d`).
* **Logs.** `Result/*.log` are the stdout of the last full run; the local repository path is replaced by `<repo>`.

## Environment

Tested with Python 3.12, Cantera 3.2.0, NumPy 2.5, SciPy 1.18, pandas, Matplotlib, openpyxl (see `../Requirements.txt`).

```bash
cd <repo>
python -m venv .venv && .venv/bin/pip install -r Requirements.txt
bash revision/run_all.sh          # or: cd revision && ../.venv/bin/python <script>.py
```

`P4_solar_basis.py` needs the Solargis workbook listed in `SOURCES.md` in `revision/sources/`.
