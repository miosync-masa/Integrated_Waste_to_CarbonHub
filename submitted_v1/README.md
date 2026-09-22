# Integrated Waste-to-Solid-Carbon System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15_kDIGbXsrWM8ixbpGA8rqWov2XR4zH7)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Companion code for:**  
*Mass-Balance-Consistent Carbon Management: Thermodynamic Design of an Integrated Waste-to-Solid-Carbon System*  
M. Iizumi, Miosync Inc.

---

## Overview

This repository contains two self-contained Python scripts that reproduce all computational results reported in the paper and Supplementary Information.

| Script | Purpose |
|---|---|
| `workflow_cantera.py` | Three-stage thermochemical workflow (pyrolysis → RWGS → CFR) with Gibbs equilibrium and exploratory kinetic models |
| `screening_tea.py` | Screening-level techno-economic assessment (3 scenarios, sensitivity analysis) |

## Requirements

```
Python >= 3.9
cantera >= 2.6
numpy
pandas
matplotlib
```

Install:
```bash
pip install cantera numpy pandas matplotlib
```

## Quick Start

```bash
python workflow_cantera.py
python screening_tea.py
```

Each script prints results to stdout and saves CSV data files + PNG figures to the working directory.

## workflow_cantera.py

### What it computes

**Gibbs equilibrium baseline** (used for all main-text claims):
- Stage 1: CH₄ pyrolysis at 1200 K → solid carbon + H₂
- Stage 2: RWGS at 950 K → CO + H₂O, with water condensation and membrane separation
- Stage 3: Carbon formation reaction at 650 K (Gibbs minimisation, gas + graphite phases)
- Elemental balance verification (C, H, O closure < 10⁻¹⁴ relative error)

**Exploratory kinetic model** (SI S3 — not used for main-text claims):
- All three stages replaced with finite-residence-time PFR approximations
- Stage 3 uses 3 independent reversible reactions (CO₂ methanation, CO carbon deposition, CH₄ cracking)
- Rate constants are **uncalibrated fitting parameters** — see paper SI S3 for details

**Parametric sweeps:**
- CFR: temperature × water removal × H₂ fraction (240 cases)
- CFR carbon window: temperature × catalyst selectivity multiplier (48 cases)
- RWGS: temperature × residence time (60 cases)
- Pyrolysis: temperature × residence time (60 cases)

**Validation-ready series:**
- Stage 1: CH₄ conversion vs τ at 1200 K
- Stage 2: CO₂ conversion vs τ at 950 K
- Stage 3: Carbon deposition + gas composition vs T at 750–850 K

### Switching between equilibrium and kinetic modes

The default `run_baseline()` uses kinetic versions of all stages. To revert to Gibbs equilibrium (as used in the main text), replace in `run_baseline()`:

```python
# Kinetic (default in code):
s1 = run_stage1_pyrolysis_kinetic(...)
s2 = run_stage2_rwgs_kinetic(...)
s3 = run_stage3_cfr_kinetic(...)

# Equilibrium (as in main text):
s1 = run_stage1_pyrolysis(CH4_tpd, T_pyro, P_pyro)
s2 = run_stage2_rwgs(rf, T_rwgs, P_rwgs)
s3 = run_stage3_cfr(sep["cfr_feed"], T_cfr, P_cfr)
```

### Key outputs

| File | Contents |
|---|---|
| `workflow_summary.csv` | Baseline material/energy balance |
| `balance_stage{1,2,3}.csv` | Elemental balance verification |
| `workflow_sweep.csv` | CFR parametric sweep (240 cases) |
| `cfr_carbon_window.csv` | Carbon deposition window map |
| `rwgs_tau_sweep.csv` | RWGS τ × T sweep |
| `pyro_tau_sweep.csv` | Pyrolysis τ × T sweep |
| `validation_stage{1,2,3}_*.csv` | Validation-ready series |
| `heatmap_*.png` | Visualisations |

## screening_tea.py

### What it computes

Three economic scenarios (Conservative / Base / Opportunity) evaluated under two paths:
- **Carbon product path**: solid carbon sales only
- **Carbon fixation path**: + carbonate mineralisation + carbon credits

Sensitivity analysis: ±30% perturbation of each parameter around Base scenario.

### Key outputs

| File | Contents |
|---|---|
| `screening_tea_summary.csv` | Annual value and payback for all scenarios |
| `sensitivity_product.csv` | Sensitivity analysis (product path) |
| `sensitivity_fixation.csv` | Sensitivity analysis (fixation path) |
| `tornado_*.png` | Tornado charts |

## Citation

If you use this code, please cite the accompanying paper (submitted to Resources, Conservation and Recycling).

## License

MIT
