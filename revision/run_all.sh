#!/bin/bash
# Re-run every revision script in dependency order. Run from anywhere:  bash revision/run_all.sh
# Total wall time on a 12-core laptop: about 80 min (P1 ~30 min, F1 ~13 min, T3b ~7 min, E7 ~2 min on first run).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../.venv/bin/python}"
# On Apple silicon make sure the interpreter runs natively (a shell started under Rosetta would otherwise load the arm64 wheels as x86_64 and fail).
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then PY="arch -arm64 $PY"; fi
run() { echo "== $1  ($(date +%H:%M:%S))"; $PY "$1" > "Result/${1%.py}.log" 2>&1 || { echo "FAILED: $1 (see Result/${1%.py}.log)"; exit 1; }; }
run K3_stage2_extent_decomposition.py      # K3: Stage 2 extent decomposition and heat apportionment
run K3_stage2_two_reaction_kinetic.py      # K3: two-reaction Stage 2 reduced model, tau sweep
run KIN_chain_design_case.py               # KIN: Stage 1 design residence time tau* and kinetic chain
run P1_recycle_analysis.py                 # P1/P2: recycle analysis with carbon-origin tracking (long)
run P1_recycle_plot.py
run T3_pressure_analysis.py                # T3: pressure effects, per-stage
run T3b_pressure_order_sensitivity.py      # T3: pressure-order sensitivity of the kinetic result
run T1_carbon_form_sensitivity.py          # T1: Gibbs offset of the solid-carbon phase
run P4_solar_basis.py                      # P4: basis of the 5 MW electrolysis figure
run E9_gate_fee_scenarios.py               # E9: gate-fee basis
run F1_full_biogas_CO2.py                  # F1: all biogas CO2, no DAC (X3/X4)
run F1b_heat_H2_balance.py                 # F1b: heat and H2 balance
run X4_fig1_streams.py                     # X4: full stream table for the new Fig. 1 (eq + kinetic design case)
run X4_sensitivity_dG_pressure.py          # X4: once-through reference, carbon-form and pressure sensitivities
run X4_fig2_fixation.py                    # X4: Fig. 2, CO2-carbon fixation bars (eq once-through / eq recycle / kinetic)
run P3_auxiliary_power.py                  # P3: auxiliary power
run T4_heat_cascade.py                     # T4: pinch analysis
run T4b_surplus_heat_use.py                # T4b: use of the 650 K surplus heat
run X4_exergy_heat.py                      # X4: heat exergy of the T4 streams (independent check of the pinch conclusion)
run E7_TEA_X4.py                           # E7/E11: screening TEA
run K2_validation_design.py                # K2: identifiability and validation design
echo "== all done ($(date +%H:%M:%S))"
