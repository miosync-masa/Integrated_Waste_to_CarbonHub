"""
X4_T3_sensitivity.py -- Stage 3 temperature sensitivity of the X4 base configuration with converged recycle.

X4: 2b (all biogas CO2 to Stage 2, no DAC, no electrolysis), y_CH4 = 0.60, h = 0.50, r_CH4 = 0.95, p = 0.05, 1 atm.
T3 in {650 (base), 700, 750, 800, 850} K for
  * EQUILIBRIUM  : Gibbs at every stage (F1 flowsheet, stage3_eq_robust bound to T3)
  * KINETIC      : design case tau1* = 8.905 s, Stage 2 two-reaction model phi = 1, Stage 3 reduced 3-reaction
                   model at T3 (Arrhenius from Tref = 650 K; uncalibrated), tau3 = 3 s
Everything else is F1_full_biogas_CO2 (one_pass / solve_recycle / make_row incl. species-resolved tracer).

Outputs (revision/Result/): X4_T3_sensitivity.csv (F1 make_row columns + S3 inlet/outlet extras), X4_T3_sensitivity_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python X4_T3_sensitivity.py   (~5 min, 10 cases in parallel)
"""
import os, time, functools
import numpy as np, pandas as pd
from multiprocessing import Pool
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import F1_full_biogas_CO2 as F1m
import T3b_pressure_order_sensitivity as T3b
from P1_recycle_analysis import tpd, carbon

ATM = F1m.ATM
T3_GRID = [650.0, 700.0, 750.0, 800.0, 850.0]

def run(args):
    mode, T3 = args; t0 = time.time()
    F1m.stage3_eq_robust = functools.partial(F1m.stage3_eq_robust, T=T3)   # equilibrium Stage 3 at T3
    T3b.T3 = T3                                                            # kinetic Stage 3 at T3
    fd = F1m.feeds()["2b_y0.60"]
    cfg = dict(config="2b_y0.60", feed=fd, mode=mode, variant="V1", n=None, P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.50, p=0.05, r=0.95,
               phi=(1.0 if mode == "kinetic" else np.nan), k2=(F1m.k2_from_phi(1.0) if mode == "kinetic" else 0.0))
    label = f"X4 {mode} recycle T3={T3:g} K" + (" (tau1*=8.9 s, phi=1)" if mode == "kinetic" else "")
    try:
        rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label)
        cf, g3 = rec["cfr_feed"], rec["gas3"]
        row.update(T3_K=T3, tear_err=info["err"], wall_s=time.time() - t0, error="",
                   S3_inlet_H2_to_CO2=cf.get("H2", 0) / max(cf.get("CO2", 0), 1e-12), S3_inlet_CO_kmol_d=cf.get("CO", 0), S3_inlet_CO2_kmol_d=cf.get("CO2", 0),
                   S3_inlet_H2_kmol_d=cf.get("H2", 0), S3_inlet_CH4_kmol_d=cf.get("CH4", 0),
                   S3_dCH4_kmol_d=g3.get("CH4", 0) - cf.get("CH4", 0), S3_dCO_kmol_d=g3.get("CO", 0) - cf.get("CO", 0), S3_dCO2_kmol_d=g3.get("CO2", 0) - cf.get("CO2", 0),
                   S3_out_gas_carbon_tpd=tpd("C(s)", carbon(g3)), recycle_CH4_to_S1_kmol_d=rec["ch4_rec"], purge_H2_tpd=tpd("H2", rec["purge"].get("H2", 0)))
        return row
    except Exception as ex:
        return dict(case=label, mode=mode, value_type=mode, T3_K=T3, converged=False, error=str(ex)[:300], wall_s=time.time() - t0)

def main():
    t0 = time.time(); jobs = [(m, T) for m in ("equilibrium", "kinetic") for T in T3_GRID]
    with Pool(len(jobs)) as pool: rows = pool.map(run, jobs)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESULT, "X4_T3_sensitivity.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    ff = lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.1f}"
    cols = ["T3_K", "converged", "iterations", "C_stage1_tpd", "C_stage3_tpd", "C_total_tpd", "C_from_biogasCO2_tpd", "CO2_carbon_fixed_fraction",
            "Q1_kW", "Q2_kW", "Q3_kW", "purge_CO2_tpd", "purge_CO_tpd", "purge_CH4_tpd", "purge_carbon_as_CO2_if_oxidised_tpd",
            "H2_net_exportable_tpd", "water_total_tpd", "S1_inlet_CH4_ratio_vs_fresh", "S3_inlet_H2_to_CO2", "S3_dCH4_kmol_d", "S3_dCO_kmol_d", "S3_dCO2_kmol_d", "recycle_to_S2_total_kmol_d"]
    L = [f"X4 Stage 3 temperature sensitivity (wall {time.time()-t0:.0f} s). X4 = 2b, y = 0.60, h = 0.50, r_CH4 = 0.95, p = 0.05, 1 atm.",
         "EQUILIBRIUM rows: Gibbs at every stage. KINETIC rows: tau1* = 8.905 s, phi = 1, Stage 3 reduced model (uncalibrated).",
         f"Unconverged / failed: {int((~D.converged.astype(bool)).sum())}", ""]
    for m in ("equilibrium", "kinetic"):
        sub = D[D["mode"] == m] if "mode" in D else D[D.value_type == m]
        L += [f"=== {m}", sub[[c for c in cols if c in sub.columns]].to_string(index=False, float_format=ff), ""]
    if (~D.converged.astype(bool)).any(): L += ["Failures:", D[~D.converged.astype(bool)][["case", "error"]].to_string(index=False)]
    open(os.path.join(RESULT, "X4_T3_sensitivity_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
