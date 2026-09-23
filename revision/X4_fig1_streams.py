"""
X4_fig1_streams.py -- full stream table for the X4 base configuration (new Fig. 1 and Section 2).

Case: configuration 2b (all biogas CO2 to Stage 2, no DAC, no electrolysis), y_CH4 = 0.60,
converged recycle (r_CH4 = 0.95 to Stage 1, purge p = 0.05, membrane H2 split h = 0.50), 1 atm.
  * EQUILIBRIUM  : Gibbs at every stage (F1 flowsheet, F1_cases.csv row "2b_y0.60 eq recycle h=0.5")
  * KINETIC      : design case tau1* = 8.905 s, Stage 2 two-reaction model phi = 1 (tau2 = 3 s),
                   Stage 3 submitted 3-reaction model (650 K, tau3 = 3 s) -- the case used by E7_TEA_X4.py
Nothing is re-derived: the flowsheet is F1_full_biogas_CO2.solve_recycle; this script only stores every
inter-stage stream (F1_cases.csv keeps aggregates only).

Outputs (revision/Result/):
  X4_fig1_streams.csv   long table: case, stream, species, kmol_d, t_d
  X4_fig1_stage_table.csv  per-stage inlet/outlet in t/d (CH4, CO2, CO, H2, H2O, other) + solids
  X4_fig1_scalars.csv   duties, solids, water, exportable H2, recycle flows, purge, element closures
  X4_fig1_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python X4_fig1_streams.py   (~5 min, two cases in parallel)
"""
import os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W
import F1_full_biogas_CO2 as F1m
from P1_recycle_analysis import FRESH_CH4, tpd, carbon, add

SPECIES = ["CH4", "CO2", "CO", "H2", "H2O"]
ATM = F1m.ATM

def cases():
    F = F1m.feeds(); fd = F["2b_y0.60"]
    common = dict(config="2b_y0.60", feed=fd, variant="V1", n=None, P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.50, p=0.05, r=0.95)
    return [("X4 equilibrium (2b, y=0.60, h=0.50, eq recycle)", dict(common, mode="equilibrium", phi=np.nan, k2=0.0)),
            ("X4 kinetic design case (2b, y=0.60, h=0.50, tau1*=8.9 s, phi=1, 1 atm)", dict(common, mode="kinetic", phi=1.0, k2=F1m.k2_from_phi(1.0)))]

def run(args):
    label, cfg = args; t0 = time.time()
    rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label); row["wall_s"] = time.time() - t0
    fd = cfg["feed"]
    streams = {
        "S1 fresh CH4 (waste)":            {"CH4": FRESH_CH4},
        "S1 recycled CH4 (from S3 outlet)": {"CH4": rec["ch4_rec"]},
        "S1 inlet":                        {"CH4": FRESH_CH4 + rec["ch4_rec"]},
        "S1 outlet gas":                   rec["gas1"],
        "S2 fresh CO2 (biogas)":           {"CO2": fd["co2_bio"] + fd["co2_dac"]},
        "S2 recycle from S3 outlet":       rec["to_s2"],
        "S2 inlet":                        rec["feed2"],
        "S2 outlet gas":                   rec["gas2"],
        "S2 condensed water":              rec["water2"],
        "S2 dry gas to membrane":          rec["pw2"],
        "membrane permeate (H2 export)":   rec["permeate"],
        "S3 inlet (membrane retentate)":   rec["cfr_feed"],
        "S3 outlet gas":                   rec["gas3"],
        "S3 condensed water":              rec["water3"],
        "S3 dry gas":                      rec["pw3"],
        "purge":                           rec["purge"],
    }
    solids = {"S1 solid carbon": rec["C1"], "S3 solid carbon": rec["C3"]}
    return label, row, streams, solids, rec, info

def main():
    t0 = time.time()
    with Pool(2) as pool: res = pool.map(run, cases())
    long, stage, scal = [], [], []
    for label, row, streams, solids, rec, info in res:
        basis = "kinetic" if "kinetic" in label else "equilibrium"
        for name, d in streams.items():
            tot = sum(d.values())
            for sp in SPECIES:
                long.append(dict(case=label, value_type=basis, stream=name, species=sp, kmol_d=d.get(sp, 0.0), t_d=tpd(sp, d.get(sp, 0.0))))
            other = {k: v for k, v in d.items() if k not in SPECIES}
            long.append(dict(case=label, value_type=basis, stream=name, species="other", kmol_d=sum(other.values()),
                             t_d=sum(v * W.MW.get(k, np.nan) / 1000.0 for k, v in other.items()) if other else 0.0))
            long.append(dict(case=label, value_type=basis, stream=name, species="total", kmol_d=tot, t_d=sum(v * W.MW.get(k, np.nan) / 1000.0 for k, v in d.items())))
            stage.append(dict(case=label, value_type=basis, stream=name, total_kmol_d=tot,
                              **{f"{sp}_tpd": tpd(sp, d.get(sp, 0.0)) for sp in SPECIES},
                              other_tpd=sum(v * W.MW.get(k, np.nan) / 1000.0 for k, v in other.items()) if other else 0.0,
                              other_species=";".join(f"{k}={v:.3g}" for k, v in sorted(other.items(), key=lambda kv: -kv[1])[:4])))
        for name, c in solids.items():
            long.append(dict(case=label, value_type=basis, stream=name, species="C(s)", kmol_d=c, t_d=tpd("C(s)", c)))
        purge = rec["purge"]
        scal.append(dict(case=label, value_type=basis, converged=info["converged"], iterations=info["iterations"], method=info["method"], tear_err=info["err"],
                         Q1_kW=rec["Q1"], Q2_kW=rec["Q2"], Q3_kW=rec["Q3"],
                         C_stage1_tpd=tpd("C(s)", rec["C1"]), C_stage3_tpd=tpd("C(s)", rec["C3"]), C_total_tpd=tpd("C(s)", rec["C1"] + rec["C3"]),
                         C_from_biogasCO2_tpd=row["C_from_biogasCO2_tpd"], CO2_carbon_fixed_fraction=row["CO2_carbon_fixed_fraction"],
                         water_S2_tpd=tpd("H2O", rec["water2"].get("H2O", 0)), water_S3_tpd=tpd("H2O", rec["water3"].get("H2O", 0)), water_total_tpd=row["water_total_tpd"],
                         H2_permeate_tpd=row["H2_permeate_tpd"], H2_in_purge_tpd=tpd("H2", purge.get("H2", 0)), H2_net_exportable_tpd=row["H2_net_exportable_tpd"],
                         recycle_CH4_to_S1_kmol_d=rec["ch4_rec"], recycle_to_S2_kmol_d=sum(rec["to_s2"].values()), S3_dry_gas_kmol_d=sum(rec["pw3"].values()),
                         S1_inlet_CH4_ratio_vs_fresh=row["S1_inlet_CH4_ratio_vs_fresh"], S1_X_CH4=row["S1_X_CH4"], S2_X_CO2_per_pass=row["S2_X_CO2_per_pass"],
                         purge_total_kmol_d=sum(purge.values()), purge_CO2_tpd=row["purge_CO2_tpd"], purge_CO_tpd=row["purge_CO_tpd"], purge_CH4_tpd=row["purge_CH4_tpd"],
                         purge_H2_tpd=tpd("H2", purge.get("H2", 0)), purge_carbon_kmol_d=carbon(purge), purge_carbon_as_CO2_if_oxidised_tpd=row["purge_carbon_as_CO2_if_oxidised_tpd"],
                         closure_C_rel=row["closure_C_rel"], closure_H_rel=row["closure_H_rel"], closure_O_rel=row["closure_O_rel"], wall_s=row["wall_s"]))
    L_, S_, C_ = pd.DataFrame(long), pd.DataFrame(stage), pd.DataFrame(scal)
    L_.to_csv(os.path.join(RESULT, "X4_fig1_streams.csv"), index=False)
    S_.to_csv(os.path.join(RESULT, "X4_fig1_stage_table.csv"), index=False)
    C_.to_csv(os.path.join(RESULT, "X4_fig1_scalars.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    lines = [f"X4 base-configuration stream table for Fig. 1 (wall {time.time()-t0:.0f} s). Values labelled equilibrium / kinetic.",
             "Kinetic = design case tau1* = 8.905 s, phi = 1, 1 atm (uncalibrated reduced kinetics; same case as E7 TEA).", ""]
    for label, *_ in res:
        lines += [f"=== {label}", "--- stage table [t/d]",
                  S_[S_.case == label].drop(columns=["case", "value_type"]).to_string(index=False, float_format=lambda x: f"{x:,.2f}"), "",
                  "--- scalars", C_[C_.case == label].drop(columns=["case"]).T.to_string(header=False), ""]
    open(os.path.join(RESULT, "X4_fig1_summary.txt"), "w").write("\n".join(lines)); print("\n".join(lines))

if __name__ == "__main__":
    main()
