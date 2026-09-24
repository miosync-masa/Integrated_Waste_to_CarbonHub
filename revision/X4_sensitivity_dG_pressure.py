"""
X4_sensitivity_dG_pressure.py -- once-through reference, carbon-form and pressure sensitivities
recomputed for the X4 base configuration (Section 3 of the revised manuscript).

X4: configuration 2b (all biogas CO2 to Stage 2, no DAC, no electrolysis), y_CH4 = 0.60, h = 0.50,
recycle r_CH4 = 0.95 / purge p = 0.05, all stages Gibbs (EQUILIBRIUM values throughout).
The flowsheet is F1_full_biogas_CO2 (one_pass / solve_recycle / make_row, species-resolved tracer);
the carbon phase with a Gibbs offset is T1_carbon_form_sensitivity.make_carbon (dH = dG, dS = 0).

Parts
  A  once-through (p = 1, r = 0) at 1 atm, graphite: full inter-stage stream table -> the reference
     against which the converged recycle (X4_fig1_streams.py) is compared.
  B  carbon form: dG in {0, 5, 10, 20} kJ/mol above graphite, 1 atm, once-through and recycle.
     Stage 1, Stage 3 and the stream enthalpies all use the shifted phase (F1's robust Stage 3 solver
     is re-bound to the shifted solid; everything else unchanged).
  C  pressure: all stages at 1 atm / 5 / 10 / 20 bar, graphite, once-through and recycle.

Outputs (revision/Result/)
  X4S_cases.csv               one row per (part, dG, P, once-through|recycle): F1 make_row columns
  X4S_once_through_streams.csv  Part A stage table [kmol/d, t/d]
  X4S_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python X4_sensitivity_dG_pressure.py   (~5 min, parallel)
"""
import os, time
import numpy as np, pandas as pd, cantera as ct
from multiprocessing import Pool
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W
import F1_full_biogas_CO2 as F1m
import T1_carbon_form_sensitivity as T1m
from P1_recycle_analysis import FRESH_CH4, tpd, carbon

ATM, BAR = F1m.ATM, F1m.BAR
SPECIES = ["CH4", "CO2", "CO", "H2", "H2O"]
DG_GRID = [0.0, 5.0, 10.0, 20.0]
P_GRID = [ATM, 5 * BAR, 10 * BAR, 20 * BAR]

def bind_solid(dG):
    """Route every graphite call of the flowsheet through the shifted carbon phase."""
    T1m.patch_solid(dG)                       # W.equilibrate_gas_plus_graphite, W.total_stream_enthalpy_J_per_day, W.reaction_Kp
    base = F1m.stage3_eq_robust.__wrapped__ if hasattr(F1m.stage3_eq_robust, "__wrapped__") else F1m.stage3_eq_robust
    def stage3(feed, P, T=F1m.T3K):
        g = ct.Solution("gri30.yaml"); sld = T1m.make_carbon(dG)
        x, n = W.normalize_mole_dict(feed); g.TPX = T, P, x
        tries = [dict(solver="gibbs", rtol=1e-6, max_steps=200000, max_iter=100, estimate_equil=-1, log_level=0),
                 dict(solver="gibbs", rtol=1e-6, max_steps=200000, max_iter=200, estimate_equil=0, log_level=0),
                 dict(solver="gibbs", rtol=1e-6, max_steps=200000, max_iter=200, estimate_equil=1, log_level=0),
                 dict(solver="gibbs", rtol=1e-5, max_steps=400000, max_iter=300, estimate_equil=-1, log_level=0)]
        ism = None
        try:
            ge = W.equilibrate_gas(feed, T, P)
            if ge["converged"]:
                v = np.zeros(g.n_species + 1)
                for sp, val in ge["gas_kmol_d"].items(): v[g.species_index(sp)] = val
                seed = 0.05 * carbon(feed); iCO, iCO2, iCH4, iH2 = (g.species_index(k) for k in ("CO", "CO2", "CH4", "H2"))
                s1 = min(seed, 0.5 * v[iCO]); v[iCO] -= 2 * s1; v[iCO2] += s1; v[-1] += s1
                s2 = min(seed - s1, v[iCH4]); v[iCH4] -= s2; v[iH2] += 2 * s2; v[-1] += s2
                ism = v
        except Exception: pass
        attempts = [(T, None, kw) for kw in tries] + [(T, ism, kw) for kw in tries[:2]] + [(T + 0.25, ism, tries[0]), (T - 0.25, ism, tries[0])]
        last = None
        for Tt, ism_t, kw in attempts:
            if ism_t is None and Tt != T: continue
            r = W._equilibrate_mixture([(g, n), (sld, 0)], Tt, P, g.species_names, sld.species_names[0], ism_t, [kw])
            if r["converged"]:
                fH = W.total_stream_enthalpy_J_per_day(feed, 0, T, P); pH = W.total_stream_enthalpy_J_per_day(r["gas_kmol_d"], r["Csolid_kmol_d"], T, P)
                cin = carbon(feed); cout = carbon(r["gas_kmol_d"]) + r["Csolid_kmol_d"]
                if abs(cout - cin) > 1e-6 * max(cin, 1e-30): last = f"carbon imbalance {cout-cin:.3e}"; continue
                return r["gas_kmol_d"], r["Csolid_kmol_d"], (pH - fH) / 86400e3
            last = r["error"]
        raise RuntimeError("Stage 3 Gibbs failed after fallbacks: " + str(last)[:200])
    stage3.__wrapped__ = base
    F1m.stage3_eq_robust = stage3

def cfg_for(dG, P, recycle):
    fd = F1m.feeds()["2b_y0.60"]
    c = dict(config="2b_y0.60", feed=fd, mode="equilibrium", phi=np.nan, k2=0.0, n=None, variant="V1", P1_Pa=P, P2_Pa=P, P3_Pa=P, h=0.50)
    c.update(dict(p=0.05, r=0.95) if recycle else dict(p=1.0, r=0.0)); return c

def run(args):
    part, dG, P, recycle = args; t0 = time.time(); bind_solid(dG)
    pl = "1 atm" if abs(P - ATM) < 1 else f"{P/BAR:g} bar"; label = f"{part} dG={dG:g} kJ/mol {pl} " + ("recycle" if recycle else "once-through")
    cfg = cfg_for(dG, P, recycle)
    try:
        rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label)
        row.update(part=part, dG_kJ_per_mol=dG, P_bar=P / BAR, tear_err=info["err"],
                   S3_inlet_H2_to_CO2=rec["cfr_feed"].get("H2", 0) / max(rec["cfr_feed"].get("CO2", 0), 1e-12),
                   S3_inlet_CO_kmol_d=rec["cfr_feed"].get("CO", 0), S3_inlet_CO2_kmol_d=rec["cfr_feed"].get("CO2", 0), S3_inlet_H2_kmol_d=rec["cfr_feed"].get("H2", 0),
                   S3_out_CH4_tpd=tpd("CH4", rec["gas3"].get("CH4", 0)), S3_out_CO_tpd=tpd("CO", rec["gas3"].get("CO", 0)), S3_out_CO2_tpd=tpd("CO2", rec["gas3"].get("CO2", 0)),
                   S3_out_gas_carbon_tpd=tpd("C(s)", carbon(rec["gas3"])), recycle_CH4_to_S1_kmol_d=rec["ch4_rec"], wall_s=time.time() - t0, error="")
        streams = None
        if part == "A":
            fd = cfg["feed"]
            S = {"S1 inlet (fresh CH4)": {"CH4": FRESH_CH4}, "S1 outlet gas": rec["gas1"], "S2 fresh CO2 (biogas)": {"CO2": fd["co2_bio"]},
                 "S2 inlet": rec["feed2"], "S2 outlet gas": rec["gas2"], "S2 condensed water": rec["water2"], "membrane permeate (H2 export)": rec["permeate"],
                 "S3 inlet (membrane retentate)": rec["cfr_feed"], "S3 outlet gas": rec["gas3"], "S3 condensed water": rec["water3"], "S3 dry gas (lost without recycle)": rec["pw3"]}
            streams = [dict(stream=k, total_kmol_d=sum(d.values()), **{f"{sp}_kmol_d": d.get(sp, 0.0) for sp in SPECIES}, **{f"{sp}_tpd": tpd(sp, d.get(sp, 0.0)) for sp in SPECIES},
                            other_tpd=sum(v * W.MW.get(s, np.nan) / 1000 for s, v in d.items() if s not in SPECIES)) for k, d in S.items()]
            streams += [dict(stream="S1 solid carbon", total_kmol_d=rec["C1"], C_tpd=tpd("C(s)", rec["C1"])), dict(stream="S3 solid carbon", total_kmol_d=rec["C3"], C_tpd=tpd("C(s)", rec["C3"]))]
        return row, streams
    except Exception as ex:
        return dict(case=label, part=part, dG_kJ_per_mol=dG, P_bar=P / BAR, recycle=recycle, converged=False, error=str(ex)[:300], wall_s=time.time() - t0), None

def main():
    t0 = time.time()
    jobs = [("A", 0.0, ATM, False)]
    jobs += [("B", dG, ATM, rc) for dG in DG_GRID for rc in (False, True)]
    jobs += [("C", 0.0, P, rc) for P in P_GRID for rc in (False, True)]
    with Pool(min(len(jobs), 10)) as pool: res = pool.map(run, jobs)
    rows = [r for r, _ in res]; df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULT, "X4S_cases.csv"), index=False)
    st = pd.DataFrame(next(s for _, s in res if s is not None)); st.to_csv(os.path.join(RESULT, "X4S_once_through_streams.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    ff = lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.1f}"
    cols = ["dG_kJ_per_mol", "P_bar", "recycle", "converged", "iterations", "S1_X_CH4", "S1_inlet_CH4_ratio_vs_fresh", "C_stage1_tpd", "C_stage3_tpd", "C_total_tpd",
            "C_from_biogasCO2_tpd", "CO2_carbon_fixed_fraction", "S3_inlet_H2_to_CO2", "S3_out_gas_carbon_tpd", "purge_CO2_tpd", "H2_net_exportable_tpd", "water_total_tpd", "Q1_kW", "Q2_kW", "Q3_kW"]
    L = [f"X4 sensitivities (wall {time.time()-t0:.0f} s). ALL EQUILIBRIUM. X4 = 2b, y = 0.60, h = 0.50, r_CH4 = 0.95, p = 0.05.",
         f"Unconverged / failed: {int((~df.converged.astype(bool)).sum())}", "",
         "=== Part A: once-through reference at 1 atm, graphite (stream table [t/d])",
         st.to_string(index=False, float_format=lambda x: f"{x:,.2f}"), "",
         "=== Part B: carbon form (dG above graphite), 1 atm",
         df[df.part == "B"][cols].to_string(index=False, float_format=ff), "",
         "=== Part C: uniform pressure, graphite",
         df[df.part == "C"][cols].to_string(index=False, float_format=ff), ""]
    if (~df.converged.astype(bool)).any(): L += ["Failures:", df[~df.converged.astype(bool)][["case", "error"]].to_string(index=False)]
    open(os.path.join(RESULT, "X4S_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
