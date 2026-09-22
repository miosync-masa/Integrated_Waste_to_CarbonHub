# -*- coding: utf-8 -*-
"""
K2: Identifiability of the reduced-model parameters and a minimal multi-temperature
validation design (reviewer R2-1; calibration priority from X1-l)
=======================================================================================
FUPROC-D-26-00269 Major Revision.

The submitted validation series were single-temperature tau sweeps (Stage 1 at 1200 K, Stage 2 at
950 K). Because the rate constants are written as k = k_ref exp[-Ea/R (1/T - 1/T_ref)], the
sensitivity of any measured conversion to Ea is identically zero at T = T_ref: a single-temperature
sweep can only determine k_ref. This script quantifies that with a local (linearised) design
analysis on the reduced models exactly as coded (all rate parameters UNCALIBRATED):

  measured outputs   y = conversions / yields at each design point (T, tau, P)
  parameters         theta = ln(k_ref), Ea, (n = pressure order of methanation, Stage 3 only)
  Jacobian           J_ij = dy_i / d theta_j by central finite differences
  Fisher information F = J^T J / sigma^2 with sigma = 0.02 (absolute error on a conversion)
  outputs            condition number of F, relative standard errors sqrt(diag(F^-1)) of ln k_ref
                     (= relative error of k_ref), Ea (relative to its nominal value) and n

Models and nominal parameters (as submitted / as used in K3-T3b):
  Stage 1  CH4 pyrolysis, run_stage1_pyrolysis_kinetic: k_ref 4.0 mol/m3/s, Ea 180 kJ/mol, T_ref 1200 K
  Stage 2  two-reaction model (K3): RWGS k1_ref 0.05 m3/mol/s, Ea1 80 kJ/mol (submitted); CO2 methanation
           k2_ref from phi = 1 at the Feed A inlet, Ea2 100 kJ/mol assumed (as the Stage 3 methanation);
           T_ref 950 K; outputs X_CO2 and CH4 yield
  Stage 3  submitted 3-reaction CFR model: CO2 methanation k_ref 0.25, Ea 100 kJ/mol, T_ref 650 K;
           pressure order n (native 5) via the T3b factor (P/P0)^(n-5); outputs CH4 yield and COx conversion
Feeds: design-case inlet compositions (Stage 1 pure CH4; Stage 2 feed of the kinetic design case;
Stage 3 CFR feed of the kinetic design case), all at 1 atm unless P is varied.

Designs compared per stage: single temperature (T_ref) x tau; two temperatures; three temperatures;
for Stage 3 additionally two pressures (1 and 5 bar) to identify n.

Reproducibility: cd <repo>/revision ; ../.venv/bin/python K2_validation_design.py   (~1-2 min)
Outputs: K2_identifiability.csv, K2_recommended_design.csv, K2_summary.txt
"""
import os, sys, itertools, time
import numpy as np, pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import P1_recycle_analysis as P1m
import KIN_chain_design_case as KC

OUT = os.path.join(RESULT, "K2_identifiability.csv"); OUT_REC = os.path.join(RESULT, "K2_recommended_design.csv"); OUT_TXT = os.path.join(RESULT, "K2_summary.txt")
ATM = ct.one_atm; BAR = 1e5; SIGMA = 0.02; R = ct.gas_constant / 1000.0; P0 = ct.one_atm
SP2 = ["CO2", "H2", "CO", "H2O", "CH4"]; NU1 = np.array([-1, -1, 1, 1, 0.0]); NU2 = np.array([-1, -4, 0, 2, 1.0])

# ---------------- design-case feeds
s1k = KC.stage1_kin(8.905); gas1 = s1k["result"]["gas_kmol_d"]
FEED2 = P1m.add(gas1, {"CO2": P1m.FRESH_CO2_BIO + P1m.FRESH_CO2_DAC, "H2": P1m.SOLAR_H2})
K2_PHI1 = P1m.k2_from_phi(1.0)
gas2, _ = P1m.stage2_kinetic(FEED2, K2_PHI1)
pw2, _ = W.remove_species(gas2, "H2O", W.water_remove_frac)
FEED3, _ = W.membrane_split(pw2, {"H2": 0.35, "CO2": 1.0, "CO": 1.0, "CH4": 1.0}, 1.0)

# ---------------- models as functions of (design point, parameters)
def stage1_model(pt, th):      # th = [ln kref, Ea]
    r = W.run_stage1_pyrolysis_kinetic(W.CH4_tpd, pt["T"], pt["P"], tau_s=pt["tau"], n_steps=W.pyro_n_steps, kref=np.exp(th[0]), Ea=th[1], Tref=1200.0)
    return np.array([r["X_CH4"]])

def stage2_model(pt, th):      # th = [ln k1ref, Ea1, ln k2ref, Ea2]
    T, P, tau = pt["T"], pt["P"], pt["tau"]; feed = W.clean_species_dict(FEED2)
    g = ct.Solution("gri30.yaml"); g.TP = T, P0; gRT = g.standard_gibbs_RT; i = [g.species_index(s) for s in SP2]
    Kc1 = float(np.exp(-NU1 @ gRT[i])); Kp2 = float(np.exp(-NU2 @ gRT[i]))
    k1 = np.exp(th[0]) * np.exp(-th[1] / R * (1 / T - 1 / 950.0)); k2 = np.exp(th[2]) * np.exp(-th[3] / R * (1 / T - 1 / 950.0))
    F0 = np.array([feed.get(s, 0.0) for s in SP2]) * 1000 / 86400; V = tau * F0.sum() * R * T / P; cT = P / (R * T)
    def rhs(v, F):
        F = np.maximum(F, 0); y = F / F.sum(); c = y * cT; a = y * P / P0
        return NU1 * k1 * (c[0] * c[1] - c[2] * c[3] / Kc1) + NU2 * k2 * (a[0] * a[1] ** 4 - a[4] * a[3] ** 2 / Kp2)
    sol = solve_ivp(rhs, (0, V), F0, method="BDF", rtol=1e-9, atol=1e-12); F = np.maximum(sol.y[:, -1], 0)
    X = 1 - F[0] / F0[0]; Ych4 = (F[4] - F0[4]) / F0[0]
    return np.array([X, Ych4])

def stage3_model(pt, th):      # th = [ln kref_meth, Ea_meth, n]
    T, P, tau = pt["T"], pt["P"], pt["tau"]
    kref = np.exp(th[0]) * (P / P0) ** (th[2] - 5.0)           # pressure-order factor (T3b), Arrhenius inside the model with Ea below
    W.cfr_Ea_co2_meth = th[1]                                   # module-level Ea used by run_stage3_cfr_kinetic
    r = W.run_stage3_cfr_kinetic(FEED3, T, P, eta=1.0, tau_s=tau, n_steps=W.cfr_n_steps, kref_co2_meth=kref)
    g = r["result"]["gas_kmol_d"]; f = FEED3
    ych4 = (g.get("CH4", 0) - f.get("CH4", 0)) / (f.get("CO2", 0) + f.get("CO", 0))
    xcox = 1 - (g.get("CO2", 0) + g.get("CO", 0)) / (f.get("CO2", 0) + f.get("CO", 0))
    return np.array([ych4, xcox])

STAGES = {
    "Stage 3 CO2 methanation (priority 1)": dict(model=stage3_model, theta=np.array([np.log(0.25), 100e3, 5.0]), names=["k_ref", "Ea", "n"], Tref=650.0,
        designs={"single T (650 K) x tau 1/3/10 s, 1 atm": dict(T=[650], tau=[1, 3, 10], P=[ATM]),
                 "two T (600/700 K) x tau 1/3/10 s, 1 atm": dict(T=[600, 700], tau=[1, 3, 10], P=[ATM]),
                 "three T (600/650/700 K) x tau 1/3/10 s, 1 atm": dict(T=[600, 650, 700], tau=[1, 3, 10], P=[ATM]),
                 "three T x tau 1/3/10 s x P 1/5 bar": dict(T=[600, 650, 700], tau=[1, 3, 10], P=[ATM, 5 * BAR]),
                 "three T x tau 0.3/1/3/10/30 s x P 1/5 bar": dict(T=[600, 650, 700], tau=[0.3, 1, 3, 10, 30], P=[ATM, 5 * BAR]),
                 "four T (600-750 K) x tau 1/3/10 s x P 1/5/10 bar": dict(T=[600, 650, 700, 750], tau=[1, 3, 10], P=[ATM, 5 * BAR, 10 * BAR]),
                 "four T (600-750 K) x tau 0.3/1/3/10/30 s x P 1/5/10 bar": dict(T=[600, 650, 700, 750], tau=[0.3, 1, 3, 10, 30], P=[ATM, 5 * BAR, 10 * BAR])},
        subsets={"(k_ref, Ea) only, n fixed": [0, 1]}),
    "Stage 2 RWGS + CO2 methanation (priority 2/3)": dict(model=stage2_model, theta=np.array([np.log(0.05), 80e3, np.log(K2_PHI1), 100e3]), names=["k1_ref", "Ea1", "k2_ref", "Ea2"], Tref=950.0,
        designs={"single T (950 K) x tau 1/3/10 s": dict(T=[950], tau=[1, 3, 10], P=[ATM]),
                 "two T (900/1000 K) x tau 1/3/10 s": dict(T=[900, 1000], tau=[1, 3, 10], P=[ATM]),
                 "three T (900/950/1000 K) x tau 1/3/10 s": dict(T=[900, 950, 1000], tau=[1, 3, 10], P=[ATM]),
                 "three T x tau 0.3/1/3/10/30 s": dict(T=[900, 950, 1000], tau=[0.3, 1, 3, 10, 30], P=[ATM])}),
    "Stage 1 CH4 pyrolysis (priority 4)": dict(model=stage1_model, theta=np.array([np.log(4.0), 180e3]), names=["k_ref", "Ea"], Tref=1200.0,
        designs={"single T (1200 K) x tau 1/3/10 s": dict(T=[1200], tau=[1, 3, 10], P=[ATM]),
                 "two T (1150/1250 K) x tau 1/3/10 s": dict(T=[1150, 1250], tau=[1, 3, 10], P=[ATM]),
                 "three T (1150/1200/1250 K) x tau 1/3/10 s": dict(T=[1150, 1200, 1250], tau=[1, 3, 10], P=[ATM]),
                 "three T x tau 0.3/1/3/10/30 s": dict(T=[1150, 1200, 1250], tau=[0.3, 1, 3, 10, 30], P=[ATM])}),
}

def jacobian(model, pts, theta, names):
    ys = []; J = []
    for pt in pts:
        y0 = model(pt, theta); ys.append(y0); rows = []
        for j in range(len(theta)):
            h = 0.02 if names[j].startswith("k") else (0.02 * abs(theta[j]) if names[j].startswith("Ea") else 0.05)   # ln k: 2 %; Ea: 2 %; n: 0.05
            tp = theta.copy(); tm = theta.copy(); tp[j] += h; tm[j] -= h
            d = (model(pt, tp) - model(pt, tm)) / (2 * h)
            if names[j].startswith("Ea"): d = d * abs(theta[j])          # sensitivity per relative change of Ea
            rows.append(d)
        J.append(np.array(rows).T)   # (n_outputs x n_params)
    return np.vstack(J), np.concatenate(ys)

def fisher_stats(J, names):
    Fm = J.T @ J / SIGMA ** 2; ev = np.linalg.eigvalsh(Fm); cond = ev.max() / max(ev.min(), 1e-300)
    if ev.min() <= 1e-10 * ev.max():
        se = np.full(J.shape[1], np.inf)
    else:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(Fm)), 0, None))
    return cond, se

def analyse(stage, spec):
    rows = []
    for dname, d in spec["designs"].items():
        pts = [dict(T=T, tau=tau, P=P) for T, tau, P in itertools.product(d["T"], d["tau"], d["P"])]
        t0 = time.time(); J, y = jacobian(spec["model"], pts, spec["theta"], spec["names"])
        subsets = {"all parameters": list(range(len(spec["names"])))}; subsets.update(spec.get("subsets", {}))
        for sname, idx in subsets.items():
            cond, se = fisher_stats(J[:, idx], [spec["names"][i] for i in idx])
            row = dict(stage=stage, design=dname, parameter_set=sname, n_points=len(pts), n_measurements=len(y), y_range=f"{y.min():.3f}-{y.max():.3f}", cond_number=cond)
            for k, i in enumerate(idx): row[f"rel_se_{spec['names'][i]}"] = se[k]
            for j, n in enumerate(spec["names"]): row[f"max_abs_sens_{n}"] = float(np.abs(J[:, j]).max())
            row["wall_s"] = time.time() - t0; rows.append(row)
            print(f"{stage} | {dname} | {sname}: cond {cond:.2e}, rel SE {dict(zip([spec['names'][i] for i in idx], np.round(se, 3)))}", flush=True)
    return rows

def main():
    rows = []
    for stage, spec in STAGES.items(): rows += analyse(stage, spec)
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    rec = pd.DataFrame([
        dict(priority=1, stage="Stage 3 CO2 methanation (650 K reactor)", measure="CH4 yield and COx conversion at reactor outlet", temperatures="600 / 650 / 700 K", residence_times="1 / 3 / 10 s (add 0.3 and 30 s if the 1-10 s points are near equilibrium)", pressures="1 and 5 bar", min_points="3 T x 3 tau x 2 P = 18", parameters="k_ref, Ea, pressure order n"),
        dict(priority=2, stage="Stage 2 CO2 methanation and RWGS (950 K reactor)", measure="CO2 conversion and CH4 yield (both needed to separate the two reactions)", temperatures="900 / 950 / 1000 K", residence_times="1 / 3 / 10 s", pressures="1 bar (5 bar optional)", min_points="3 T x 3 tau = 9", parameters="k1_ref, Ea1 (RWGS); k2_ref, Ea2 (methanation)"),
        dict(priority=4, stage="Stage 1 CH4 pyrolysis (1200 K reactor)", measure="CH4 conversion (solid carbon yield as check)", temperatures="1150 / 1200 / 1250 K", residence_times="1 / 3 / 10 s", pressures="1 bar", min_points="3 T x 3 tau = 9", parameters="k_ref, Ea")])
    rec.to_csv(OUT_REC, index=False)
    with open(OUT_TXT, "w") as fh:
        fh.write("K2 identifiability analysis (sigma_X = 0.02, linearised Fisher information; rate parameters uncalibrated)\n")
        with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.3g}"): fh.write(df.to_string(index=False) + "\n\n" + rec.to_string(index=False) + "\n")
    print("Saved:", OUT, OUT_REC, OUT_TXT)

if __name__ == "__main__":
    main()
