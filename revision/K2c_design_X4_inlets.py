"""
K2c_design_X4_inlets.py -- K2b repeated with the Stage 3 and Stage 2 inlets of the CURRENT base configuration X4
(2b, y = 0.60, h = 0.50, r_CH4 = 0.95, p = 0.05; converged recycle), reviewer R2-1.

Inlets (from X4_fig1_streams.csv, no re-solve):
  Stage 3, headline : kinetic design case (tau1* = 8.905 s, phi = 1) recycle-converged Stage 3 inlet
                      (CO-rich, H2/CO2 = 0.42; the as-modelled kinetic loop)
  Stage 3, reference: equilibrium-recycle Stage 3 inlet (H2/CO2 = 1.18; the loop the TEA is sized on)
  Stage 2, headline : equilibrium-recycle Stage 2 inlet (H2/CO2 = 1.64, as requested)
  Stage 2, reference: kinetic-loop Stage 2 inlet (H2/CO2 = 0.87)
Framework identical to K2/K2b: linearised Fisher information, sigma = 0.02 absolute on every output, central
differences, exhaustive D-optimal subset selection of temperatures (every chosen T measured at all tau x P).
  Stage 3: T 600-850 K step 25 x tau 1/3/10 s x P 1/5 bar; parameter sets P3 (ln k_meth, Ea_meth, n) and
           P7 (+ ln k_carb, Ea_carb, ln k_crack, Ea_crack); outputs Y_CH4, X_COx, Y_Csolid (P3 uses the first two)
  Stage 2: T 850-1050 K step 25 x tau 1/3/10 s x 1 atm; two-reaction model (K3): ln k1, Ea1 (RWGS), ln k2, Ea2
           (CO2 methanation, k2 from phi = 1 as in the design case); outputs X_CO2, Y_CH4
Rate parameters UNCALIBRATED (submitted reduced models). KINETIC values.

Outputs (revision/Result/): K2c_candidate_points.csv, K2c_per_temperature_information.csv, K2c_designs.csv,
  K2c_recommended_design.csv, K2c_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python K2c_design_X4_inlets.py   (~2 min, parallel)
"""
import os, time, itertools
import numpy as np, pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W
import K2_validation_design as K2m

ATM, BAR, SIGMA, P0, R = K2m.ATM, K2m.BAR, K2m.SIGMA, K2m.P0, K2m.R
SP2, NU1, NU2 = K2m.SP2, K2m.NU1, K2m.NU2

def x4_stream(value_type, stream):
    L = pd.read_csv(os.path.join(RESULT, "X4_fig1_streams.csv"))
    g = L[(L.value_type == value_type) & (L.stream == stream) & (~L.species.isin(["total", "other"]))]
    return {r.species: float(r.kmol_d) for r in g.itertuples() if r.kmol_d > 0}

FEEDS3 = {"S3 kinetic-loop inlet (X4, H2/CO2 0.42)": x4_stream("kinetic", "S3 inlet (membrane retentate)"),
          "S3 equilibrium-loop inlet (X4, H2/CO2 1.18)": x4_stream("equilibrium", "S3 inlet (membrane retentate)")}
FEEDS2 = {"S2 equilibrium-loop inlet (X4, H2/CO2 1.64)": x4_stream("equilibrium", "S2 inlet"),
          "S2 kinetic-loop inlet (X4, H2/CO2 0.87)": x4_stream("kinetic", "S2 inlet")}

# ---------------- Stage 3 (as K2b)
T3_CAND = list(np.arange(600.0, 850.1, 25.0)); TAU3 = [1.0, 3.0, 10.0]; P3 = [ATM, 5 * BAR]
N7 = ["k_meth", "Ea_meth", "k_carb", "Ea_carb", "k_crack", "Ea_crack", "n"]
TH7 = np.array([np.log(0.25), 100e3, np.log(0.08), 110e3, np.log(0.01), 160e3, 5.0]); IDX3 = [0, 1, 6]; OUT3 = ["Y_CH4", "X_COx", "Y_Csolid"]
def model3(feed, pt, th):
    T, P, tau = pt["T"], pt["P"], pt["tau"]
    W.cfr_Ea_co2_meth, W.cfr_Ea_co_carb, W.cfr_Ea_ch4_carb = th[1], th[3], th[5]
    r = W.run_stage3_cfr_kinetic(dict(feed), T, P, eta=1.0, tau_s=tau, n_steps=W.cfr_n_steps,
                                 kref_co2_meth=np.exp(th[0]) * (P / P0) ** (th[6] - 5.0), kref_co_carb=np.exp(th[2]), kref_ch4_carb=np.exp(th[4]))
    g = r["result"]["gas_kmol_d"]; cox = feed.get("CO2", 0) + feed.get("CO", 0)
    return np.array([(g.get("CH4", 0) - feed.get("CH4", 0)) / cox, 1 - (g.get("CO2", 0) + g.get("CO", 0)) / cox, r["result"]["Csolid_kmol_d"] / cox])

# ---------------- Stage 2 (as K2, feed as argument)
T2_CAND = list(np.arange(850.0, 1050.1, 25.0)); TAU2 = [1.0, 3.0, 10.0]; P2 = [ATM]
N4 = ["k1_ref", "Ea1", "k2_ref", "Ea2"]; TH4 = np.array([np.log(0.05), 80e3, np.log(K2m.K2_PHI1), 100e3]); OUT2 = ["X_CO2", "Y_CH4"]
def model2(feed, pt, th):
    T, P, tau = pt["T"], pt["P"], pt["tau"]; feed = W.clean_species_dict(feed)
    g = ct.Solution("gri30.yaml"); g.TP = T, P0; gRT = g.standard_gibbs_RT; i = [g.species_index(s) for s in SP2]
    Kc1 = float(np.exp(-NU1 @ gRT[i])); Kp2 = float(np.exp(-NU2 @ gRT[i]))
    k1 = np.exp(th[0]) * np.exp(-th[1] / R * (1 / T - 1 / 950.0)); k2 = np.exp(th[2]) * np.exp(-th[3] / R * (1 / T - 1 / 950.0))
    F0 = np.array([feed.get(s, 0.0) for s in SP2]) * 1000 / 86400; Fo = sum(v for k, v in feed.items() if k not in SP2) * 1000 / 86400
    V = tau * (F0.sum() + Fo) * R * T / P; cT = P / (R * T)
    def rhs(v, F):
        F = np.maximum(F, 0); y = F / (F.sum() + Fo); c = y * cT; a = y * P / P0
        return NU1 * k1 * (c[0] * c[1] - c[2] * c[3] / Kc1) + NU2 * k2 * (a[0] * a[1] ** 4 - a[4] * a[3] ** 2 / Kp2)
    sol = solve_ivp(rhs, (0, V), F0, method="BDF", rtol=1e-9, atol=1e-12); F = np.maximum(sol.y[:, -1], 0)
    return np.array([1 - F[0] / F0[0], (F[4] - F0[4]) / F0[0]])

def point_jacobian(args):
    kind, flabel, feed, pt = args; t0 = time.time()
    model, th, names = (model3, TH7, N7) if kind == "S3" else (model2, TH4, N4)
    y0 = model(feed, pt, th); cols = []
    for j, nm in enumerate(names):
        h = 0.02 if nm.startswith("k") else (0.02 * th[j] if nm.startswith("Ea") else 0.05)
        tp, tm = th.copy(), th.copy(); tp[j] += h; tm[j] -= h
        d = (model(feed, pt, tp) - model(feed, pt, tm)) / (2 * h)
        if nm.startswith("Ea"): d = d * th[j]
        cols.append(d)
    return dict(kind=kind, feed=flabel, pt=pt, y=y0, J=np.array(cols).T, wall=time.time() - t0)

def stats(F):
    ev = np.linalg.eigvalsh(F)
    if ev.min() <= 1e-10 * ev.max(): return np.inf, np.full(F.shape[0], np.inf), -np.inf
    return ev.max() / ev.min(), np.sqrt(np.clip(np.diag(np.linalg.inv(F)), 0, None)), float(np.sum(np.log(ev)))

def main():
    t0 = time.time(); jobs = []
    for fl, fd in FEEDS3.items(): jobs += [("S3", fl, fd, dict(T=T, tau=tau, P=P)) for T in T3_CAND for tau in TAU3 for P in P3]
    for fl, fd in FEEDS2.items(): jobs += [("S2", fl, fd, dict(T=T, tau=tau, P=P)) for T in T2_CAND for tau in TAU2 for P in P2]
    with Pool(min(cpu_count(), 14)) as pool: res = pool.map(point_jacobian, jobs)
    PT, PER, DES, REC = [], [], [], []
    groups = [("S3", fl, N7, OUT3, T3_CAND, len(TAU3) * len(P3), [("P3 (k_meth, Ea_meth, n)", IDX3), ("P7 (all three reactions + n)", list(range(7)))],
               {"K2 a priori 600/650/700": (600.0, 650.0, 700.0), "K2b (old inlet) 600/725/800": (600.0, 725.0, 800.0), "K2b (old inlet) 600/700/800/825": (600.0, 700.0, 800.0, 825.0)}) for fl in FEEDS3]
    groups += [("S2", fl, N4, OUT2, T2_CAND, len(TAU2), [("P4 (k1, Ea1, k2, Ea2)", list(range(4)))],
                {"K2 a priori 900/950/1000": (900.0, 950.0, 1000.0), "K2 two T 900/1000": (900.0, 1000.0)}) for fl in FEEDS2]
    for kind, fl, names, outs, cand, nper, psets, refs in groups:
        rr = [r for r in res if r["kind"] == kind and r["feed"] == fl]
        for r in rr:
            row = dict(stage=kind, feed=fl, T_K=r["pt"]["T"], tau_s=r["pt"]["tau"], P_bar=r["pt"]["P"] / BAR, **{o: r["y"][i] for i, o in enumerate(outs)})
            for i, o in enumerate(outs):
                for j, nm in enumerate(names): row[f"dy_{o}_d{nm}"] = r["J"][i, j]
            PT.append(row)
        Fbig = {T: sum(r["J"].T @ r["J"] / SIGMA ** 2 for r in rr if r["pt"]["T"] == T) for T in cand}
        for T in cand:
            ys = np.array([r["y"] for r in rr if r["pt"]["T"] == T])
            PER.append(dict(stage=kind, feed=fl, T_K=T, **{f"{o}_min": ys[:, i].min() for i, o in enumerate(outs)}, **{f"{o}_max": ys[:, i].max() for i, o in enumerate(outs)},
                            trace_F=np.trace(Fbig[T]), **{f"F_{nm}": Fbig[T][j, j] for j, nm in enumerate(names)}))
        for pset, idx in psets:
            pn = [names[i] for i in idx]; best = {}
            for k in (2, 3, 4):
                for Ts in itertools.combinations(cand, k):
                    F = sum(Fbig[T] for T in Ts)[np.ix_(idx, idx)]; cond, se, ld = stats(F)
                    row = dict(stage=kind, feed=fl, parameter_set=pset, n_T=k, temperatures="/".join(f"{T:.0f}" for T in Ts), n_points=k * nper, logdet_F=ld, cond_number=cond, **{f"rel_se_{nm}": se[i] for i, nm in enumerate(pn)})
                    DES.append(row)
                    if k not in best or ld > best[k]["logdet_F"]: best[k] = row
            for k in (2, 3, 4): REC.append(dict(kind="D-optimal", **best[k]))
            for lab, Ts in refs.items():
                F = sum(Fbig[T] for T in Ts)[np.ix_(idx, idx)]; cond, se, ld = stats(F)
                REC.append(dict(kind=lab, stage=kind, feed=fl, parameter_set=pset, n_T=len(Ts), temperatures="/".join(f"{T:.0f}" for T in Ts), n_points=len(Ts) * nper, logdet_F=ld, cond_number=cond, **{f"rel_se_{nm}": se[i] for i, nm in enumerate(pn)}))
    PT, PER, DES, REC = map(pd.DataFrame, (PT, PER, DES, REC))
    DES = DES.sort_values(["stage", "feed", "parameter_set", "n_T", "logdet_F"], ascending=[True, True, True, True, False])
    PT.to_csv(os.path.join(RESULT, "K2c_candidate_points.csv"), index=False); PER.to_csv(os.path.join(RESULT, "K2c_per_temperature_information.csv"), index=False)
    DES.to_csv(os.path.join(RESULT, "K2c_designs.csv"), index=False); REC.to_csv(os.path.join(RESULT, "K2c_recommended_design.csv"), index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); ff = lambda x: f"{x:.3g}"
    L = [f"K2c: D-optimal validation temperatures with the X4 inlets (wall {time.time()-t0:.0f} s). sigma = {SIGMA}; KINETIC reduced models, UNCALIBRATED.",
         "Inlets [kmol/d]: " + " | ".join(f"{k}: " + ", ".join(f"{s} {v:,.0f}" for s, v in d.items()) for k, d in {**FEEDS3, **FEEDS2}.items()), ""]
    for kind, fl, *_ in groups:
        L += [f"=== {fl}: per-temperature outputs and Fisher diagonal", PER[(PER.stage == kind) & (PER.feed == fl)].drop(columns=["stage", "feed"]).to_string(index=False, float_format=ff), "",
              f"=== {fl}: D-optimal sets vs reference designs", REC[(REC.stage == kind) & (REC.feed == fl)].drop(columns=["stage", "feed"]).dropna(axis=1, how="all").to_string(index=False, float_format=ff), ""]
    open(os.path.join(RESULT, "K2c_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
