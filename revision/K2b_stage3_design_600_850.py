"""
K2b_stage3_design_600_850.py -- re-selection of the Stage 3 validation temperatures over a 600-850 K
candidate set (reviewer R2-1; follows K2 and the T3 window results of KIN_cfr_window_origin / X4_T3_sensitivity).

K2 fixed the Stage 3 candidates a priori at 600/650/700(/750) K. The temperature sweeps showed that the
reduced Stage 3 model only develops appreciable conversion above ~750 K, so the informative region was
outside K2's candidate set. Here the design is CHOSEN BY THE FISHER INFORMATION over
   T in {600, 625, ..., 850} K (11 candidates) x tau in {1, 3, 10} s x P in {1, 5} bar
by exhaustive D-optimal subset selection (2, 3 and 4 temperatures; all tau and P of a chosen T measured).
Same linearised framework as K2 (sigma = 0.02 absolute on every output, central differences, Jacobian
additive over points, so each candidate point is simulated once).

Two parameter sets on the submitted reduced Stage 3 model (rate parameters UNCALIBRATED, as coded):
  P3  = K2 set: ln k_meth, Ea_meth, n (pressure order of methanation; factor (P/P0)^(n-5)); outputs CH4 yield, COx conversion
  P7  = all three reactions: ln k_meth, Ea_meth, ln k_carb, Ea_carb (CO + H2 -> C + H2O), ln k_crack, Ea_crack (CH4 -> C + 2H2), n;
        outputs CH4 yield, COx conversion, solid-carbon yield C/(CO2+CO)_in   -- the deposition reactions dominate above 750 K
Nominal: k_meth 0.25 / Ea 100 kJ/mol, k_carb 0.08 / 110, k_crack 0.01 / 160, T_ref 650 K, n = 5 (native).
Feed: Stage 3 inlet of the kinetic design case (tau1* = 8.905 s, phi = 1, h = 0.35), as in K2.

Outputs (revision/Result/): K2b_candidate_points.csv (outputs and sensitivities per point), K2b_per_temperature_information.csv,
  K2b_designs.csv (all subsets ranked), K2b_recommended_design.csv, K2b_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python K2b_stage3_design_600_850.py   (~5-10 min, parallel)
"""
import os, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool, cpu_count
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W
import K2_validation_design as K2m

ATM, BAR, SIGMA, P0 = K2m.ATM, K2m.BAR, K2m.SIGMA, K2m.P0
FEED3 = K2m.FEED3
T_CAND = list(np.arange(600.0, 850.1, 25.0)); TAU = [1.0, 3.0, 10.0]; PRES = [ATM, 5 * BAR]
NAMES7 = ["k_meth", "Ea_meth", "k_carb", "Ea_carb", "k_crack", "Ea_crack", "n"]
THETA7 = np.array([np.log(0.25), 100e3, np.log(0.08), 110e3, np.log(0.01), 160e3, 5.0])
IDX3 = [0, 1, 6]; OUTS = ["Y_CH4", "X_COx", "Y_Csolid"]

def model7(pt, th):
    T, P, tau = pt["T"], pt["P"], pt["tau"]
    W.cfr_Ea_co2_meth, W.cfr_Ea_co_carb, W.cfr_Ea_ch4_carb = th[1], th[3], th[5]
    r = W.run_stage3_cfr_kinetic(dict(FEED3), T, P, eta=1.0, tau_s=tau, n_steps=W.cfr_n_steps,
                                 kref_co2_meth=np.exp(th[0]) * (P / P0) ** (th[6] - 5.0), kref_co_carb=np.exp(th[2]), kref_ch4_carb=np.exp(th[4]))
    g, f = r["result"]["gas_kmol_d"], FEED3; cox = f.get("CO2", 0) + f.get("CO", 0)
    return np.array([(g.get("CH4", 0) - f.get("CH4", 0)) / cox, 1 - (g.get("CO2", 0) + g.get("CO", 0)) / cox, r["result"]["Csolid_kmol_d"] / cox])

def point_jacobian(pt):
    t0 = time.time(); y0 = model7(pt, THETA7); cols = []
    for j, nm in enumerate(NAMES7):
        h = 0.02 if nm.startswith("k") else (0.02 * THETA7[j] if nm.startswith("Ea") else 0.05)
        tp, tm = THETA7.copy(), THETA7.copy(); tp[j] += h; tm[j] -= h
        d = (model7(pt, tp) - model7(pt, tm)) / (2 * h)
        if nm.startswith("Ea"): d = d * THETA7[j]
        cols.append(d)
    return dict(pt=pt, y=y0, J=np.array(cols).T, wall=time.time() - t0)   # J: (3 outputs x 7 params)

def stats(F):
    ev = np.linalg.eigvalsh(F)
    if ev.min() <= 1e-10 * ev.max(): return np.inf, np.full(F.shape[0], np.inf), -np.inf
    return ev.max() / ev.min(), np.sqrt(np.clip(np.diag(np.linalg.inv(F)), 0, None)), float(np.sum(np.log(ev)))

def main():
    t0 = time.time(); pts = [dict(T=T, tau=tau, P=P) for T in T_CAND for tau in TAU for P in PRES]
    with Pool(min(cpu_count(), len(pts))) as pool: res = pool.map(point_jacobian, pts)
    rows = []
    for r in res:
        row = dict(T_K=r["pt"]["T"], tau_s=r["pt"]["tau"], P_bar=r["pt"]["P"] / BAR, **{o: r["y"][i] for i, o in enumerate(OUTS)})
        for i, o in enumerate(OUTS):
            for j, nm in enumerate(NAMES7): row[f"dy_{o}_d{nm}"] = r["J"][i, j]
        row["wall_s"] = r["wall"]; rows.append(row)
    PT = pd.DataFrame(rows); PT.to_csv(os.path.join(RESULT, "K2b_candidate_points.csv"), index=False)
    # Fisher contribution per temperature (all tau x P of that T)
    Fbig = {T: sum(r["J"].T @ r["J"] / SIGMA ** 2 for r in res if r["pt"]["T"] == T) for T in T_CAND}
    per = []
    for T in T_CAND:
        sub = PT[PT.T_K == T]
        per.append(dict(T_K=T, Y_CH4_max=sub.Y_CH4.max(), X_COx_max=sub.X_COx.max(), Y_Csolid_max=sub.Y_Csolid.max(),
                        trace_F_P7=np.trace(Fbig[T]), trace_F_P3=np.trace(Fbig[T][np.ix_(IDX3, IDX3)]),
                        **{f"F_{nm}{nm}": Fbig[T][j, j] for j, nm in enumerate(NAMES7)}))
    PER = pd.DataFrame(per); PER.to_csv(os.path.join(RESULT, "K2b_per_temperature_information.csv"), index=False)
    # exhaustive D-optimal subset selection
    des = []
    for pset, idx, names in (("P3 (k_meth, Ea_meth, n)", IDX3, [NAMES7[i] for i in IDX3]), ("P7 (all three reactions + n)", list(range(7)), NAMES7)):
        for k in (2, 3, 4):
            for Ts in itertools.combinations(T_CAND, k):
                F = sum(Fbig[T] for T in Ts)[np.ix_(idx, idx)]; cond, se, ld = stats(F)
                row = dict(parameter_set=pset, n_T=k, temperatures="/".join(f"{T:.0f}" for T in Ts), n_points=k * len(TAU) * len(PRES), logdet_F=ld, cond_number=cond)
                row.update({f"rel_se_{nm}": se[i] for i, nm in enumerate(names)}); des.append(row)
    DES = pd.DataFrame(des).sort_values(["parameter_set", "n_T", "logdet_F"], ascending=[True, True, False]); DES.to_csv(os.path.join(RESULT, "K2b_designs.csv"), index=False)
    # K2 reference designs within the new framework (same tau/P grid)
    refs = {"K2 three T 600/650/700": (600.0, 650.0, 700.0), "K2 four T 600-750": (600.0, 650.0, 700.0, 750.0)}
    best = DES.groupby(["parameter_set", "n_T"]).head(1)
    rec = []
    for pset, idx in (("P3 (k_meth, Ea_meth, n)", IDX3), ("P7 (all three reactions + n)", list(range(7)))):
        for k in (2, 3, 4):
            b = best[(best.parameter_set == pset) & (best.n_T == k)].iloc[0]
            rec.append(dict(parameter_set=pset, n_T=k, kind="D-optimal", temperatures=b.temperatures, logdet_F=b.logdet_F, cond_number=b.cond_number,
                            **{c: b[c] for c in b.index if c.startswith("rel_se_")}))
        for lab, Ts in refs.items():
            F = sum(Fbig[T] for T in Ts)[np.ix_(idx, idx)]; cond, se, ld = stats(F)
            rec.append(dict(parameter_set=pset, n_T=len(Ts), kind=lab, temperatures="/".join(f"{T:.0f}" for T in Ts), logdet_F=ld, cond_number=cond,
                            **{f"rel_se_{NAMES7[i]}": se[j] for j, i in enumerate(idx)}))
    REC = pd.DataFrame(rec); REC.to_csv(os.path.join(RESULT, "K2b_recommended_design.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    ff = lambda x: f"{x:.3g}"
    L = [f"K2b Stage 3 design re-selection over 600-850 K (wall {time.time()-t0:.0f} s). sigma = {SIGMA}; KINETIC reduced model, UNCALIBRATED; feed = kinetic design case Stage 3 inlet (h = 0.35).",
         f"Candidates: T {T_CAND[0]:.0f}-{T_CAND[-1]:.0f} K step 25 (11) x tau {TAU} s x P {[p/BAR for p in PRES]} bar = {len(pts)} points; each chosen T is measured at all tau x P (6 points).", "",
         "=== per-temperature outputs (max over tau, P) and Fisher diagonal (information on each parameter from that temperature alone)",
         PER.to_string(index=False, float_format=ff), "",
         "=== D-optimal temperature sets vs the K2 a-priori sets (rel. SE = sqrt(diag F^-1): k -> relative error of k_ref; Ea -> relative error of Ea; n -> absolute)",
         REC.to_string(index=False, float_format=ff), "",
         "=== top 5 subsets per (parameter set, n_T)",
         DES.groupby(["parameter_set", "n_T"]).head(5).to_string(index=False, float_format=ff)]
    open(os.path.join(RESULT, "K2b_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
