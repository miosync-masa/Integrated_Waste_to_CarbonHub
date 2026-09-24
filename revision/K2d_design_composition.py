"""
K2d_design_composition.py -- D-optimal validation design with the feed composition (H2/CO2, and CO/CO2 for
Stage 3) as a design variable, two synthetic-feed series per stage (reviewer R2-1; follows K2b/K2c).

Stage 3 (reduced 3-reaction model; parameters ln k_meth, Ea_meth, ln k_carb, Ea_carb, ln k_crack, Ea_crack, n)
  candidate feeds : dry synthetic gas CO2 : CO : H2 = 1 : r_CO : r_H2, r_H2 in {1, 2, 3, 4}, r_CO in {0.01, 1}
                    (r_CO = 1 ~ the X4 equilibrium-loop inlet CO/CO2 = 1.13 and the 950 K RWGS product; r_CO = 0.01
                    = CO-lean series that isolates the methanation parameters from R2). No CH4, no H2O (dry feed).
                    NOTE: an exactly CO-free feed freezes the submitted reduced model (its negativity limiter sets
                    the step to zero because the TRACE activity floor gives R2 a small positive rate while the CO
                    inventory is zero); a 1 % CO trace avoids this model artefact and is physically realistic.
  candidate points: T 600-850 K step 25 x tau 1/3/10 s x P 1/5 bar (66 per feed)
  designs         : two feed series (distinct compositions), each measured at its own set of 3 or 4 temperatures
                    (all tau x P at each T); exhaustive over feeds and temperature subsets; single-series designs
                    listed for comparison. Criteria: D-optimal (max log det F) and minimax (min of the largest
                    relative SE among k_meth, Ea_meth, n, k_carb, Ea_carb, Ea_crack).
Stage 2 (two-reaction model; ln k1, Ea1 (RWGS), ln k2, Ea2 (CO2 methanation, k2 from phi = 1))
  candidate feeds : CO2 : H2 = 1 : r, r in {1.5, 2, 3, 4} (no CO: RWGS product); T 850-1050 K step 25 x tau 1/3/10 s x 1 atm
  designs         : one or two feed series x 3-4 temperatures each; question: does any design bring all four
                    parameters within 15 % relative SE?
Linearised Fisher information, sigma = 0.02 absolute on every output, central differences (as K2/K2b/K2c).
KINETIC reduced models, rate parameters UNCALIBRATED. Comparison rows are read from K2b/K2c outputs.

Outputs (revision/Result/): K2d_candidate_points.csv, K2d_stage3_designs_top.csv, K2d_stage2_designs_top.csv,
  K2d_recommended.csv, K2d_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python K2d_design_composition.py   (~5 min, parallel)
"""
import os, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool, cpu_count
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import K2c_design_X4_inlets as K2c

ATM, BAR, SIGMA = K2c.ATM, K2c.BAR, K2c.SIGMA
N7, N4 = K2c.N7, K2c.N4; OUT3, OUT2 = K2c.OUT3, K2c.OUT2
T3_CAND, TAU3, P3 = K2c.T3_CAND, K2c.TAU3, K2c.P3; T2_CAND, TAU2, P2 = K2c.T2_CAND, K2c.TAU2, K2c.P2
BASIS = 1000.0  # kmol/d CO2 (intensive results; absolute flow irrelevant at fixed tau)
FEEDS3 = {f"H2/CO2={rh:g}, CO/CO2={rc:g}": {"CO2": BASIS, "H2": rh * BASIS, **({"CO": rc * BASIS} if rc > 0 else {})} for rc in (0.01, 1.0) for rh in (1.0, 2.0, 3.0, 4.0)}
FEEDS2 = {f"H2/CO2={r:g}": {"CO2": BASIS, "H2": r * BASIS} for r in (1.5, 2.0, 3.0, 4.0)}
SE6 = ["k_meth", "Ea_meth", "n", "k_carb", "Ea_carb", "Ea_crack"]

def subset_sums(Fbig, cand, sizes):
    """List of (temperature tuple, summed Fisher matrix) for all subsets of the given sizes."""
    out = []
    for k in sizes:
        for Ts in itertools.combinations(cand, k): out.append((Ts, sum(Fbig[T] for T in Ts)))
    return out

def batch_stats(Fs):
    """Fs: (N, p, p). Returns logdet (N,), rel SE (N, p) (inf where singular)."""
    sign, ld = np.linalg.slogdet(Fs); ld = np.where(sign > 0, ld, -np.inf)
    se = np.full(Fs.shape[:2], np.inf); ok = np.isfinite(ld)
    if ok.any():
        inv = np.linalg.inv(Fs[ok]); d = np.einsum("nii->ni", inv); se[ok] = np.sqrt(np.clip(d, 0, None))
    return ld, se

def enumerate_designs(Fbig_by_feed, cand, sizes, names, se_names, n_per_T, keep=20):
    feeds = list(Fbig_by_feed); subs = {f: subset_sums(Fbig_by_feed[f], cand, sizes) for f in feeds}
    idx_se = [names.index(n) for n in se_names]; best_d, best_m = [], []
    def consider(rows_meta, Fs):
        ld, se = batch_stats(Fs); mx = se[:, idx_se].max(axis=1)
        for arr, store, key in ((ld, best_d, lambda i: -ld[i]), (-mx, best_m, lambda i: mx[i])):
            top = np.argsort(-arr)[:keep]
            for i in top:
                if np.isfinite(arr[i]): store.append((key(i), rows_meta[i], ld[i], se[i]))
            store.sort(key=lambda t: t[0]); del store[keep:]
    # single series
    for f in feeds:
        meta = [dict(series_1=f, T_1="/".join(f"{T:.0f}" for T in Ts), series_2="", T_2="", n_points=len(Ts) * n_per_T) for Ts, _ in subs[f]]
        consider(meta, np.array([F for _, F in subs[f]]))
    # two distinct series
    for f1, f2 in itertools.combinations(feeds, 2):
        S2 = np.array([F for _, F in subs[f2]]); T2l = ["/".join(f"{T:.0f}" for T in Ts) for Ts, _ in subs[f2]]; n2 = [len(Ts) for Ts, _ in subs[f2]]
        for Ts1, F1 in subs[f1]:
            meta = [dict(series_1=f1, T_1="/".join(f"{T:.0f}" for T in Ts1), series_2=f2, T_2=T2l[j], n_points=(len(Ts1) + n2[j]) * n_per_T) for j in range(len(S2))]
            consider(meta, S2 + F1[None])
    def frame(store, crit):
        rows = []
        for rank, (_, m, ld, se) in enumerate(store, 1):
            rows.append(dict(criterion=crit, rank=rank, **m, logdet_F=ld, max_rel_se=max(se[i] for i in idx_se), **{f"rel_se_{n}": se[i] for i, n in enumerate(names)}))
        return pd.DataFrame(rows)
    return pd.concat([frame(best_d, "D-optimal"), frame(best_m, "minimax SE")], ignore_index=True)

def main():
    t0 = time.time(); jobs = []
    for fl, fd in FEEDS3.items(): jobs += [("S3", fl, fd, dict(T=T, tau=tau, P=P)) for T in T3_CAND for tau in TAU3 for P in P3]
    for fl, fd in FEEDS2.items(): jobs += [("S2", fl, fd, dict(T=T, tau=tau, P=P)) for T in T2_CAND for tau in TAU2 for P in P2]
    with Pool(min(cpu_count(), 14)) as pool: res = pool.map(K2c.point_jacobian, jobs)
    PT = []
    for r in res:
        outs, names = (OUT3, N7) if r["kind"] == "S3" else (OUT2, N4)
        row = dict(stage=r["kind"], feed=r["feed"], T_K=r["pt"]["T"], tau_s=r["pt"]["tau"], P_bar=r["pt"]["P"] / BAR, **{o: r["y"][i] for i, o in enumerate(outs)})
        for i, o in enumerate(outs):
            for j, nm in enumerate(names): row[f"dy_{o}_d{nm}"] = r["J"][i, j]
        PT.append(row)
    PT = pd.DataFrame(PT); PT.to_csv(os.path.join(RESULT, "K2d_candidate_points.csv"), index=False)
    def fisher_blocks(kind, feeds, cand):
        return {fl: {T: sum(r["J"].T @ r["J"] / SIGMA ** 2 for r in res if r["kind"] == kind and r["feed"] == fl and r["pt"]["T"] == T) for T in cand} for fl in feeds}
    F3, F2 = fisher_blocks("S3", FEEDS3, T3_CAND), fisher_blocks("S2", FEEDS2, T2_CAND)
    t1 = time.time()
    D3 = enumerate_designs(F3, T3_CAND, (3, 4), N7, SE6, len(TAU3) * len(P3)); D3.to_csv(os.path.join(RESULT, "K2d_stage3_designs_top.csv"), index=False)
    D2 = enumerate_designs(F2, T2_CAND, (3, 4), N4, N4, len(TAU2)); D2.to_csv(os.path.join(RESULT, "K2d_stage2_designs_top.csv"), index=False)
    # per-feed conversion ranges (informative region check)
    rng = PT.groupby(["stage", "feed"]).agg(Y_CH4_max=("Y_CH4", "max"), X_COx_max=("X_COx", "max"), Y_Csolid_max=("Y_Csolid", "max"), X_CO2_max=("X_CO2", "max"), X_CO2_min=("X_CO2", "min")).reset_index()
    # comparison rows from K2b / K2c
    comp = []
    try:
        b = pd.read_csv(os.path.join(RESULT, "K2b_recommended_design.csv")); b = b[(b.parameter_set.str.startswith("P7")) & (b.kind == "D-optimal") & (b.n_T == 4)].iloc[0]
        comp.append(dict(source="K2b, old inlet (submitted design case, H2/CO2 ~3, CO/CO2 1.7)", design=b.temperatures, n_points=24, **{f"rel_se_{n}": b[f"rel_se_{n}"] for n in SE6}))
        c = pd.read_csv(os.path.join(RESULT, "K2c_recommended_design.csv"))
        for fl in c[c.stage == "S3"].feed.unique():
            r = c[(c.stage == "S3") & (c.feed == fl) & (c.parameter_set.str.startswith("P7")) & (c.kind == "D-optimal") & (c.n_T == 4)].iloc[0]
            comp.append(dict(source=f"K2c, {fl}", design=r.temperatures, n_points=24, **{f"rel_se_{n}": r[f"rel_se_{n}"] for n in SE6}))
    except Exception as ex: comp.append(dict(source=f"comparison unavailable: {ex}"))
    for crit in ("D-optimal", "minimax SE"):
        r = D3[D3.criterion == crit].iloc[0]
        comp.append(dict(source=f"K2d {crit} (two series)", design=f"{r.series_1} @ {r.T_1} + {r.series_2} @ {r.T_2}", n_points=r.n_points, **{f"rel_se_{n}": r[f"rel_se_{n}"] for n in SE6}))
    COMP = pd.DataFrame(comp); COMP["max_rel_se_6"] = COMP[[f"rel_se_{n}" for n in SE6]].max(axis=1)
    # Stage 2 answer: any design with all four within 15 %?
    ok2 = D2[(D2.criterion == "minimax SE")].copy(); feasible = ok2[ok2.max_rel_se <= 0.15]
    REC = pd.concat([COMP.assign(stage="S3"), D2[D2.criterion == "minimax SE"].head(5).assign(stage="S2")], ignore_index=True); REC.to_csv(os.path.join(RESULT, "K2d_recommended.csv"), index=False)
    pd.set_option("display.width", 340); pd.set_option("display.max_columns", 40); ff = lambda x: f"{x:.3g}"
    L = [f"K2d composition-aware D-optimal design (wall {time.time()-t0:.0f} s, enumeration {time.time()-t1:.0f} s). sigma = {SIGMA}. KINETIC reduced models, UNCALIBRATED.",
         f"Stage 3: {len(FEEDS3)} synthetic feeds x {len(T3_CAND)} T x {len(TAU3)} tau x {len(P3)} P; designs = 1 or 2 feed series x 3-4 T each (all tau x P per T).",
         f"Stage 2: {len(FEEDS2)} synthetic feeds x {len(T2_CAND)} T x {len(TAU2)} tau x 1 atm; same design rule.", "",
         "=== conversion ranges per feed (max over T, tau, P)", rng.to_string(index=False, float_format=ff), "",
         "=== Stage 3: top designs (D-optimal and minimax over k_meth, Ea_meth, n, k_carb, Ea_carb, Ea_crack)",
         D3.groupby("criterion").head(8).to_string(index=False, float_format=ff), "",
         "=== Stage 3: comparison with K2b (old inlet) and K2c (X4 inlets)", COMP.to_string(index=False, float_format=ff), "",
         "=== Stage 2: top designs", D2.groupby("criterion").head(8).to_string(index=False, float_format=ff), "",
         f"Stage 2 designs with all four parameters within 15 %: {len(feasible)} of the top-{len(ok2)} minimax list" + (f"; best max SE = {ok2.max_rel_se.min():.3f}" if len(ok2) else "")]
    open(os.path.join(RESULT, "K2d_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
