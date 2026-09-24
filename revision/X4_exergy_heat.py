"""
X4_exergy_heat.py -- thermal (heat) exergy of the X4 process streams, as an independent check of the
T4 pinch conclusion that the 650 K Stage 3 exotherm cannot serve the 1200 K Stage 1 endotherm.

Scope: heat exergy only (no chemical exergy, no overall exergy balance). Environment T0 = 298.15 K, 1 atm.
Streams: the 13 hot/cold streams of T4_heat_cascade for the X4 representative case (2b, y = 0.60,
h = 0.50, EQUILIBRIUM recycle). Compositions are taken from X4_fig1_streams.csv (no re-solve) and the
T-H curves are rebuilt with T4_heat_cascade.curve (GRI-3.0 enthalpies, graphite, water condensation
below the dew point); duties are checked against T4_streams_X4_rep.csv.

Exergy of a heat flow Q delivered between T_in and T_out:
  isothermal reaction duty          Ex = Q (1 - T0/T)
  log-mean approximation            Ex = Q (1 - T0/T_lm),  T_lm = (T_in - T_out)/ln(T_in/T_out)
  curve integral (reference value)  Ex = sum over segments dH_i (1 - T0/T_lm,i)  -- exact for piecewise-constant cp
Condensing streams are split into a sensible part (gas incl. vapour cooled as ideal gas, curve with
condensing=False) and a latent part (remainder; exergy = total - sensible, effective temperature reported).

Outputs (revision/Result/): X4_exergy_heat_streams.csv, X4_exergy_heat_aggregates.csv, X4_exergy_heat_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python X4_exergy_heat.py   (seconds)
"""
import os, time
import numpy as np, pandas as pd
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import T4_heat_cascade as T4m

T0 = 298.15; T_COND = T4m.T_COND
ETA_ORC = [0.20, 0.25]

def tlm(a, b): return a if abs(a - b) < 1e-9 else (a - b) / np.log(a / b)
def ex_curve(c):
    """Integral of (1 - T0/T) dH along a T4 curve (ascending T, cumulative H)."""
    T, H = c["T"], c["H"]; ex = 0.0
    for i in range(len(T) - 1):
        dH = H[i + 1] - H[i]
        if dH > 0: ex += dH * (1.0 - T0 / tlm(T[i + 1], T[i]))
    return ex

def comps():
    L = pd.read_csv(os.path.join(RESULT, "X4_fig1_streams.csv")); L = L[L.value_type == "equilibrium"]
    d = {}
    for name, g in L.groupby("stream"):
        g = g[~g.species.isin(["total", "other"])]; d[name] = {r.species: r.kmol_d for r in g.itertuples() if r.kmol_d > 0}
    return d

def main():
    t0 = time.time(); C = comps(); ref = pd.read_csv(os.path.join(RESULT, "T4_streams_X4_rep.csv")).set_index("stream").duty_kW
    S1 = C["S1 solid carbon"]["C(s)"]; S3 = C["S3 solid carbon"]["C(s)"]
    Q1, Q2, Q3 = ref["Stage 1 endotherm @1200 K"], ref["Stage 2 endotherm @950 K"], ref["Stage 3 exotherm @650 K"]
    spec = [  # (kind, name, T_from, T_to, comp, solid, duty)
        ("hot", "Stage 1 gas 1200->950 K", 1200.0, 950.0, C["S1 outlet gas"], 0.0, None),
        ("hot", "Stage 1 solid C 1200->313 K", 1200.0, T_COND, None, S1, None),
        ("hot", "Stage 2 gas 950->313 K (condensing)", 950.0, T_COND, C["S2 outlet gas"], 0.0, None),
        ("hot", "Stage 3 exotherm @650 K", 650.0, 650.0, None, 0.0, Q3),
        ("hot", "Stage 3 gas 650->313 K (condensing)", 650.0, T_COND, C["S3 outlet gas"], 0.0, None),
        ("hot", "Stage 3 solid C 650->313 K", 650.0, T_COND, None, S3, None),
        ("cold", "fresh CH4 298->1200 K", T0, 1200.0, C["S1 fresh CH4 (waste)"], 0.0, None),
        ("cold", "recycled CH4 313->1200 K", T_COND, 1200.0, C["S1 recycled CH4 (from S3 outlet)"], 0.0, None),
        ("cold", "Stage 1 endotherm @1200 K", 1200.0, 1200.0, None, 0.0, Q1),
        ("cold", "fresh CO2 298->950 K", T0, 950.0, C["S2 fresh CO2 (biogas)"], 0.0, None),
        ("cold", "recycle to Stage 2 313->950 K", T_COND, 950.0, C["S2 recycle from S3 outlet"], 0.0, None),
        ("cold", "Stage 2 endotherm @950 K", 950.0, 950.0, None, 0.0, Q2),
        ("cold", "CFR feed 313->650 K", T_COND, 650.0, C["S3 inlet (membrane retentate)"], 0.0, None)]
    rows = []
    for kind, name, Ta, Tb, comp, solid, duty in spec:
        if duty is not None:
            Q, Tm, ex_lm, ex_int, sens, lat, ex_s, ex_l = duty, Ta, duty * (1 - T0 / Ta), duty * (1 - T0 / Ta), np.nan, np.nan, np.nan, np.nan
        else:
            c = T4m.curve(kind, name, Ta, Tb, comp=comp, solid=solid); Q = c["duty"]; Tm = tlm(max(Ta, Tb), min(Ta, Tb))
            ex_lm = Q * (1 - T0 / Tm); ex_int = ex_curve(c)
            if comp is not None and comp.get("H2O", 0) > 0 and "condensing" in name:
                # sensible part: same gas cooled without condensation
                Ts = c["T"]; Hs = np.array([T4m.h_gas_kW(comp, T) for T in Ts]); Hs -= Hs[0]
                cs = dict(T=Ts, H=Hs); sens = float(Hs[-1]); ex_s = ex_curve(cs); lat = Q - sens; ex_l = ex_int - ex_s
            else: sens, lat, ex_s, ex_l = Q, 0.0, ex_int, 0.0
        rows.append(dict(kind=kind, stream=name, T_in_K=Ta, T_out_K=Tb, duty_kW=Q, duty_T4_kW=float(ref[name]), T_lm_K=Tm,
                         carnot_factor_lm=1 - T0 / Tm, Ex_lm_kW=ex_lm, Ex_integral_kW=ex_int, Ex_over_Q=ex_int / Q,
                         sensible_kW=sens, latent_kW=lat, Ex_sensible_kW=ex_s, Ex_latent_kW=ex_l,
                         T_eff_latent_K=(T0 / (1 - ex_l / lat) if (lat == lat and lat > 0) else np.nan)))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESULT, "X4_exergy_heat_streams.csv"), index=False)
    hot, cold = D[D.kind == "hot"], D[D.kind == "cold"]
    ex3 = float(D[D.stream == "Stage 3 exotherm @650 K"].Ex_integral_kW.iloc[0]); ex1 = float(D[D.stream == "Stage 1 endotherm @1200 K"].Ex_integral_kW.iloc[0])
    f1 = 1 - T0 / 1200.0; f3 = 1 - T0 / 650.0
    agg = dict(hot_duty_kW=hot.duty_kW.sum(), hot_Ex_kW=hot.Ex_integral_kW.sum(), hot_Ex_lm_kW=hot.Ex_lm_kW.sum(),
               cold_duty_kW=cold.duty_kW.sum(), cold_Ex_kW=cold.Ex_integral_kW.sum(), cold_Ex_lm_kW=cold.Ex_lm_kW.sum(),
               hot_Ex_excl_S3_exotherm_kW=hot.Ex_integral_kW.sum() - ex3, hot_Ex_below_650K_incl_kW=hot[hot.T_in_K <= 650].Ex_integral_kW.sum(),
               S3_exotherm_Q_kW=Q3, S3_exotherm_Ex_kW=ex3, S3_carnot_factor=f3, S1_endotherm_Q_kW=Q1, S1_endotherm_Ex_kW=ex1, S1_carnot_factor=f1,
               Q_ratio_S3_over_S1=Q3 / Q1, Ex_ratio_S3_over_S1=ex3 / ex1, S3_heat_as_1200K_heat_reversible_kW=ex3 / f1,
               S3_heat_needed_for_S1_reversible_kW=ex1 / f3)
    for eta in ETA_ORC:
        W = eta * Q3; agg[f"ORC_eta{eta:.2f}_power_kW"] = W; agg[f"ORC_eta{eta:.2f}_exergy_efficiency"] = W / ex3; agg[f"ORC_eta{eta:.2f}_over_carnot"] = eta / f3
    A = pd.DataFrame([agg]).T.rename(columns={0: "value"}); A.to_csv(os.path.join(RESULT, "X4_exergy_heat_aggregates.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["kind", "stream", "T_in_K", "T_out_K", "duty_kW", "duty_T4_kW", "T_lm_K", "Ex_lm_kW", "Ex_integral_kW", "Ex_over_Q", "sensible_kW", "latent_kW", "Ex_sensible_kW", "Ex_latent_kW", "T_eff_latent_K"]
    L = [f"X4 heat exergy (wall {time.time()-t0:.0f} s). EQUILIBRIUM streams (X4 representative: 2b, y = 0.60, h = 0.50). T0 = 298.15 K. Heat exergy only.",
         f"max |duty - T4 duty| = {(D.duty_kW - D.duty_T4_kW).abs().max():.2f} kW", "",
         D[cols].to_string(index=False, float_format=lambda x: f"{x:,.1f}" if abs(x) >= 10 else f"{x:.3f}"), "",
         A.to_string(float_format=lambda x: f"{x:,.3f}")]
    open(os.path.join(RESULT, "X4_exergy_heat_summary.txt"), "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
