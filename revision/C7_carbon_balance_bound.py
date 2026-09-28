"""
C7_carbon_balance_bound.py -- tracer-independent lower bound on the fixation of biogas-CO2 carbon.

Every carbon atom in the solid product came either from the waste CH4 or from the biogas CO2. If ALL waste-CH4 carbon
ended up in the solid, the CO2-derived share would be at its minimum; hence
    f_lb = (C_solid,total - C_in,CH4) / C_in,CO2   <=   f_tracer  <=  1,
with C_in,CH4 = 2,992 kmol/d (35.93 t C/d) and C_in,CO2 = biogas CO2 carbon (23.96 t C/d at y = 0.60). The bound uses
only the overall carbon balance (no origin tracking) and is exact when no CH4-derived carbon leaves in the purge.
Added as a column to the cases of F1 (h sweep, once-through and recycle), X4_T3_sensitivity (T3 sweep), X4S (dG and
pressure sweeps) and X4_fig2. Water removal (95 %) is unchanged by the refrigeration update (C1), so these values hold.

Outputs (revision/Result/): C7_carbon_balance_bound.csv, C7_summary.txt
Reproducibility: cd <repo>/revision ; ../.venv/bin/python C7_carbon_balance_bound.py   (seconds)
"""
import os
import numpy as np, pandas as pd
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
from P1_recycle_analysis import FRESH_CH4, tpd

C_CH4_IN = tpd("C(s)", FRESH_CH4)   # 35.93 t C/d

def bound(df, c_co2_col="CO2_total_carbon_tpd"):
    f_lb = (df["C_total_tpd"] - C_CH4_IN) / df[c_co2_col]
    out = df.copy(); out["C_in_CH4_tpd"] = C_CH4_IN; out["f_lb_carbon_balance"] = f_lb
    if "CO2_carbon_fixed_fraction" in out: out["f_tracer"] = out["CO2_carbon_fixed_fraction"]; out["tracer_minus_bound"] = out["f_tracer"] - f_lb
    return out

def main():
    rows = []
    F1 = pd.read_csv(os.path.join(RESULT, "F1_cases.csv")); f1 = F1[F1.config.str.startswith("2b") & (F1.value_type == "equilibrium")]
    b = bound(f1); rows.append(b.assign(source="F1_cases.csv (Table 6/7: h sweep, once-through and recycle)")[["source", "case", "y_CH4_biogas", "h_H2_to_CFR", "recycle", "C_total_tpd", "CO2_total_carbon_tpd", "f_lb_carbon_balance", "f_tracer", "tracer_minus_bound"]])
    T3 = pd.read_csv(os.path.join(RESULT, "X4_T3_sensitivity.csv")); b = bound(T3)
    rows.append(b.assign(source="X4_T3_sensitivity.csv (Table S2: T3 sweep)", y_CH4_biogas=0.60, h_H2_to_CFR=0.50, recycle=True)[["source", "case", "y_CH4_biogas", "h_H2_to_CFR", "recycle", "C_total_tpd", "CO2_total_carbon_tpd", "f_lb_carbon_balance", "f_tracer", "tracer_minus_bound"]])
    S = pd.read_csv(os.path.join(RESULT, "X4S_cases.csv")); b = bound(S)
    rows.append(b.assign(source="X4S_cases.csv (Table S3: dG sweep; pressure sweep)", y_CH4_biogas=0.60, h_H2_to_CFR=0.50)[["source", "case", "y_CH4_biogas", "h_H2_to_CFR", "recycle", "C_total_tpd", "CO2_total_carbon_tpd", "f_lb_carbon_balance", "f_tracer", "tracer_minus_bound"]])
    K = pd.read_csv(os.path.join(RESULT, "X4_fig2_kinetic_cases.csv")); b = bound(K)
    rows.append(b.assign(source="X4_fig2_kinetic_cases.csv (Fig. 2 kinetic bars)")[["source", "case", "y_CH4_biogas", "h_H2_to_CFR", "recycle", "C_total_tpd", "CO2_total_carbon_tpd", "f_lb_carbon_balance", "f_tracer", "tracer_minus_bound"]])
    D = pd.concat(rows, ignore_index=True); D.to_csv(os.path.join(RESULT, "C7_carbon_balance_bound.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    L = [f"C7 carbon-balance lower bound f_lb = (C_total - {C_CH4_IN:.3f}) / C_in,CO2 versus the species-resolved tracer. Kinetic rows are labelled in the case name.",
         f"Base case (2b, y = 0.60, h = 0.50, eq recycle): f_lb = {float(D[D.case=='2b_y0.60 eq recycle h=0.5'].f_lb_carbon_balance.iloc[0]):.5f}, tracer = {float(D[D.case=='2b_y0.60 eq recycle h=0.5'].f_tracer.iloc[0]):.5f}",
         f"max |tracer - bound| over all rows: {D.tracer_minus_bound.abs().max():.4f}; the bound never exceeds the tracer: {bool((D.tracer_minus_bound >= -1e-9).all())}", "",
         D.to_string(index=False, float_format=lambda x: f"{x:.4f}")]
    open(os.path.join(RESULT, "C7_summary.txt"), "w").write("\n".join(L)); print("\n".join(L[:4]))

if __name__ == "__main__":
    main()
