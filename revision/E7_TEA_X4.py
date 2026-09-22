# -*- coding: utf-8 -*-
"""
E7 / E11: Screening TEA of the X4 configuration, rebuilt with explicit assumptions
==================================================================================
FUPROC-D-26-00269 Major Revision (R1-6, R2-7, R4-4, E8, E10, X1-k, X3-f).

Scope: AACE Class 5 screening estimate (expected accuracy about -30 % / +50 %; AACE RP 18R-97 gives
-20 to -50 % / +30 to +100 % for Class 5). All unit costs are 2024 USD ASSUMPTIONS unless a source is
given; every line of the CAPEX table carries its basis. Yields are given on two bases and are never
mixed: EQUILIBRIUM (Gibbs, upper bound) and KINETIC design case (uncalibrated reduced models).

Process data (all read from the revision outputs, so the TEA follows the flowsheet exactly):
  F1_cases.csv        : carbon, water, exportable H2, CO2 fixed, stream sizes (equilibrium recycle, X4 y/h cases,
                        submitted once-through and recycle)
  T4_selfsufficiency  : minimum hot utility -> H2 fired for Stage 1/2 (pinch, dTmin 20 K)
  P3_selfsufficiency  : auxiliary electricity (excl./incl. biogas upgrading)
  kinetic X4 case     : solved here (F1 flowsheet, kinetic, h = 0.50, phi = 0 / 0.1 / 1, 1 atm), cached in E7_kinetic_X4.csv;
                        H2 fired for the kinetic case = (Q1 + Q2+) / 0.85 (T4 showed Q_H,min = Q1 + Q2+ within 1 %)

Configurations
  X4-grid      : X4, auxiliaries from the grid (base)
  X4-ORC       : X4, auxiliaries mostly from a bottoming cycle on the 650 K exotherm (T4: eta 0.25), residual from grid
  X4-PVaux     : X4, auxiliaries from PV sized for the continuous load (P4 capacity factor 0.20; storage/backup NOT costed)
  X4-upgrading : X4 with biogas upgrading (membrane) inside the boundary (P3/T4b)
  SUB-once     : submitted configuration (electrolysis 5 MW, DAC 10 t/d, once-through equilibrium 36.3 t/d) with the same method
  SUB-recycle  : submitted configuration with recycle (P1)

CAPEX method: purchased-equipment costs from capacity-based unit costs (six-tenths rule where scaled),
Lang factor 3.63 for a solid-fluid processing plant to fixed capital investment (Peters, Timmerhaus & West,
Plant Design and Economics for Chemical Engineers, 5th ed., 2003; Lang 1948), no working capital.
Sensitivity: Lang 3.10 (solid) / 4.74 (fluid).
OPEX: fixed O&M 4 % of FCI (assumption; maintenance 2-10 % of FCI in Peters & Timmerhaus), catalyst and
membrane replacement 5 %/y of the purchased cost of reactors + membranes (assumption), grid electricity
0.04 / 0.06 / 0.08 USD/kWh (submitted TEA), 330 operating days (submitted TEA), 20-year life, discount rate
8 % (5 / 10 %).
Revenues: solid carbon 180 / 350 / 700 USD/t (submitted scenarios, S2 grades), gate fee per E9 (chargeable
share 0 / 50 / 100 % of 800 t/d x 8 / 15 / 25 USD/t, avoided-cost basis), recovered water 0.30 / 0.75 / 1.50 USD/t,
net exportable H2 2 / 3 / 5 USD/kg (assumption). Carbon credits only as an Opportunity sensitivity
(0.5 creditable x 80 USD/tCO2; note that biogas CO2 is biogenic).

Reproducibility: cd <repo>/revision ; ../.venv/bin/python E7_TEA_X4.py   (~5 min first run: kinetic cases; seconds afterwards)
Outputs: E7_kinetic_X4.csv, E7_capex.csv, E7_annual.csv, E7_economics.csv, E7_tornado.csv, E7_tornado.png, E7_breakeven.csv, E7_summary.txt
"""
import os, sys, time
import numpy as np, pandas as pd
import cantera as ct
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES  # revision/_paths.py
import Workflow_cantera as W
import F1_full_biogas_CO2 as F1m
from P1_recycle_analysis import FRESH_CH4, tpd

ATM = ct.one_atm; DAYS = 330; LIFE = 20; ETA_FURNACE = 0.85; LHV = 33.32; NM3_PER_KMOL = 22.414
OUTS = {k: os.path.join(RESULT, v) for k, v in dict(kin="E7_kinetic_X4.csv", capex="E7_capex.csv", annual="E7_annual.csv", econ="E7_economics.csv",
        torn="E7_tornado.csv", png="E7_tornado.png", be="E7_breakeven.csv", txt="E7_summary.txt").items()}
F1 = pd.read_csv(os.path.join(RESULT, "F1_cases.csv")); T4 = pd.read_csv(os.path.join(RESULT, "T4_selfsufficiency.csv")); T4P = pd.read_csv(os.path.join(RESULT, "T4_pinch_results.csv"))
P3 = pd.read_csv(os.path.join(RESULT, "P3_selfsufficiency_map.csv")); P3 = P3[(P3.heat_recovery_fraction == 0) & (P3.power_source == "fuel cell 0.50")]

# ---------------- unit-cost assumptions (2024 USD) — each with basis
UC = dict(
    furnace_per_MW=0.50e6,            # Stage 1 H2-fired pyrolysis furnace incl. reactor section, per MW fired duty (assumption; fired-heater class)
    stage2_per_m3=50e3,               # 950 K catalytic reactor, alloy, per m3 (assumption)
    stage3_per_m3=40e3,               # 650 K methanation/Boudouard reactor, per m3 (assumption)
    h2_membrane_per_Nm3h=400.0,       # H2-selective polymeric membrane skid incl. vacuum pump, per Nm3/h feed (assumption)
    ch4_sep_per_Nm3h=1500.0,          # CH4/CO2/CO/H2 separation at Stage 3 outlet, biogas-upgrading class, per Nm3/h feed (assumption; SGC 2013:270 Fig. 5 order)
    upgrading_per_Nm3h=1500.0,        # biogas upgrading (membrane or amine), per Nm3/h raw biogas (assumption; SGC 2013:270 Fig. 5 order)
    hen_per_m2=600.0,                 # heat-exchanger network, alloy gas-gas, per m2 (assumption); area from T4 pinch target
    aircooler_per_kW=100.0,           # air-cooled condensers per kW duty (assumption)
    blower_per_kW=1500.0,             # blowers / vacuum pump per kW shaft (assumption)
    solids_handling=0.6e6,            # solid carbon discharge, cooling, conveying, bagging (assumption, lump sum)
    h2_export_compression=0.3e6,      # small H2 export compressor to 30 bar + purge/flare (assumption)
    orc_per_kWe=3000.0,               # bottoming ORC/steam, per kWe (Quoilin et al. 2013 report ~2,000-4,000 EUR/kW for < 1 MW; assumption)
    pv_per_kWp=900.0,                 # utility PV per kWp installed (assumption; no storage)
    electrolyser_per_kW=770.0,        # IRENA 2020 'average investment of USD 770/kW' (today) — submitted configuration only
    dac_capex_per_tpd=0.5e6,          # DAC 10 t/d: 0.5 MUSD per t/d capacity (assumption; TO BE VERIFIED)
    dac_opex_per_t=300.0,             # DAC operating cost per t CO2 (assumption; literature ranges ~125-600 USD/t; TO BE VERIFIED)
)
LANG = 3.63; LANG_LOW, LANG_HIGH = 3.10, 4.74; FIXED_OM = 0.04; CATALYST_REPL = 0.05
SCEN = {"Conservative": dict(cprice=180.0, fee=8.0, water=0.30, h2=2000.0, elec=0.08),
        "Base":         dict(cprice=350.0, fee=15.0, water=0.75, h2=3000.0, elec=0.06),
        "Opportunity":  dict(cprice=700.0, fee=25.0, water=1.50, h2=5000.0, elec=0.04)}
GATE_SHARE = {"0 %": 0.0, "50 %": 0.5, "100 %": 1.0}

# ---------------- kinetic X4 cases (cached)
def kin_case(args):
    label, cfg = args; rec, info = F1m.solve_recycle(cfg); row = F1m.make_row(cfg, rec, info, label)
    row["S2_inlet_kmol_d"] = sum(rec["feed2"].values()); row["S3_inlet_kmol_d"] = sum(rec["cfr_feed"].values()); row["pw2_kmol_d"] = sum(rec["pw2"].values()); row["pw3_kmol_d"] = sum(rec["pw3"].values())
    row["Q_fired_kW"] = (rec["Q1"] + max(rec["Q2"], 0.0)); return row
def kinetic_x4():
    if os.path.exists(OUTS["kin"]): return pd.read_csv(OUTS["kin"])
    F = F1m.feeds(); cases = []
    for phi in [0.0, 0.1, 1.0]:
        cases.append((f"2b_y0.60 kin recycle phi={phi:g} h=0.5", dict(config="2b_y0.60", feed=F["2b_y0.60"], mode="kinetic", phi=phi, k2=F1m.k2_from_phi(phi), n=None, variant="V1",
                                                                     P1_Pa=ATM, P2_Pa=ATM, P3_Pa=ATM, h=0.5, p=0.05, r=0.95)))
    with Pool(3) as pool: rows = pool.map(kin_case, cases)
    df = pd.DataFrame(rows); df.to_csv(OUTS["kin"], index=False); return df

# ---------------- process data per configuration/yield
def eq_row(case): return F1[F1.case == case].iloc[0]
def h2_fired_eq(case):
    m = T4[(T4.case == case) & (T4.dTmin_K == 20)]
    if len(m): return float(m.H2_fired_tpd.iloc[0]), float(m.Q_H_min_kW.iloc[0])
    q = float(T4P[(T4P.case == case) & (T4P.dTmin_K == 20)].Q_H_min_kW.iloc[0]); return q / ETA_FURNACE * 24 / LHV / 1000, q
def aux_kW(case, incl=False):
    m = P3[(P3.case == case) & (P3.boundary.str.startswith("incl" if incl else "excl"))]
    return float(m.aux_power_kW.iloc[0]) if len(m) else np.nan

def process_data(config, basis, kin=None, y=0.60):
    """Return dict of process quantities for a configuration and yield basis."""
    d = {}
    if config.startswith("X4"):
        case = f"2b_y{y:.2f} eq recycle h=0.5"
        if basis == "equilibrium":
            r = eq_row(case); h2f, qh = h2_fired_eq(case)
        else:
            r = kin[kin.phi == 1.0].iloc[0]; qh = float(r.Q_fired_kW); h2f = qh / ETA_FURNACE * 24 / LHV / 1000
        d.update(C_tpd=float(r.C_total_tpd), water_tpd=float(r.water_total_tpd), CO2_fixed_tpd=float(r.CO2_fixed_as_CO2_tpd), H2_export_raw=float(r.H2_net_exportable_tpd),
                 H2_fired=h2f, Q_fired_kW=qh, aux_kW=aux_kW(case, incl=(config == "X4-upgrading")), y=y,
                 raw_biogas_Nm3h=FRESH_CH4 * NM3_PER_KMOL / y / 24, S2_feed_Nm3h=float(r.S2_inlet_total_kmol_d) * NM3_PER_KMOL / 24,
                 S3_out_Nm3h=(float(r.pw3_kmol_d) if "pw3_kmol_d" in r else 2612.0) * NM3_PER_KMOL / 24,   # 2,612 kmol/d = P3 value for the eq rep case
                 hen_m2=1670.0 if basis == "equilibrium" else 1200.0, cooler_kW=5152.0 if basis == "equilibrium" else 3000.0,
                 blower_kW=86.0 + 136.0, orc_kWe=793.0 if basis == "equilibrium" else 30.0, C_stage3_tpd=float(r.C_stage3_tpd))
        d["aux_kW"] = float(d["aux_kW"]) if basis == "equilibrium" else float(aux_kW(case)) * 0.9
    else:  # submitted
        case = "submitted eq once-through h=0.35" if config == "SUB-once" else "submitted eq recycle h=0.35"
        r = eq_row(case); h2f, qh = h2_fired_eq(case)
        d.update(C_tpd=float(r.C_total_tpd), water_tpd=float(r.water_total_tpd), CO2_fixed_tpd=float(r.CO2_fixed_as_CO2_tpd), H2_export_raw=float(r.H2_net_exportable_tpd),
                 H2_fired=h2f, Q_fired_kW=qh, aux_kW=float(aux_kW("submitted eq recycle h=0.35")) + 5000.0, y=0.60,
                 raw_biogas_Nm3h=FRESH_CH4 * NM3_PER_KMOL / 0.60 / 24, S2_feed_Nm3h=float(r.S2_inlet_total_kmol_d) * NM3_PER_KMOL / 24, S3_out_Nm3h=1400.0,
                 hen_m2=870.0 if config == "SUB-once" else 1150.0, cooler_kW=3700.0, blower_kW=60.0 + 330.0, orc_kWe=0.0, C_stage3_tpd=float(r.C_stage3_tpd))
    return d

# ---------------- CAPEX build-up
def capex_table(config, pdata):
    items = []
    def it(name, cost, basis, source): items.append(dict(config=config, item=name, purchased_MUSD=cost / 1e6, basis=basis, source=source))
    it("Stage 1 pyrolysis furnace + reactor (H2-fired, 1200 K)", UC["furnace_per_MW"] * pdata["Q_fired_kW"] / 1000, f"{pdata['Q_fired_kW']/1000:.2f} MW fired x 0.50 MUSD/MW", "assumption (fired-heater class)")
    v2 = 3.0 * (pdata["S2_feed_Nm3h"] / 3600) * (950 / 273.15); it("Stage 2 reactor (950 K, tau 3 s)", UC["stage2_per_m3"] * v2, f"{v2:.0f} m3 x 50 kUSD/m3", "assumption")
    v3 = 3.0 * (pdata["S3_out_Nm3h"] * 1.3 / 3600) * (650 / 273.15); it("Stage 3 reactor (650 K, tau 3 s)", UC["stage3_per_m3"] * v3, f"{v3:.0f} m3 x 40 kUSD/m3", "assumption")
    it("H2 membrane (vacuum permeate)", UC["h2_membrane_per_Nm3h"] * pdata["S2_feed_Nm3h"], f"{pdata['S2_feed_Nm3h']:,.0f} Nm3/h feed x 400 USD/(Nm3/h)", "assumption")
    it("CH4 separation at Stage 3 outlet", UC["ch4_sep_per_Nm3h"] * pdata["S3_out_Nm3h"], f"{pdata['S3_out_Nm3h']:,.0f} Nm3/h x 1,500 USD/(Nm3/h)", "assumption (SGC 2013:270 Fig.5 order)")
    it("Heat exchanger network", UC["hen_per_m2"] * pdata["hen_m2"], f"{pdata['hen_m2']:,.0f} m2 (T4 area target, U=50) x 600 USD/m2", "assumption")
    it("Air-cooled condensers", UC["aircooler_per_kW"] * pdata["cooler_kW"], f"{pdata['cooler_kW']:,.0f} kW x 100 USD/kW", "assumption")
    it("Blowers / vacuum pump", UC["blower_per_kW"] * pdata["blower_kW"], f"{pdata['blower_kW']:.0f} kW x 1,500 USD/kW", "assumption")
    it("Solid carbon handling", UC["solids_handling"], "lump sum", "assumption")
    it("H2 export compression, purge/flare", UC["h2_export_compression"], "lump sum", "assumption")
    if config == "X4-ORC": it("Bottoming ORC on 650 K exotherm", UC["orc_per_kWe"] * pdata["orc_kWe"], f"{pdata['orc_kWe']:.0f} kWe x 3,000 USD/kWe", "Quoilin et al. 2013 range; assumption")
    if config == "X4-PVaux":
        kwp = pdata["aux_kW"] / 0.20; it("PV for auxiliaries (no storage)", UC["pv_per_kWp"] * kwp, f"{kwp:,.0f} kWp (CF 0.20, P4) x 900 USD/kWp", "assumption; backup not costed")
    if config == "X4-upgrading": it("Biogas upgrading (membrane), inside boundary", UC["upgrading_per_Nm3h"] * pdata["raw_biogas_Nm3h"], f"{pdata['raw_biogas_Nm3h']:,.0f} Nm3/h raw x 1,500 USD/(Nm3/h)", "assumption (SGC 2013:270 Fig.5 order)")
    if config.startswith("SUB"):
        it("Electrolyser 5.3 MW", UC["electrolyser_per_kW"] * 5263, "5,263 kW x 770 USD/kW", "IRENA 2020 (today average)")
        it("DAC 10 t/d", UC["dac_capex_per_tpd"] * 10, "10 t/d x 0.5 MUSD/(t/d)", "assumption, TO BE VERIFIED")
    df = pd.DataFrame(items); purchased = df.purchased_MUSD.sum()
    return df, purchased, purchased * LANG

# ---------------- annual cash flow
def annual(config, pdata, scen, gate_share=0.5, credits=False, overrides=None):
    o = dict(SCEN[scen]); o.update(overrides or {})
    C = pdata["C_tpd"]; rev_c = C * DAYS * o["cprice"]; rev_gate = 800.0 * gate_share * DAYS * o["fee"]; rev_w = pdata["water_tpd"] * DAYS * o["water"]
    h2_export = max(0.0, pdata["H2_export_raw"] - pdata["H2_fired"]); rev_h2 = h2_export * 1000 * DAYS * o["h2"] / 1000.0 * 1.0   # t/d -> kg/d ; USD/kg = o["h2"]/1000
    rev_h2 = h2_export * 1000 * DAYS * (o["h2"] / 1000.0)
    rev_credit = pdata["CO2_fixed_tpd"] * 0.5 * DAYS * 80.0 if credits else 0.0
    grid_kW = pdata["aux_kW"]
    if config == "X4-ORC": grid_kW = max(0.0, pdata["aux_kW"] - 0.25 * 3171.0 * (pdata["orc_kWe"] / 793.0))
    if config == "X4-PVaux": grid_kW = 0.0
    c_el = grid_kW * 24 * DAYS * o["elec"]
    c_dac = 10.0 * DAYS * UC["dac_opex_per_t"] if config.startswith("SUB") else 0.0
    return dict(config=config, scenario=scen, gate_share=gate_share, carbon_revenue=rev_c, gate_revenue=rev_gate, water_revenue=rev_w, H2_export_tpd=h2_export, H2_revenue=rev_h2,
                credit_revenue=rev_credit, electricity_cost=c_el, dac_opex=c_dac, grid_kW=grid_kW)

def economics(fci_musd, purchased_musd, react_memb_musd, ann, r=0.08, om=FIXED_OM):
    fci = fci_musd * 1e6; om_cost = om * fci; repl = CATALYST_REPL * react_memb_musd * 1e6
    cf = ann["carbon_revenue"] + ann["gate_revenue"] + ann["water_revenue"] + ann["H2_revenue"] + ann["credit_revenue"] - ann["electricity_cost"] - ann["dac_opex"] - om_cost - repl
    af = (1 - (1 + r) ** -LIFE) / r; npv = -fci + cf * af
    # IRR by bisection
    def npv_at(rate): return -fci + cf * ((1 - (1 + rate) ** -LIFE) / rate if rate > 1e-9 else LIFE)
    irr = np.nan
    if cf > 0:
        lo, hi = 1e-6, 2.0
        if npv_at(hi) < 0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if npv_at(mid) > 0: lo = mid
                else: hi = mid
            irr = 0.5 * (lo + hi)
        else: irr = hi
    dpb = np.nan
    if cf > 0:
        cum = 0.0
        for t in range(1, 200):
            cum += cf / (1 + r) ** t
            if cum >= fci: dpb = t - 1 + (fci - (cum - cf / (1 + r) ** t)) / (cf / (1 + r) ** t); break
    return dict(fixed_OM=om_cost, catalyst_membrane_replacement=repl, annual_cash_flow=cf, NPV_MUSD=npv / 1e6, IRR=irr, discounted_payback_y=dpb, simple_payback_y=(fci / cf if cf > 0 else np.nan))

def main():
    t0 = time.time(); kin = kinetic_x4()
    configs = ["X4-grid", "X4-ORC", "X4-PVaux", "X4-upgrading", "SUB-once", "SUB-recycle"]
    cap_rows, ann_rows, econ_rows = [], [], []
    PD = {}; CAP = {}; KIN_ASMODELLED = {}
    for cfg in configs:
        for basis in (["equilibrium", "kinetic"] if cfg.startswith("X4") else ["equilibrium"]):
            pdata = process_data(cfg, basis, kin); PD[(cfg, basis)] = pdata
            df, purchased, fci = capex_table(cfg, pdata); df["yield_basis"] = basis
            react_memb = df[df.item.str.contains("reactor|membrane|separation|upgrading", case=False)].purchased_MUSD.sum()
            if basis == "kinetic":
                # Yield-risk convention: SAME PLANT as the equilibrium design (equipment sized on the equilibrium recycle flows),
                # only yields, utilities and sales follow the kinetic case. The CAPEX that the kinetic loop 'as modelled' would
                # imply (huge CO recycle -> separation equipment) is kept for information only.
                df["note"] = "as-modelled kinetic loop (information only; TEA uses equilibrium-sized plant)"
                KIN_ASMODELLED[cfg] = (purchased, fci)
                purchased, fci, react_memb = CAP[(cfg, "equilibrium")]
            cap_rows.append(df); CAP[(cfg, basis)] = (purchased, fci, react_memb)
            for scen in SCEN:
                for gname, gs in GATE_SHARE.items():
                    for credits in ([False, True] if scen == "Opportunity" else [False]):
                        a = annual(cfg, pdata, scen, gs, credits); e = economics(fci, purchased, react_memb, a)
                        a.update(yield_basis=basis, credits=credits, purchased_MUSD=purchased, FCI_MUSD=fci, C_tpd=pdata["C_tpd"], water_tpd=pdata["water_tpd"], CO2_fixed_tpd=pdata["CO2_fixed_tpd"], aux_kW=pdata["aux_kW"], H2_fired_tpd=pdata["H2_fired"], **e)
                        ann_rows.append(a)
    capex = pd.concat(cap_rows, ignore_index=True); capex.to_csv(OUTS["capex"], index=False)
    ann = pd.DataFrame(ann_rows); ann.to_csv(OUTS["annual"], index=False)
    # tornado: X4-grid, Base, equilibrium, gate 50 %
    cfg, basis = "X4-grid", "equilibrium"; pdata = PD[(cfg, basis)]; purchased, fci, rm = CAP[(cfg, basis)]
    base_e = economics(fci, purchased, rm, annual(cfg, pdata, "Base", 0.5))["NPV_MUSD"]
    torn = []
    def add(param, lo_label, lo_val, hi_label, hi_val): torn.append(dict(parameter=param, low_case=lo_label, NPV_low=lo_val, high_case=hi_label, NPV_high=hi_val, base=base_e, swing=abs(hi_val - lo_val)))
    e = lambda **kw: economics(fci * kw.pop("fci_mult", 1.0), purchased, rm, annual(cfg, pdata, "Base", kw.pop("gs", 0.5), overrides=kw.pop("ov", None)), r=kw.pop("r", 0.08), om=kw.pop("om", FIXED_OM))["NPV_MUSD"]
    add("Carbon price (245 / 455 USD/t, ±30 %)", "245", e(ov=dict(cprice=245)), "455", e(ov=dict(cprice=455)))
    add("Gate fee basis (0 % / 100 % of 800 t/d at 15 USD/t)", "0 %", e(gs=0.0), "100 %", e(gs=1.0))
    add("CAPEX (Lang 3.10 / 4.74)", "3.10", e(fci_mult=LANG_LOW / LANG), "4.74", e(fci_mult=LANG_HIGH / LANG))
    add("Discount rate (10 % / 5 %)", "10 %", e(r=0.10), "5 %", e(r=0.05))
    kp = PD[(cfg, "kinetic")]; pk, fk, rk = CAP[(cfg, "kinetic")]
    add("Yield (kinetic design case / equilibrium)", "kinetic 36 t/d", economics(fk, pk, rk, annual(cfg, kp, "Base", 0.5))["NPV_MUSD"], "equilibrium 59 t/d", base_e)
    add("Electricity price (0.08 / 0.04 USD/kWh)", "0.08", e(ov=dict(elec=0.08)), "0.04", e(ov=dict(elec=0.04)))
    add("Fixed O&M (5 % / 3 % of FCI)", "5 %", e(om=0.05), "3 %", e(om=0.03))
    for yv, lab in [(0.55, "y=0.55"), (0.65, "y=0.65")]:
        pdy = process_data(cfg, "equilibrium", kin, y=yv); py, fy, ry = capex_table(cfg, pdy)[1], capex_table(cfg, pdy)[2], 0.0
        ry = capex_table(cfg, pdy)[0].pipe(lambda d: d[d.item.str.contains("reactor|membrane|separation", case=False)].purchased_MUSD.sum())
        globals()[f"npv_{lab}"] = economics(fy, py, ry, annual(cfg, pdy, "Base", 0.5))["NPV_MUSD"]
    add("Biogas CH4 fraction y (0.55 / 0.65)", "0.55", globals()["npv_y=0.55"], "0.65", globals()["npv_y=0.65"])
    add("H2 price (2 / 5 USD/kg)", "2", e(ov=dict(h2=2000)), "5", e(ov=dict(h2=5000)))
    td = pd.DataFrame(torn).sort_values("swing"); td.to_csv(OUTS["torn"], index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5)); yy = np.arange(len(td))
    for i, r in enumerate(td.itertuples()):
        ax.plot([r.NPV_low, r.NPV_high], [i, i], marker="o"); ax.annotate(r.low_case, (r.NPV_low, i), textcoords="offset points", xytext=(-4, 4), ha="right", fontsize=7); ax.annotate(r.high_case, (r.NPV_high, i), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axvline(base_e, ls="--", color="k", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels(td.parameter, fontsize=8); ax.set_xlabel("NPV [MUSD] (8 %, 20 y)"); ax.set_title(f"E11 tornado — X4-grid, Base, equilibrium yield, gate 50 % (base NPV {base_e:.1f} MUSD)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTS["png"], dpi=160); plt.close(fig)
    # break-even carbon price (NPV = 0), Base otherwise, X4-grid, both yields, gate 0/50/100 %
    be = []
    for basis in ["equilibrium", "kinetic"]:
        pdb = PD[(cfg, basis)]; pb, fb, rb = CAP[(cfg, basis)]
        for gname, gs in GATE_SHARE.items():
            lo, hi = 0.0, 5000.0
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if economics(fb, pb, rb, annual(cfg, pdb, "Base", gs, overrides=dict(cprice=mid)))["NPV_MUSD"] > 0: hi = mid
                else: lo = mid
            be.append(dict(config=cfg, yield_basis=basis, gate_share=gname, breakeven_carbon_price_USD_per_t=0.5 * (lo + hi)))
    bed = pd.DataFrame(be); bed.to_csv(OUTS["be"], index=False)
    # summary
    L = [f"E7 TEA (wall {time.time()-t0:.0f} s). Class 5 screening. Yields: equilibrium (upper bound) and kinetic design case (uncalibrated)."]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 400, "display.float_format", lambda v: f"{v:,.2f}"):
        L.append("\n--- CAPEX, X4-grid, equilibrium ---\n" + capex[(capex.config == "X4-grid") & (capex.yield_basis == "equilibrium")][["item", "purchased_MUSD", "basis", "source"]].to_string(index=False))
        tot = pd.DataFrame([dict(config=c, yield_basis=b, purchased_MUSD=CAP[(c, b)][0], FCI_MUSD=CAP[(c, b)][1]) for (c, b) in CAP]); L.append("\n--- Totals (Lang 3.63); kinetic rows use the equilibrium-sized plant ---\n" + tot.to_string(index=False))
        L.append("\n--- For information: CAPEX the kinetic loop 'as modelled' would imply (CO recycle inflates separation equipment) ---\n" + pd.DataFrame([dict(config=c, purchased_MUSD=v[0], FCI_MUSD=v[1]) for c, v in KIN_ASMODELLED.items()]).to_string(index=False))
        sel = ann[(ann.gate_share == 0.5) & (~ann.credits)]
        L.append("\n--- Economics, gate 50 % ---\n" + sel[["config", "yield_basis", "scenario", "C_tpd", "carbon_revenue", "gate_revenue", "water_revenue", "H2_export_tpd", "H2_revenue", "electricity_cost", "dac_opex", "fixed_OM", "catalyst_membrane_replacement", "annual_cash_flow", "FCI_MUSD", "NPV_MUSD", "IRR", "discounted_payback_y"]].to_string(index=False))
        L.append("\n--- X4-grid, all gate shares ---\n" + ann[(ann.config == "X4-grid")][["yield_basis", "scenario", "gate_share", "credits", "annual_cash_flow", "NPV_MUSD", "IRR", "discounted_payback_y"]].to_string(index=False))
        L.append("\n--- Tornado ---\n" + td.to_string(index=False)); L.append("\n--- Break-even carbon price ---\n" + bed.to_string(index=False))
        L.append("\n--- Kinetic X4 cases (h = 0.50) ---\n" + kin[["case", "C_total_tpd", "C_from_CO2_total_tpd", "water_total_tpd", "H2_net_exportable_tpd", "Q1_kW", "Q2_kW", "Q3_kW", "Q_fired_kW"]].to_string(index=False))
    with open(OUTS["txt"], "w") as fh: fh.write("\n".join(L) + "\n")
    print("\n".join(L)); print("Saved:", *OUTS.values())

if __name__ == "__main__":
    main()
