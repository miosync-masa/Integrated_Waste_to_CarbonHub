# -*- coding: utf-8 -*-
# ============================================================
# Waste/biogas -> CH4 pyrolysis -> RWGS -> separation -> CFR
# Refactored thermodynamics + unit-operations hybrid model
# ============================================================

# If needed in Colab/Jupyter:
# !pip install cantera

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct

# ------------------------------------------------------------
# User inputs
# ------------------------------------------------------------
CH4_tpd = 48.0
CO2_from_biogas_tpd = 32.0
CO2_from_DAC_tpd    = 10.0
CO2_total_tpd = CO2_from_biogas_tpd + CO2_from_DAC_tpd

T_pyro = 1200.0
P_pyro = 1.0 * ct.one_atm

T_rwgs = 950.0
P_rwgs = 1.0 * ct.one_atm

T_cfr  = 650.0
P_cfr  = 1.0 * ct.one_atm

water_remove_frac = 0.95
h2_to_cfr_frac    = 0.35
co2_to_cfr_frac   = 1.00
co_to_cfr_frac    = 1.00
ch4_to_cfr_frac   = 1.00

solar_power_kW   = 5000.0
electrolyzer_eff = 0.70

eta_cfr_realization = 1.0

TRACE = 1e-14

# ------------------------------------------------------------
# Pyrolysis kinetic model inputs
# ------------------------------------------------------------
pyro_tau_s   = 3.0       # nominal residence time [s]
pyro_n_steps = 1000      # integration steps
pyro_kref    = 4.0       # [mol/m^3/s] at Tref (fitting knob)
pyro_Ea      = 180e3     # [J/mol] (fitting knob)
pyro_Tref    = 1200.0    # [K]

# Pyrolysis tau sweep grid
pyro_tau_grid = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0])
pyro_T_grid   = np.array([1000, 1050, 1100, 1150, 1200, 1300])

# ------------------------------------------------------------
# RWGS kinetic model inputs
# ------------------------------------------------------------
rwgs_tau_s   = 3.0      # residence time [s]
rwgs_n_steps = 400      # integration steps
rwgs_kref    = 0.05     # [m^3/mol/s] at Tref (fitting knob)
rwgs_Ea      = 80e3     # [J/mol] (fitting knob)
rwgs_Tref    = 950.0    # [K]

# RWGS tau sweep grid
rwgs_tau_grid = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0])
rwgs_T_grid   = np.array([800, 850, 900, 950, 1000, 1050])

# ------------------------------------------------------------
# CFR kinetic model inputs (4-reaction network)
# ------------------------------------------------------------
cfr_tau_s   = 3.0
cfr_n_steps = 2000

# Starter kinetic knobs (not literature-calibrated)
# 3 independent reactions (CO_methanation removed: linearly dependent on CO_carbon + reverse CH4_cracking)
cfr_kref_co2_meth = 0.25   # CO2 + 4H2 <=> CH4 + 2H2O
cfr_Ea_co2_meth   = 100e3

cfr_kref_co_carb  = 0.08   # CO + H2 <=> C(s) + H2O
cfr_Ea_co_carb    = 110e3

cfr_kref_ch4_carb = 0.01   # CH4 <=> C(s) + 2H2
cfr_Ea_ch4_carb   = 160e3

T_cfr_grid  = np.linspace(575, 750, 8)
water_grid  = np.linspace(0.85, 0.99, 5)
h2frac_grid = np.linspace(0.15, 0.40, 6)

# ------------------------------------------------------------
# Validation-ready series definitions
# ------------------------------------------------------------
pyro_validation_T = 1200.0
rwgs_validation_T = 950.0
cfr_validation_T_grid = np.array([750.0, 775.0, 800.0, 825.0, 850.0])
cfr_validation_tau_s  = 3.0

EQ_TRIES_GAS = [
    dict(solver="element_potential", rtol=1e-7, max_steps=50000,  max_iter=100, estimate_equil=-1, log_level=0),
    dict(solver="gibbs",             rtol=1e-6, max_steps=150000, max_iter=100, estimate_equil=-1, log_level=0),
]
EQ_TRIES_MULTIPHASE = [
    dict(solver="gibbs", rtol=1e-6, max_steps=200000, max_iter=100, estimate_equil=-1, log_level=0),
]

GAS_REF = ct.Solution("gri30.yaml")
SOLID_REF = ct.Solution("graphite.yaml")
MW = {sp: GAS_REF.molecular_weights[GAS_REF.species_index(sp)] for sp in GAS_REF.species_names}
MW_C_SOLID = SOLID_REF.molecular_weights[0]
ELEMENTS = ["C", "H", "O"]

def tpd_to_kmol_per_day(tpd, mw): return tpd * 1000.0 / mw
def kmol_per_day_to_tpd(kmol_d, mw): return kmol_d * mw / 1000.0
def solar_h2_kmol_per_day(power_kW, eff=0.70, LHV=120e6):
    return (power_kW*1000*86400*eff/LHV) / MW["H2"]

def clean_species_dict(d, cutoff=TRACE):
    return {k: float(v) for k, v in d.items() if abs(v) > cutoff}

def normalize_mole_dict(d):
    d = clean_species_dict(d); total = sum(d.values())
    if total <= 0: raise ValueError("zero moles")
    return {k: v/total for k,v in d.items()}, total

def dataframe_from_species(sm):
    rows = []
    for sp, n in sorted(clean_species_dict(sm).items(), key=lambda kv: kv[1], reverse=True):
        mw = MW.get(sp, np.nan)
        rows.append([sp, n, kmol_per_day_to_tpd(n, mw) if np.isfinite(mw) else np.nan])
    return pd.DataFrame(rows, columns=["Species","kmol/day","t/day"])

def elemental_inventory_gas(sm):
    inv = {e:0 for e in ELEMENTS}
    for sp, n in sm.items():
        if sp not in GAS_REF.species_names: continue
        k = GAS_REF.species_index(sp)
        for e in ELEMENTS: inv[e] += n * GAS_REF.n_atoms(k, GAS_REF.element_index(e))
    return inv

def elemental_inventory_total(gsm, sc=0): inv = elemental_inventory_gas(gsm); inv["C"]+=sc; return inv

def element_balance_table(fg, og, oc=0):
    inv_in = elemental_inventory_total(fg,0); inv_out = elemental_inventory_total(og,oc)
    rows = []
    for e in ELEMENTS:
        err = inv_out[e]-inv_in[e]; rel = err/inv_in[e] if abs(inv_in[e])>0 else np.nan
        rows.append([e, inv_in[e], inv_out[e], err, rel])
    return pd.DataFrame(rows, columns=["Element","In (kmol-atoms/d)","Out (kmol-atoms/d)","Abs Error","Rel Error"])

def total_stream_enthalpy_J_per_day(gsm, sc, T, P):
    H = 0; gsm = clean_species_dict(gsm)
    if gsm:
        g = ct.Solution("gri30.yaml"); x,n = normalize_mole_dict(gsm); g.TPX = T,P,x
        H += g.enthalpy_mole * n
    if sc > 0:
        s = ct.Solution("graphite.yaml"); s.TP = T,P; H += s.enthalpy_mole * sc
    return H

def ordered_temperature_grid(Tg, Tr): return sorted(Tg, key=lambda T: abs(T-Tr))

def _equilibrate_mixture(phases, T, P, gsn, ssn=None, ism=None, tries=None):
    mix = ct.Mixture(phases); mix.T=T; mix.P=P
    if ism is not None:
        try: mix.species_moles = ism
        except: pass
    for kw in tries:
        try:
            mix.equilibrate("TP", **kw)
            asp = dict(zip(mix.species_names, mix.species_moles)); asp = clean_species_dict(asp)
            gs = {sp: asp.get(sp,0) for sp in gsn if asp.get(sp,0) > TRACE}
            sc = asp.get(ssn, 0) if ssn else 0
            return {"T_K":T,"P_Pa":P,"gas_kmol_d":gs,"Csolid_kmol_d":sc,"all_species_kmol_d":asp,"species_moles_vector":mix.species_moles,"converged":1,"error":None}
        except Exception as e: last_err = e
    return {"T_K":T,"P_Pa":P,"gas_kmol_d":{},"Csolid_kmol_d":np.nan,"all_species_kmol_d":{},"species_moles_vector":None,"converged":0,"error":str(last_err)}

def equilibrate_gas(feed, T, P, ism=None):
    g = ct.Solution("gri30.yaml"); x,n = normalize_mole_dict(feed); g.TPX=T,P,x
    return _equilibrate_mixture([(g,n)],T,P,g.species_names,None,ism,EQ_TRIES_GAS)

def equilibrate_gas_plus_graphite(feed, T, P, ism=None):
    g = ct.Solution("gri30.yaml"); s = ct.Solution("graphite.yaml")
    x,n = normalize_mole_dict(feed); g.TPX=T,P,x
    return _equilibrate_mixture([(g,n),(s,0)],T,P,g.species_names,s.species_names[0],ism,EQ_TRIES_MULTIPHASE)

def remove_species(stream, species, frac):
    out=dict(stream); n=out.get(species,0); take=frac*n; out[species]=n-take
    if out[species]<TRACE: out.pop(species,None)
    return clean_species_dict(out), clean_species_dict({species:take})

def membrane_split(stream, kfm=None, dk=1.0):
    if kfm is None: kfm={}
    ret={}; perm={}
    for sp,n in clean_species_dict(stream).items():
        f=max(0,min(1,kfm.get(sp,dk))); ret[sp]=n*f; perm[sp]=n*(1-f)
    return clean_species_dict(ret), clean_species_dict(perm)

def run_stage1_pyrolysis(ch4, Tp, Pp):
    feed={"CH4":tpd_to_kmol_per_day(ch4,MW["CH4"])}
    res=equilibrate_gas_plus_graphite(feed,Tp,Pp)
    if not res["converged"]: raise RuntimeError(f"Stage 1 failed: {res.get('error')}")
    fH=total_stream_enthalpy_J_per_day(feed,0,Tp,Pp)
    pH=total_stream_enthalpy_J_per_day(res["gas_kmol_d"],res["Csolid_kmol_d"],Tp,Pp)
    return {"feed":feed,"result":res,"Q_kW":(pH-fH)/86400/1000,"balance":element_balance_table(feed,res["gas_kmol_d"],res["Csolid_kmol_d"])}

# ------------------------------------------------------------
# Pyrolysis kinetic model (minimal PFR approximation)
# ------------------------------------------------------------
def pyro_Kp_from_equilibrium(T, P):
    """
    Kp for CH4 <=> C(s) + 2H2, reconstructed from equilibrium solver.
    Kp = (p_H2/P0)^2 / (p_CH4/P0), activity of graphite = 1.
    """
    ref_feed = {"CH4": 1.0}
    res = equilibrate_gas_plus_graphite(ref_feed, T, P)
    if not res["converged"]:
        raise RuntimeError(f"pyro_Kp equilibrium failed: {res.get('error')}")
    gas = clean_species_dict(res["gas_kmol_d"])
    total_gas = sum(gas.values())
    if total_gas <= 0: raise ValueError("pyro_Kp: zero gas at equilibrium")
    y_ch4 = gas.get("CH4", 0.0) / total_gas
    y_h2  = gas.get("H2", 0.0) / total_gas
    p_ratio = P / ct.one_atm
    pch4_hat = max(y_ch4 * p_ratio, TRACE)
    ph2_hat  = max(y_h2  * p_ratio, TRACE)
    return (ph2_hat ** 2) / pch4_hat

def pyro_k_forward(T, kref=4.0, Ea=180e3, Tref=1200.0):
    """Arrhenius scaling for forward CH4 decomposition."""
    return kref * np.exp(-Ea / R_MOL * (1.0/T - 1.0/Tref))

def run_stage1_pyrolysis_kinetic(ch4_tpd, Tp, Pp,
                                  tau_s=3.0, n_steps=1000,
                                  kref=4.0, Ea=180e3, Tref=1200.0):
    """
    Minimal kinetic CH4 pyrolysis reactor:
    isothermal, isobaric, PFR approximation, single reversible reaction.
    CH4 <=> C(s) + 2H2, Kp from equilibrium solver.
    """
    feed = {"CH4": tpd_to_kmol_per_day(ch4_tpd, MW["CH4"])}
    feed = clean_species_dict(feed)

    reactive = ["CH4", "H2"]
    all_species = sorted(set(feed.keys()) | set(reactive))

    # inlet molar flow [mol/s]
    F = {sp: kmol_per_day_to_mol_per_s(feed.get(sp, 0.0)) for sp in all_species}
    F_Csolid = 0.0  # solid carbon [mol/s]

    Ftot_in = sum(F.values())
    if Ftot_in <= 0: raise ValueError("Stage 1 feed has zero total flow")

    # nominal reactor volume from inlet conditions
    vdot_in = Ftot_in * R_MOL * Tp / Pp
    V_reactor = tau_s * vdot_in
    dV = V_reactor / n_steps

    Kp = pyro_Kp_from_equilibrium(Tp, Pp)
    kf = pyro_k_forward(Tp, kref=kref, Ea=Ea, Tref=Tref)

    for _ in range(n_steps):
        Ftot = sum(max(F.get(sp, 0.0), 0.0) for sp in all_species)
        if Ftot <= 0: break

        y = {sp: max(F.get(sp, 0.0), 0.0) / Ftot for sp in all_species}
        p_ratio = Pp / ct.one_atm
        pch4_hat = max(y.get("CH4", 0.0) * p_ratio, 0.0)
        ph2_hat  = max(y.get("H2", 0.0) * p_ratio, 0.0)

        # net rate [mol/m^3/s]
        r = kf * (pch4_hat - (ph2_hat ** 2) / Kp)

        # prevent overshoot
        if r >= 0:
            r = min(r, F.get("CH4", 0.0) / dV)
        else:
            r = max(r, -F.get("H2", 0.0) / (2.0 * dV), -F_Csolid / dV)

        F["CH4"] = max(F.get("CH4", 0.0) - r * dV, 0.0)
        F["H2"]  = max(F.get("H2", 0.0) + 2.0 * r * dV, 0.0)
        F_Csolid = max(F_Csolid + r * dV, 0.0)

    gas_out = clean_species_dict({
        sp: mol_per_s_to_kmol_per_day(F.get(sp, 0.0)) for sp in all_species
    })
    Csolid_out = mol_per_s_to_kmol_per_day(F_Csolid)

    fH = total_stream_enthalpy_J_per_day(feed, 0, Tp, Pp)
    pH = total_stream_enthalpy_J_per_day(gas_out, Csolid_out, Tp, Pp)

    ch4_in = feed.get("CH4", 0.0); ch4_out = gas_out.get("CH4", 0.0)
    x_ch4 = (ch4_in - ch4_out) / ch4_in if ch4_in > 0 else np.nan

    return {
        "feed": feed,
        "result": {"T_K": Tp, "P_Pa": Pp, "gas_kmol_d": gas_out,
                   "Csolid_kmol_d": Csolid_out, "converged": 1, "error": None,
                   "Kp": Kp, "kf": kf, "tau_s": tau_s},
        "Q_kW": (pH - fH) / 86400.0 / 1000.0,
        "balance": element_balance_table(feed, gas_out, Csolid_out),
        "X_CH4": x_ch4,
    }

def run_pyro_tau_sweep(ch4_tpd, T_grid, tau_grid, Pp,
                       kref=4.0, Ea=180e3, Tref=1200.0):
    """Sweep tau x T for pyrolysis kinetic model."""
    rows = []
    for Tp in T_grid:
        # equilibrium reference
        eq = run_stage1_pyrolysis(ch4_tpd, Tp, Pp)
        ch4_in = eq["feed"].get("CH4", 0.0)
        ch4_eq = eq["result"]["gas_kmol_d"].get("CH4", 0.0)
        x_eq = (ch4_in - ch4_eq) / ch4_in if ch4_in > 0 else np.nan

        for tau in tau_grid:
            res = run_stage1_pyrolysis_kinetic(ch4_tpd, Tp, Pp, tau_s=tau,
                                               kref=kref, Ea=Ea, Tref=Tref)
            rows.append({
                "T_K": Tp, "tau_s": tau,
                "X_CH4_kinetic": res["X_CH4"],
                "X_CH4_equilibrium": x_eq,
                "approach_to_eq": res["X_CH4"] / x_eq if x_eq > 0 else np.nan,
                "C_solid_tpd": kmol_per_day_to_tpd(res["result"]["Csolid_kmol_d"], MW_C_SOLID),
                "Q_kW": res["Q_kW"],
            })
    return pd.DataFrame(rows)

def build_rwgs_feed(s1, co2t, spow, eeff):
    co2k=tpd_to_kmol_per_day(co2t,MW["CO2"]); h2s=solar_h2_kmol_per_day(spow,eeff)
    feed={"CO2":co2k,"H2":s1["result"]["gas_kmol_d"].get("H2",0)+h2s}
    ch4c=s1["result"]["gas_kmol_d"].get("CH4",0)
    if ch4c>TRACE: feed["CH4"]=ch4c
    return clean_species_dict(feed), h2s

def run_stage2_rwgs(feed, Tr, Pr):
    res=equilibrate_gas(feed,Tr,Pr)
    if not res["converged"]: raise RuntimeError(f"Stage 2 failed: {res.get('error')}")
    fH=total_stream_enthalpy_J_per_day(feed,0,Tr,Pr)
    pH=total_stream_enthalpy_J_per_day(res["gas_kmol_d"],0,Tr,Pr)
    return {"feed":feed,"result":res,"Q_kW":(pH-fH)/86400/1000,"balance":element_balance_table(feed,res["gas_kmol_d"],0)}

# ------------------------------------------------------------
# RWGS kinetic model (minimal PFR approximation)
# ------------------------------------------------------------
R_MOL = ct.gas_constant / 1000.0  # J/mol/K

def kmol_per_day_to_mol_per_s(x):
    return x * 1000.0 / 86400.0

def mol_per_s_to_kmol_per_day(x):
    return x * 86400.0 / 1000.0

def rwgs_Kc(T):
    """Equilibrium constant for CO2 + H2 <=> CO + H2O (Kp=Kc since dn=0)."""
    g = ct.Solution("gri30.yaml")
    g.TPX = T, ct.one_atm, "CO2:1.0"
    i = {sp: g.species_index(sp) for sp in ["CO2","H2","CO","H2O"]}
    dG_RT = (g.standard_gibbs_RT[i["CO"]] + g.standard_gibbs_RT[i["H2O"]]
           - g.standard_gibbs_RT[i["CO2"]] - g.standard_gibbs_RT[i["H2"]])
    return np.exp(-dG_RT)

def rwgs_k_forward(T, kref=0.05, Ea=80e3, Tref=950.0):
    """Arrhenius scaling around Tref. kref is a fitting parameter."""
    return kref * np.exp(-Ea / R_MOL * (1.0/T - 1.0/Tref))

def run_stage2_rwgs_kinetic(feed, Tr, Pr, tau_s=3.0, n_steps=400,
                            kref=0.05, Ea=80e3, Tref=950.0):
    """
    Minimal kinetic RWGS reactor:
    isothermal, isobaric, PFR-in-residence-time, single reversible reaction.
    """
    feed = clean_species_dict(feed)
    reactive = ["CO2","H2","CO","H2O"]
    all_sp = sorted(set(feed.keys()) | set(reactive))

    # inlet molar flow [mol/s]
    F_in = {sp: kmol_per_day_to_mol_per_s(feed.get(sp,0)) for sp in all_sp}
    Ftot = sum(F_in.values())
    if Ftot <= 0: raise ValueError("RWGS feed has zero total flow")

    # ideal-gas volumetric flow [m^3/s]
    vdot = Ftot * R_MOL * Tr / Pr

    # inlet concentrations [mol/m^3]
    C = {sp: F_in.get(sp,0) / vdot for sp in all_sp}
    for sp in reactive: C.setdefault(sp, 0.0)

    Kc = rwgs_Kc(Tr)
    kf = rwgs_k_forward(Tr, kref=kref, Ea=Ea, Tref=Tref)
    dt = tau_s / n_steps

    for _ in range(n_steps):
        c_co2 = max(C.get("CO2",0), 0)
        c_h2  = max(C.get("H2",0),  0)
        c_co  = max(C.get("CO",0),  0)
        c_h2o = max(C.get("H2O",0), 0)

        r = kf * (c_co2 * c_h2 - (c_co * c_h2o) / Kc)

        # prevent overshoot
        if r >= 0:
            r = min(r, c_co2/dt, c_h2/dt)
        else:
            r = max(r, -c_co/dt, -c_h2o/dt)

        C["CO2"] = c_co2 - r*dt
        C["H2"]  = c_h2  - r*dt
        C["CO"]  = c_co  + r*dt
        C["H2O"] = c_h2o + r*dt

    # outlet molar flow [mol/s] -> kmol/day
    F_out = {}
    for sp in all_sp:
        F_out[sp] = max(C[sp],0)*vdot if sp in reactive else F_in[sp]

    gas_out = clean_species_dict({sp: mol_per_s_to_kmol_per_day(F_out[sp]) for sp in F_out})

    fH = total_stream_enthalpy_J_per_day(feed, 0, Tr, Pr)
    pH = total_stream_enthalpy_J_per_day(gas_out, 0, Tr, Pr)

    co2_in = feed.get("CO2",0); co2_out = gas_out.get("CO2",0)
    x_co2 = (co2_in - co2_out)/co2_in if co2_in > 0 else np.nan

    return {
        "feed": feed,
        "result": {"T_K":Tr, "P_Pa":Pr, "gas_kmol_d":gas_out,
                   "converged":1, "error":None, "Kc":Kc, "kf":kf, "tau_s":tau_s},
        "Q_kW": (pH - fH)/86400/1000,
        "balance": element_balance_table(feed, gas_out, 0),
        "X_CO2": x_co2,
    }

def run_rwgs_tau_sweep(feed, T_grid, tau_grid, Pr,
                       kref=0.05, Ea=80e3, Tref=950.0):
    """Sweep tau x T for RWGS kinetic model. Returns DataFrame."""
    rows = []
    for Tr in T_grid:
        # equilibrium reference
        Kc = rwgs_Kc(Tr)
        eq = run_stage2_rwgs(dict(feed), Tr, Pr)
        co2_eq = eq["result"]["gas_kmol_d"].get("CO2",0)
        co2_in = feed.get("CO2",0)
        x_eq = (co2_in - co2_eq)/co2_in if co2_in > 0 else np.nan

        for tau in tau_grid:
            res = run_stage2_rwgs_kinetic(dict(feed), Tr, Pr, tau_s=tau,
                                          kref=kref, Ea=Ea, Tref=Tref)
            rows.append({
                "T_K": Tr, "tau_s": tau,
                "X_CO2_kinetic": res["X_CO2"],
                "X_CO2_equilibrium": x_eq,
                "approach_to_eq": res["X_CO2"]/x_eq if x_eq > 0 else np.nan,
                "Q_kW": res["Q_kW"],
                "Kc": Kc,
            })
    return pd.DataFrame(rows)

def build_cfr_feed(s2out, wrf, h2f, co2f, cof, ch4f):
    pws,rw = remove_species(s2out,"H2O",wrf)
    cfrf,mp = membrane_split(pws,{"H2":h2f,"CO2":co2f,"CO":cof,"CH4":ch4f},1.0)
    return {"post_water_sep":pws,"recovered_water":rw,"cfr_feed":cfrf,"membrane_permeate":mp}

def run_stage3_cfr(feed, Tc, Pc, eta=1.0, ism=None):
    res=equilibrate_gas_plus_graphite(feed,Tc,Pc,ism)
    if not res["converged"]:
        return {"feed":feed,"result":res,"Q_kW":np.nan,"balance":None,"C_realized_kmol_d":np.nan}
    fH=total_stream_enthalpy_J_per_day(feed,0,Tc,Pc)
    pH=total_stream_enthalpy_J_per_day(res["gas_kmol_d"],res["Csolid_kmol_d"],Tc,Pc)
    return {"feed":feed,"result":res,"Q_kW":(pH-fH)/86400/1000,
            "balance":element_balance_table(feed,res["gas_kmol_d"],res["Csolid_kmol_d"]),
            "C_realized_kmol_d":eta*res["Csolid_kmol_d"]}

# ------------------------------------------------------------
# Generic thermodynamic helpers for multi-reaction kinetics
# ------------------------------------------------------------
def reaction_Kp(T, stoich):
    """
    Generic dimensionless Kp from standard Gibbs energies.
    stoich: dict e.g. {"CO":-1, "H2":-3, "CH4":+1, "H2O":+1}
    Use "C(s)" for graphite (activity=1).
    """
    g = ct.Solution("gri30.yaml")
    s = ct.Solution("graphite.yaml")
    g.TPX = T, ct.one_atm, "H2:1.0"
    s.TP = T, ct.one_atm
    dG_RT = 0.0
    for sp, nu in stoich.items():
        if sp == "C(s)":
            mu_rt = s.standard_gibbs_RT[0]
        else:
            mu_rt = g.standard_gibbs_RT[g.species_index(sp)]
        dG_RT += nu * mu_rt
    return np.exp(-dG_RT)

def arrhenius_k(T, kref, Ea, Tref):
    """Generic Arrhenius scaling."""
    return kref * np.exp(-Ea / R_MOL * (1.0/T - 1.0/Tref))

# ------------------------------------------------------------
# CFR kinetic model (4-reaction network)
# ------------------------------------------------------------
def run_stage3_cfr_kinetic(feed, Tc, Pc, eta=1.0,
                           tau_s=3.0, n_steps=2000,
                           kref_co2_meth=None,
                           kref_co_carb=None, kref_ch4_carb=None):
    """
    Minimal kinetic CFR reactor with 3 independent reversible reactions:
    R1: CO2 + 4H2 <=> CH4 + 2H2O    (CO2 methanation)
    R2: CO  + H2  <=> C(s) + H2O    (CO carbon deposition)
    R3: CH4       <=> C(s) + 2H2    (CH4 cracking)
    Note: CO methanation (CO + 3H2 <=> CH4 + H2O) is linearly dependent
    on R2 + reverse(R3) and is therefore excluded to maintain an
    independent reaction basis (6 species, 3 elements -> 3 independent rxns).
    Returns reaction-by-reaction extents for diagnostics.
    """
    feed = clean_species_dict(feed)

    # Allow override of kref values for sweep
    _kref_co2_meth = kref_co2_meth if kref_co2_meth is not None else cfr_kref_co2_meth
    _kref_co_carb  = kref_co_carb  if kref_co_carb  is not None else cfr_kref_co_carb
    _kref_ch4_carb = kref_ch4_carb if kref_ch4_carb is not None else cfr_kref_ch4_carb

    reactions = [
        {"name": "CO2_methanation",
         "stoich": {"CO2":-1, "H2":-4, "CH4":+1, "H2O":+2},
         "kref": _kref_co2_meth, "Ea": cfr_Ea_co2_meth, "Tref": T_cfr},
        {"name": "CO_carbon",
         "stoich": {"CO":-1, "H2":-1, "C(s)":+1, "H2O":+1},
         "kref": _kref_co_carb, "Ea": cfr_Ea_co_carb, "Tref": T_cfr},
        {"name": "CH4_cracking",
         "stoich": {"CH4":-1, "C(s)":+1, "H2":+2},
         "kref": _kref_ch4_carb, "Ea": cfr_Ea_ch4_carb, "Tref": T_cfr},
    ]

    gas_species = sorted(set(feed.keys()) | {"H2","CO","CO2","CH4","H2O"})
    F = {sp: kmol_per_day_to_mol_per_s(feed.get(sp, 0.0)) for sp in gas_species}
    F_Csolid = 0.0

    Ftot_in = sum(F.values())
    if Ftot_in <= 0: raise ValueError("CFR feed has zero total flow")

    vdot_in = Ftot_in * R_MOL * Tc / Pc
    V_reactor = tau_s * vdot_in
    dV = V_reactor / n_steps

    Kp_map = {rxn["name"]: reaction_Kp(Tc, rxn["stoich"]) for rxn in reactions}
    kf_map = {rxn["name"]: arrhenius_k(Tc, rxn["kref"], rxn["Ea"], rxn["Tref"])
              for rxn in reactions}

    # Extent tracking [mol/s cumulative]
    extent = {rxn["name"]: 0.0 for rxn in reactions}

    def gas_activity(y, sp, P):
        return max(y.get(sp, 0.0) * (P / ct.one_atm), TRACE)

    for _ in range(n_steps):
        Ftot = sum(max(F.get(sp, 0.0), 0.0) for sp in gas_species)
        if Ftot <= 0: break

        y = {sp: max(F.get(sp, 0.0), 0.0) / Ftot for sp in gas_species}

        dF = {sp: 0.0 for sp in gas_species}
        dF_C = 0.0

        for rxn in reactions:
            sto = rxn["stoich"]
            Kp = Kp_map[rxn["name"]]
            kf = kf_map[rxn["name"]]

            a_react = 1.0; a_prod = 1.0
            for sp, nu in sto.items():
                a = 1.0 if sp == "C(s)" else gas_activity(y, sp, Pc)
                if nu < 0: a_react *= a ** (-nu)
                elif nu > 0: a_prod *= a ** nu

            r = kf * (a_react - a_prod / Kp)

            # If no solid carbon exists yet, forbid reverse reactions that consume C(s)
            if sto.get("C(s)", 0) > 0 and F_Csolid <= TRACE and r < 0:
                r = 0.0

            for sp, nu in sto.items():
                if sp == "C(s)": dF_C += nu * r * dV
                else: dF[sp] += nu * r * dV

            extent[rxn["name"]] += r * dV

        # negativity limiter (skip C(s) check when no solid carbon exists)
        alpha = 1.0
        for sp in gas_species:
            if dF[sp] < 0:
                alpha = min(alpha, F[sp] / (-dF[sp]) if -dF[sp] > 0 else 1.0)
        if dF_C < 0 and F_Csolid > TRACE:
            alpha = min(alpha, F_Csolid / (-dF_C) if -dF_C > 0 else 1.0)
        alpha = max(0.0, min(alpha, 1.0))

        for sp in gas_species:
            F[sp] = max(F[sp] + alpha * dF[sp], 0.0)
        F_Csolid = max(F_Csolid + alpha * dF_C, 0.0)

    gas_out = clean_species_dict({
        sp: mol_per_s_to_kmol_per_day(F.get(sp, 0.0)) for sp in gas_species})
    Csolid_out = mol_per_s_to_kmol_per_day(F_Csolid)

    # Convert extents to kmol/day
    extent_kmol_d = {k: mol_per_s_to_kmol_per_day(v) for k, v in extent.items()}

    fH = total_stream_enthalpy_J_per_day(feed, 0, Tc, Pc)
    pH = total_stream_enthalpy_J_per_day(gas_out, Csolid_out, Tc, Pc)

    return {
        "feed": feed,
        "result": {"T_K":Tc, "P_Pa":Pc, "gas_kmol_d":gas_out,
                   "Csolid_kmol_d":Csolid_out, "converged":1, "error":None,
                   "tau_s":tau_s, "Kp_map":Kp_map, "kf_map":kf_map},
        "Q_kW": (pH - fH) / 86400.0 / 1000.0,
        "balance": element_balance_table(feed, gas_out, Csolid_out),
        "C_realized_kmol_d": eta * Csolid_out,
        "extent_kmol_d": extent_kmol_d,
    }

def run_cfr_carbon_window_sweep(cfr_feed, T_grid=None, co_carb_mults=None,
                                 tau_s=3.0, n_steps=2000):
    """
    Sweep CFR temperature and CO_carbon kref multiplier
    to find the carbon deposition window.
    """
    if T_grid is None:
        T_grid = np.array([550, 600, 650, 700, 750, 800, 850, 900])
    if co_carb_mults is None:
        co_carb_mults = np.array([1, 5, 10, 20, 50, 100])

    rows = []
    for Tc in T_grid:
        for mult in co_carb_mults:
            res = run_stage3_cfr_kinetic(
                dict(cfr_feed), Tc, P_cfr, eta=1.0,
                tau_s=tau_s, n_steps=n_steps,
                kref_co_carb=cfr_kref_co_carb * mult,
                kref_ch4_carb=cfr_kref_ch4_carb * mult,  # scale cracking too
            )
            ext = res.get("extent_kmol_d", {})
            c_tpd = kmol_per_day_to_tpd(res["result"]["Csolid_kmol_d"], MW_C_SOLID)
            ext_co_carb = ext.get("CO_carbon", 0)
            ext_ch4_crack = ext.get("CH4_cracking", 0)
            rows.append({
                "T_K": Tc,
                "co_carb_mult": mult,
                "C_solid_tpd": c_tpd,
                "ext_CO2_meth": ext.get("CO2_methanation", 0),
                "ext_CO_carb": ext_co_carb,
                "ext_CH4_crack": ext_ch4_crack,
                "net_C_extent": ext_co_carb + ext_ch4_crack,
                "Q_kW": res["Q_kW"],
            })
    return pd.DataFrame(rows)

def run_cfr_validation_series(s2_gas, T_grid, tau_s=3.0):
    """
    Validation-ready Stage 3 series:
    sweep T and record carbon deposition + gas composition.
    """
    rows = []
    sep = build_cfr_feed(s2_gas, water_remove_frac, h2_to_cfr_frac,
                         co2_to_cfr_frac, co_to_cfr_frac, ch4_to_cfr_frac)
    cfr_feed = sep["cfr_feed"]

    for Tc in T_grid:
        res = run_stage3_cfr_kinetic(cfr_feed, Tc, P_cfr,
            eta=eta_cfr_realization, tau_s=tau_s, n_steps=cfr_n_steps)
        gas = res["result"]["gas_kmol_d"]
        ext = res.get("extent_kmol_d", {})
        rows.append({
            "T_K": Tc, "tau_s": tau_s,
            "C_solid_tpd": kmol_per_day_to_tpd(res["result"]["Csolid_kmol_d"], MW_C_SOLID),
            "CH4_out_tpd": kmol_per_day_to_tpd(gas.get("CH4",0), MW["CH4"]),
            "H2_out_tpd": kmol_per_day_to_tpd(gas.get("H2",0), MW["H2"]),
            "CO_out_tpd": kmol_per_day_to_tpd(gas.get("CO",0), MW["CO"]),
            "CO2_out_tpd": kmol_per_day_to_tpd(gas.get("CO2",0), MW["CO2"]),
            "H2O_out_tpd": kmol_per_day_to_tpd(gas.get("H2O",0), MW["H2O"]),
            "ext_CO2_meth": ext.get("CO2_methanation",0),
            "ext_CO_carb": ext.get("CO_carbon",0),
            "ext_CH4_crack": ext.get("CH4_cracking",0),
            "Q3_kW": res["Q_kW"],
        })
    return pd.DataFrame(rows)

def run_baseline():
    s1=run_stage1_pyrolysis_kinetic(CH4_tpd, T_pyro, P_pyro,
        tau_s=pyro_tau_s, n_steps=pyro_n_steps,
        kref=pyro_kref, Ea=pyro_Ea, Tref=pyro_Tref)
    rf,h2s=build_rwgs_feed(s1,CO2_total_tpd,solar_power_kW,electrolyzer_eff)
    s2=run_stage2_rwgs_kinetic(rf, T_rwgs, P_rwgs,
        tau_s=rwgs_tau_s, n_steps=rwgs_n_steps,
        kref=rwgs_kref, Ea=rwgs_Ea, Tref=rwgs_Tref)
    sep=build_cfr_feed(s2["result"]["gas_kmol_d"],water_remove_frac,h2_to_cfr_frac,co2_to_cfr_frac,co_to_cfr_frac,ch4_to_cfr_frac)
    s3=run_stage3_cfr_kinetic(sep["cfr_feed"], T_cfr, P_cfr,
        eta=eta_cfr_realization, tau_s=cfr_tau_s, n_steps=cfr_n_steps)
    summary=pd.DataFrame({"Quantity":[
        "CH4 feed","CO2 feed","Solar H2",
        "Pyro X_CH4","Pyro tau_s",
        "RWGS X_CO2","RWGS tau_s",
        "CFR tau_s",
        "Stage1 C","Stage3 C","Stage3 C(eng)",
        "Water","Q1","Q2","Q3"],
        "Value":[CH4_tpd,CO2_total_tpd,kmol_per_day_to_tpd(h2s,MW["H2"]),
                 s1["X_CH4"], pyro_tau_s,
                 s2["X_CO2"], rwgs_tau_s,
                 cfr_tau_s,
                 kmol_per_day_to_tpd(s1["result"]["Csolid_kmol_d"],MW_C_SOLID),
                 kmol_per_day_to_tpd(s3["result"]["Csolid_kmol_d"],MW_C_SOLID),
                 kmol_per_day_to_tpd(s3["C_realized_kmol_d"],MW_C_SOLID),
                 kmol_per_day_to_tpd(sep["recovered_water"].get("H2O",0),MW["H2O"]),
                 s1["Q_kW"],s2["Q_kW"],s3["Q_kW"]],
        "Unit":["t/d","t/d","t/d","-","s","-","s","s",
                "t/d","t/d","t/d","t/d","kW","kW","kW"]})
    bal={"stage1":s1["balance"],"stage2":s2["balance"],"stage3":s3["balance"]}
    st={"s1_gas":dataframe_from_species(s1["result"]["gas_kmol_d"]),"rwgs_out":dataframe_from_species(s2["result"]["gas_kmol_d"]),
        "water":dataframe_from_species(sep["recovered_water"]),"cfr_feed":dataframe_from_species(sep["cfr_feed"]),
        "cfr_out":dataframe_from_species(s3["result"]["gas_kmol_d"])}
    return summary,bal,st,{"s1":s1,"s2":s2,"s3":s3,"sep":sep,"h2s":h2s}

def run_cfr_sweep(s2g):
    recs=[]; To=ordered_temperature_grid(T_cfr_grid,T_cfr)
    for wc in water_grid:
        for hf in h2frac_grid:
            sep=build_cfr_feed(s2g,wc,hf,co2_to_cfr_frac,co_to_cfr_frac,ch4_to_cfr_frac)
            cf=sep["cfr_feed"]; wt=kmol_per_day_to_tpd(sep["recovered_water"].get("H2O",0),MW["H2O"])
            tr={}
            for Tc in To:
                r=run_stage3_cfr_kinetic(cf, Tc, P_cfr, eta=1.0,
                    tau_s=cfr_tau_s, n_steps=cfr_n_steps)
                c3=kmol_per_day_to_tpd(r["result"]["Csolid_kmol_d"],MW_C_SOLID)
                tr[Tc]=[wc,hf,Tc,c3,wt,r["Q_kW"],1]
            for Tc in T_cfr_grid: recs.append(tr[Tc])
    return pd.DataFrame(recs, columns=["water_remove_frac","h2_to_cfr_frac","T_cfr_K","C_stage3_tpd","Recovered_water_tpd","Q3_kW","converged"])

if __name__ == "__main__":
    summary,bal,st,raw = run_baseline()
    print("\n===== SUMMARY ====="); print(summary.to_string(index=False))
    for k,v in st.items(): print(f"\n===== {k} ====="); print(v.to_string(index=False))
    for k,v in bal.items(): print(f"\n===== BALANCE: {k} ====="); print(v.to_string(index=False))

    # CFR reaction extents (diagnostic)
    if "extent_kmol_d" in raw["s3"]:
        print("\n===== CFR REACTION EXTENTS (kmol/day) =====")
        for rxn, ext in raw["s3"]["extent_kmol_d"].items():
            print(f"  {rxn:20s}: {ext:+12.4f}")

    # CFR parametric sweep
    sweep = run_cfr_sweep(raw["s2"]["result"]["gas_kmol_d"])
    summary.to_csv("workflow_summary.csv",index=False); sweep.to_csv("workflow_sweep.csv",index=False)
    for k,v in bal.items(): v.to_csv(f"balance_{k}.csv",index=False)

    # CFR heatmap
    piv = sweep.groupby(["water_remove_frac","T_cfr_K"],as_index=False)["C_stage3_tpd"].max().pivot(index="water_remove_frac",columns="T_cfr_K",values="C_stage3_tpd")
    plt.figure(figsize=(10,5)); plt.imshow(piv.values,aspect="auto",origin="lower"); plt.colorbar(label="Stage 3 C [t/d]")
    plt.xticks(range(len(piv.columns)),[f"{int(x)}" for x in piv.columns],rotation=45)
    plt.yticks(range(len(piv.index)),[f"{x:.2f}" for x in piv.index])
    plt.xlabel("T_cfr [K]"); plt.ylabel("Water removal fraction"); plt.title("Max CFR solid carbon (kinetic)"); plt.tight_layout()
    plt.savefig("heatmap_C_stage3.png",dpi=160); plt.close()

    # CFR carbon deposition window sweep
    print("\n===== CFR CARBON WINDOW SWEEP =====")
    cfr_window = run_cfr_carbon_window_sweep(
        raw["sep"]["cfr_feed"], tau_s=cfr_tau_s, n_steps=cfr_n_steps)
    cfr_window.to_csv("cfr_carbon_window.csv", index=False)
    print(cfr_window[["T_K","co_carb_mult","C_solid_tpd",
                       "ext_CO2_meth","ext_CO_carb","ext_CH4_crack","net_C_extent"]].to_string(index=False))

    # Carbon window heatmap
    cwpiv = cfr_window.pivot(index="co_carb_mult", columns="T_K", values="C_solid_tpd")
    plt.figure(figsize=(10,5))
    plt.imshow(cwpiv.values, aspect="auto", origin="lower", cmap="YlOrRd")
    plt.colorbar(label="Stage 3 solid carbon [t/day]")
    plt.xticks(range(len(cwpiv.columns)), [f"{int(x)}" for x in cwpiv.columns])
    plt.yticks(range(len(cwpiv.index)), [f"{x}" for x in cwpiv.index])
    plt.xlabel("T_CFR [K]"); plt.ylabel("CO_carbon kref multiplier")
    plt.title("CFR: carbon deposition window (T vs catalyst selectivity)"); plt.tight_layout()
    plt.savefig("heatmap_cfr_carbon_window.png", dpi=160); plt.close()

    # RWGS kinetic tau x T sweep
    print("\n===== RWGS TAU SWEEP =====")
    rf = raw["s2"]["feed"]
    rwgs_sweep = run_rwgs_tau_sweep(rf, rwgs_T_grid, rwgs_tau_grid, P_rwgs,
                                    kref=rwgs_kref, Ea=rwgs_Ea, Tref=rwgs_Tref)
    rwgs_sweep.to_csv("rwgs_tau_sweep.csv", index=False)
    print(rwgs_sweep[["T_K","tau_s","X_CO2_kinetic","X_CO2_equilibrium","approach_to_eq"]].to_string(index=False))

    # RWGS heatmap: approach to equilibrium
    rpiv = rwgs_sweep.pivot(index="tau_s", columns="T_K", values="approach_to_eq")
    plt.figure(figsize=(10,5))
    plt.imshow(rpiv.values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="Approach to equilibrium (X_kin / X_eq)")
    plt.xticks(range(len(rpiv.columns)), [f"{int(x)}" for x in rpiv.columns])
    plt.yticks(range(len(rpiv.index)), [f"{x:.1f}" for x in rpiv.index])
    plt.xlabel("T_RWGS [K]"); plt.ylabel("Residence time [s]")
    plt.title("RWGS: approach to equilibrium vs tau and T"); plt.tight_layout()
    plt.savefig("heatmap_rwgs_approach.png", dpi=160); plt.close()

    # Pyrolysis kinetic tau x T sweep
    print("\n===== PYROLYSIS TAU SWEEP =====")
    pyro_sweep = run_pyro_tau_sweep(CH4_tpd, pyro_T_grid, pyro_tau_grid, P_pyro,
                                    kref=pyro_kref, Ea=pyro_Ea, Tref=pyro_Tref)
    pyro_sweep.to_csv("pyro_tau_sweep.csv", index=False)
    print(pyro_sweep[["T_K","tau_s","X_CH4_kinetic","X_CH4_equilibrium","approach_to_eq","C_solid_tpd"]].to_string(index=False))

    # Pyrolysis heatmap: approach to equilibrium
    ppiv = pyro_sweep.pivot(index="tau_s", columns="T_K", values="approach_to_eq")
    plt.figure(figsize=(10,5))
    plt.imshow(ppiv.values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="magma")
    plt.colorbar(label="Approach to equilibrium (X_kin / X_eq)")
    plt.xticks(range(len(ppiv.columns)), [f"{int(x)}" for x in ppiv.columns])
    plt.yticks(range(len(ppiv.index)), [f"{x:.1f}" for x in ppiv.index])
    plt.xlabel("T_pyrolysis [K]"); plt.ylabel("Residence time [s]")
    plt.title("Pyrolysis: approach to equilibrium vs tau and T"); plt.tight_layout()
    plt.savefig("heatmap_pyro_approach.png", dpi=160); plt.close()

    # Pyrolysis heatmap: solid carbon [t/day]
    cpiv = pyro_sweep.pivot(index="tau_s", columns="T_K", values="C_solid_tpd")
    plt.figure(figsize=(10,5))
    plt.imshow(cpiv.values, aspect="auto", origin="lower", cmap="inferno")
    plt.colorbar(label="Solid carbon [t/day]")
    plt.xticks(range(len(cpiv.columns)), [f"{int(x)}" for x in cpiv.columns])
    plt.yticks(range(len(cpiv.index)), [f"{x:.1f}" for x in cpiv.index])
    plt.xlabel("T_pyrolysis [K]"); plt.ylabel("Residence time [s]")
    plt.title("Pyrolysis: solid carbon output vs tau and T"); plt.tight_layout()
    plt.savefig("heatmap_pyro_carbon.png", dpi=160); plt.close()

    # ================================================================
    # Validation-ready reduced datasets
    # ================================================================

    # Stage 1: 1200 K, CH4 conversion vs tau
    pyro_val = pyro_sweep[np.isclose(pyro_sweep["T_K"], pyro_validation_T)].copy()
    pyro_val = pyro_val[["T_K","tau_s","X_CH4_kinetic","X_CH4_equilibrium","approach_to_eq","C_solid_tpd","Q_kW"]]
    pyro_val.to_csv("validation_stage1_pyro_1200K.csv", index=False)
    print("\n===== VALIDATION: STAGE 1 (1200 K) =====")
    print(pyro_val.to_string(index=False))

    plt.figure(figsize=(8,5))
    plt.plot(pyro_val["tau_s"], pyro_val["X_CH4_kinetic"], marker="o", label="Kinetic")
    plt.axhline(pyro_val["X_CH4_equilibrium"].iloc[0], ls="--", label="Equilibrium")
    plt.xlabel("Residence time [s]"); plt.ylabel("CH4 conversion [-]")
    plt.title("Stage 1 validation: CH4 conversion at 1200 K"); plt.legend(); plt.tight_layout()
    plt.savefig("validation_stage1_pyro_1200K.png", dpi=160); plt.close()

    # Stage 2: 950 K, CO2 conversion vs tau
    rwgs_val = rwgs_sweep[np.isclose(rwgs_sweep["T_K"], rwgs_validation_T)].copy()
    rwgs_val = rwgs_val[["T_K","tau_s","X_CO2_kinetic","X_CO2_equilibrium","approach_to_eq","Q_kW"]]
    rwgs_val.to_csv("validation_stage2_rwgs_950K.csv", index=False)
    print("\n===== VALIDATION: STAGE 2 (950 K) =====")
    print(rwgs_val.to_string(index=False))

    plt.figure(figsize=(8,5))
    plt.plot(rwgs_val["tau_s"], rwgs_val["X_CO2_kinetic"], marker="o", label="Kinetic")
    plt.axhline(rwgs_val["X_CO2_equilibrium"].iloc[0], ls="--", label="Equilibrium")
    plt.xlabel("Residence time [s]"); plt.ylabel("CO2 conversion [-]")
    plt.title("Stage 2 validation: CO2 conversion at 950 K"); plt.legend(); plt.tight_layout()
    plt.savefig("validation_stage2_rwgs_950K.png", dpi=160); plt.close()

    # Stage 3: 750-850 K, carbon deposition + CH4 vs temperature
    cfr_val = run_cfr_validation_series(
        raw["s2"]["result"]["gas_kmol_d"], cfr_validation_T_grid, tau_s=cfr_validation_tau_s)
    cfr_val.to_csv("validation_stage3_cfr_750_850K.csv", index=False)
    print("\n===== VALIDATION: STAGE 3 (750-850 K) =====")
    print(cfr_val.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    ax1.plot(cfr_val["T_K"], cfr_val["C_solid_tpd"], marker="o", color="black", label="Solid carbon")
    ax1.set_xlabel("Temperature [K]"); ax1.set_ylabel("Solid carbon [t/day]")
    ax1.set_title("Stage 3: carbon deposition"); ax1.legend()
    ax2.plot(cfr_val["T_K"], cfr_val["CH4_out_tpd"], marker="s", color="tab:orange", label="CH4 out")
    ax2.plot(cfr_val["T_K"], cfr_val["H2_out_tpd"], marker="^", color="tab:blue", label="H2 out")
    ax2.set_xlabel("Temperature [K]"); ax2.set_ylabel("[t/day]")
    ax2.set_title("Stage 3: gas composition"); ax2.legend()
    plt.tight_layout()
    plt.savefig("validation_stage3_cfr_750_850K.png", dpi=160); plt.close()

    print("\nDone. Files saved.")
