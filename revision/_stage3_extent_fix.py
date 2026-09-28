"""
_stage3_extent_fix.py -- corrected reaction-extent bookkeeping for the submitted Stage 3 reduced kinetic model.

In submitted_v1/Workflow_cantera.run_stage3_cfr_kinetic the negativity limiter scales the flow update of every
step by a factor alpha, but the recorded extents were accumulated with the UNscaled rates. The outlet flows are
correct; only the diagnostic extents were wrong whenever alpha < 1 (they no longer reconstruct the outlet). This
module builds `run_stage3_cfr_kinetic_fixed` from the submitted source text with the extents accumulated as
alpha * r * dV, adds a reconstruction check (outlet from extents == integrated outlet), and leaves the baseline
file untouched. Scripts that use extents import it from here.
"""
import inspect, textwrap
import numpy as np
from _paths import HERE, REPO, BASE, RESULT, SOURCES, NOTES
import Workflow_cantera as W

_src = textwrap.dedent(inspect.getsource(W.run_stage3_cfr_kinetic))
assert 'extent[rxn["name"]] += r * dV' in _src and "alpha = max(0.0, min(alpha, 1.0))" in _src, "unexpected baseline source"
_src = _src.replace("def run_stage3_cfr_kinetic(", "def run_stage3_cfr_kinetic_fixed(", 1)
_src = _src.replace("        dF_C = 0.0\n", "        dF_C = 0.0\n        step_ext = {}\n", 1)
_src = _src.replace('            extent[rxn["name"]] += r * dV', '            step_ext[rxn["name"]] = r * dV', 1)
_src = _src.replace("        alpha = max(0.0, min(alpha, 1.0))\n", "        alpha = max(0.0, min(alpha, 1.0))\n        for _k, _v in step_ext.items(): extent[_k] += alpha * _v\n", 1)
_ns = dict(W.__dict__); exec(compile(_src, "<stage3_extent_fix>", "exec"), _ns)
run_stage3_cfr_kinetic_fixed = _ns["run_stage3_cfr_kinetic_fixed"]

STOICH = {"CO2_methanation": {"CO2": -1, "H2": -4, "CH4": 1, "H2O": 2}, "CO_carbon": {"CO": -1, "H2": -1, "C(s)": 1, "H2O": 1}, "CH4_cracking": {"CH4": -1, "C(s)": 1, "H2": 2}}

def reconstruct(feed, ext):
    """Outlet (gas dict, solid) implied by the feed and the reaction extents [kmol/d]."""
    out = dict(feed); solid = 0.0
    for rxn, e in ext.items():
        for sp, nu in STOICH[rxn].items():
            if sp == "C(s)": solid += nu * e
            else: out[sp] = out.get(sp, 0.0) + nu * e
    return out, solid

def check(feed, res, tol=1e-6):
    """Max relative mismatch between the outlet reconstructed from extents and the integrated outlet."""
    g, C = res["result"]["gas_kmol_d"], res["result"]["Csolid_kmol_d"]; rg, rC = reconstruct(feed, res["extent_kmol_d"])
    scale = max(sum(feed.values()), 1e-30)
    err = max([abs(rC - C) / scale] + [abs(rg.get(sp, 0.0) - g.get(sp, 0.0)) / scale for sp in set(rg) | set(g)])
    return err, err <= tol

if __name__ == "__main__":
    import K2_validation_design as K2m   # design-case Stage 3 inlet (tau1* = 8.905 s, phi = 1, h = 0.35)
    feed = K2m.FEED3
    for T in (650.0, 750.0, 800.0, 850.0):
        r_old = W.run_stage3_cfr_kinetic(dict(feed), T, W.P_cfr, eta=1.0, tau_s=3.0, n_steps=W.cfr_n_steps)
        r_new = run_stage3_cfr_kinetic_fixed(dict(feed), T, W.P_cfr, eta=1.0, tau_s=3.0, n_steps=W.cfr_n_steps)
        eo, _ = check(feed, r_old); en, ok = check(feed, r_new)
        C = r_new["result"]["Csolid_kmol_d"]; eold = r_old["extent_kmol_d"]; enew = r_new["extent_kmol_d"]
        print(f"T3 = {T:.0f} K: outlet solid {C:8.2f} kmol/d | from extents: submitted {eold['CO_carbon']+eold['CH4_cracking']:8.2f}, fixed {enew['CO_carbon']+enew['CH4_cracking']:8.2f}"
              f" | max rel mismatch: submitted {eo:.2e}, fixed {en:.2e} {'OK' if ok else 'FAIL'} | outlet flows identical: {abs(r_new['result']['Csolid_kmol_d']-r_old['result']['Csolid_kmol_d'])<1e-9}")
