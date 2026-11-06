"""V12 - exact reference values for the ten-variable results table.

The ten-variable results table of the manuscript reports five treatment
contrasts estimated by simulation (30 trials of 15,000 shots): the unadjusted
observational contrast, two partial stratifications, the backdoor-adjusted
contrast over the DoWhy-identified set {Age, Gender_bias, Income, Region},
and the post-surgery interventional contrast. This script computes each of
them exactly.

Primary computation (no Qiskit). The circuit angles induce a definite
conditional probability table (v11, induced_cpt). The observational joint
distribution over the ten binary variables is its Bayesian-network product,
enumerated over all 1,024 assignments; the interventional joint under
do(T = v) is the truncated factorization, enumerated the same way. The five
contrasts are then evaluated on these joints with ordinary marginalization
and the g-formula. This path shares no code with the circuit simulation.

Cross-check (Qiskit). The same five quantities are evaluated on the
statevector of the additive-family circuit built by v11, observational and
post-surgery. By Proposition 1 of the manuscript the two paths must agree to
rounding.

Comparison. Each exact value is compared with the stored simulation record
(data/results_10_variable_backdoor.json): is the exact value inside the
simulated 95% interval, and how far is the simulated mean from it.

Expected outcome: primary and cross-check agree to ~1e-16; the exact
backdoor-adjusted and post-surgery contrasts coincide to ~1e-16; all five
exact values lie inside their simulated intervals.
"""

import itertools
import json
import os

import numpy as np

import common as C
import v11_additive_exactness as v11

NODES, PARENTS, ANGLES = v11.NODES_10, v11.PARENTS_10, v11.ANGLES_10
TREATMENT, OUTCOME = "T", "O"
ADJUSTMENT_SET = ["A", "G", "I", "R"]          # DoWhy-identified set
SIM_RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "results_10_variable_backdoor.json")
ROW_KEYS = {   # row label in the simulation record -> key used here
    "Observational (unadjusted)": "unadjusted",
    "Stratified by Age (partial)": "stratified_age",
    "Stratified by Region (partial)": "stratified_region",
    "Backdoor adjustment (DoWhy set)": "backdoor",
    "Circuit surgery representation": "surgery",
}


# ----------------------------------------------- joints by enumeration

def joint_by_enumeration(cpt, do=None):
    """Dict {assignment tuple (in NODES order): probability}."""
    pos = {n: i for i, n in enumerate(NODES)}
    joint = {}
    for x in itertools.product((0, 1), repeat=len(NODES)):
        p = 1.0
        for n in NODES:
            if do is not None and n == do[0]:
                if x[pos[n]] != do[1]:
                    p = 0.0
                    break
                continue
            pa = tuple(x[pos[q]] for q in PARENTS[n])
            p1 = cpt[n][pa]
            p *= p1 if x[pos[n]] == 1 else 1.0 - p1
        joint[x] = p
    return joint


def marginal(joint, names):
    """Marginal over the named variables, as {values tuple: prob}."""
    pos = {n: i for i, n in enumerate(NODES)}
    out = {}
    for x, p in joint.items():
        key = tuple(x[pos[n]] for n in names)
        out[key] = out.get(key, 0.0) + p
    return out


def contrast_adjusted(joint, z_names):
    """g-formula: sum_z P(z) [P(O=1|T=1,z) - P(O=1|T=0,z)] on the observational joint."""
    m = marginal(joint, z_names + [TREATMENT, OUTCOME])
    total = 0.0
    for z in itertools.product((0, 1), repeat=len(z_names)):
        pz = sum(m.get(z + (t, o), 0.0) for t in (0, 1) for o in (0, 1))
        if pz == 0.0:
            continue
        n1 = m.get(z + (1, 0), 0.0) + m.get(z + (1, 1), 0.0)
        n0 = m.get(z + (0, 0), 0.0) + m.get(z + (0, 1), 0.0)
        if n1 == 0.0 or n0 == 0.0:
            raise ValueError("stratum %r has no support for one treatment arm" % (z,))
        total += pz * (m.get(z + (1, 1), 0.0) / n1 - m.get(z + (0, 1), 0.0) / n0)
    return total


def five_contrasts(obs, do1, do0):
    """The five table rows from an observational joint and two interventional joints."""
    p1 = marginal(do1, [OUTCOME]).get((1,), 0.0)
    p0 = marginal(do0, [OUTCOME]).get((1,), 0.0)
    return {
        "unadjusted": contrast_adjusted(obs, []),
        "stratified_age": contrast_adjusted(obs, ["A"]),
        "stratified_region": contrast_adjusted(obs, ["R"]),
        "backdoor": contrast_adjusted(obs, ADJUSTMENT_SET),
        "surgery": p1 - p0,
        "p_o1_do_t1": p1,
        "p_o1_do_t0": p0,
    }


# ------------------------------------------------- cross-check (Qiskit)

def joint_from_statevector(do=None):
    probs = C.basis_probabilities(v11.build_additive(NODES, PARENTS, ANGLES, do))
    n = len(NODES)
    return {tuple(C.assignment_of(k, n)): float(probs[k]) for k in range(2 ** n)}


# ------------------------------------------------------------------ main

def main():
    C.banner("V12 - exact reference values for the ten-variable table")
    cpt = v11.induced_cpt(NODES, PARENTS, ANGLES)

    # primary: enumeration of the induced table
    obs = joint_by_enumeration(cpt)
    do1 = joint_by_enumeration(cpt, (TREATMENT, 1))
    do0 = joint_by_enumeration(cpt, (TREATMENT, 0))
    exact = five_contrasts(obs, do1, do0)
    assert abs(sum(obs.values()) - 1.0) < 1e-12

    # cross-check: Qiskit statevector of the actual circuits
    qk = five_contrasts(joint_from_statevector(),
                        joint_from_statevector((TREATMENT, 1)),
                        joint_from_statevector((TREATMENT, 0)))
    xcheck = max(abs(exact[k] - qk[k]) for k in exact)

    # comparison with the stored simulation record
    sim = json.load(open(SIM_RECORD))["methods"]
    comparison = {}
    for label, key in ROW_KEYS.items():
        s = sim[label]
        lo, hi = s["ci_95"]
        comparison[key] = {
            "simulated_mean": s["ate_mean"], "simulated_ci_95": [lo, hi],
            "exact": exact[key], "mean_minus_exact": s["ate_mean"] - exact[key],
            "exact_inside_ci": bool(lo <= exact[key] <= hi)}

    print("{0:20s} {1:>11s} {2:>11s} {3:>22s} {4:>12s}  inside".format(
        "row", "exact", "sim mean", "sim 95% CI", "mean-exact"))
    for key, c in comparison.items():
        print("{0:20s} {1:+11.7f} {2:+11.6f} [{3:.6f}, {4:.6f}] {5:+12.6f}  {6}".format(
            key, c["exact"], c["simulated_mean"], *c["simulated_ci_95"],
            c["mean_minus_exact"], "yes" if c["exact_inside_ci"] else "NO"))
    print("\nP(O=1|do(T=1)) = {0:.7f}   P(O=1|do(T=0)) = {1:.7f}".format(
        exact["p_o1_do_t1"], exact["p_o1_do_t0"]))
    print("exact backdoor - exact surgery = {0:.2e}".format(exact["backdoor"] - exact["surgery"]))
    print("enumeration vs Qiskit statevector, max |diff| = {0:.2e}".format(xcheck))

    C.save_result("v12_exact_ten_variable", {
        "treatment": TREATMENT, "outcome": OUTCOME, "adjustment_set": ADJUSTMENT_SET,
        "exact": exact,
        "backdoor_minus_surgery": exact["backdoor"] - exact["surgery"],
        "enumeration_vs_statevector_max_abs_diff": xcheck,
        "comparison_with_simulation": comparison,
        "simulation_record": os.path.basename(SIM_RECORD)})


if __name__ == "__main__":
    main()
