"""V11 - exactness of the restricted additive-in-angle family (V1a / V2a).

Claim under test (V1a). A circuit in the additive-in-angle family - one base
RY on each target followed by one controlled RY per incoming edge, all acting
on the same target - prepares a computational-basis distribution equal to the
Bayesian-network factorization of the conditional probability table that the
angles induce,

    P(x_i = 1 | pa_i) = sin^2( theta_eff(pa_i) / 2 ),

where theta_eff is the base angle plus every increment whose control
condition is met under pa_i.

Claim under test (V2a). Removing the mechanism block of an intervened
register - the rotations whose target is that register, with their control
relabeling X gates - and preparing the register in the intervention value
yields the truncated factorization of the induced table. Gates of later
registers that use the intervened register as a control are kept, including
the X gates that relabel such a control.

Both claims are checked on the statevector, so there is no shot noise. They
are checked on the two models used in the manuscript and on randomly drawn
angle specifications over several graph shapes. Random angles are drawn so
that composed totals fall both inside and outside [0, pi]. Exactness is
expected in both regimes, because sin^2(theta/2) is the Born probability for
any theta; only the monotone reading of an increment is lost beyond pi.

Expected outcome: agreement at floating-point rounding level for every draw,
every intervened node and both intervention arms.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit

import common as C

SEED = 42
N_RANDOM = 250          # draws per graph shape
ANGLE_HI = 1.5          # increments and bases drawn from U(0, ANGLE_HI)


# ---------------------------------------------------------------- builder

def theta_eff(spec, pa):
    """Composed angle for one parent assignment.

    spec: {"base": theta0, "inc": [(parent, cond, theta), ...]}
    pa:   {parent_name: value}
    """
    total = spec["base"]
    for parent, cond, theta in spec["inc"]:
        if pa[parent] == cond:
            total += theta
    return total


def induced_cpt(nodes, parents, angles):
    """Conditional probability table induced by an additive specification."""
    cpt = {}
    for n in nodes:
        table = {}
        for values in itertools.product((0, 1), repeat=len(parents[n])):
            pa = dict(zip(parents[n], values))
            table[values] = C.p_from_theta(theta_eff(angles[n], pa))
        cpt[n] = table
    return cpt


def build_additive(nodes, parents, angles, do=None):
    """Additive-in-angle circuit. Nodes must be listed parents-first.

    do: None or (node_name, value). Surgery skips the intervened register's
    own mechanism block (base rotation and incoming controlled rotations)
    and prepares the register in the intervention value. Later registers
    are built as usual, so gates that use the intervened register as a
    control - and the X gates that relabel such a control - are kept.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    qc = QuantumCircuit(len(nodes))
    for n in nodes:
        t = idx[n]
        if do is not None and n == do[0]:
            if do[1] == 1:
                qc.x(t)
            continue
        spec = angles[n]
        qc.ry(spec["base"], t)
        for parent, cond, theta in spec["inc"]:
            c = idx[parent]
            if cond == 0:
                qc.x(c)
            qc.cry(theta, c, t)
            if cond == 0:
                qc.x(c)
    return qc


def max_error(nodes, parents, angles, do=None):
    """Largest |amplitude^2 - reference probability| over all basis states."""
    model = {"nodes": nodes, "parents": parents,
             "cpt": induced_cpt(nodes, parents, angles)}
    probs = C.basis_probabilities(build_additive(nodes, parents, angles, do))
    n = len(nodes)
    worst = 0.0
    for k in range(2 ** n):
        ref = C.factorization(model, C.assignment_of(k, n), do=do)
        worst = max(worst, abs(probs[k] - ref))
    return worst


def max_total_angle(nodes, parents, angles):
    """Largest composed angle over every node and parent assignment."""
    big = 0.0
    for n in nodes:
        for values in itertools.product((0, 1), repeat=len(parents[n])):
            pa = dict(zip(parents[n], values))
            big = max(big, theta_eff(angles[n], pa))
    return big


def check_model(nodes, parents, angles):
    """V1a and V2a (every non-root node, both arms) for one specification."""
    obs = max_error(nodes, parents, angles)
    surg = 0.0
    for n in nodes:
        if not parents[n]:
            continue
        for value in (0, 1):
            surg = max(surg, max_error(nodes, parents, angles, do=(n, value)))
    return obs, surg


# ---------------------------------------------------------- paper models

NODES_3 = ["G", "T", "O"]
PARENTS_3 = {"G": (), "T": ("G",), "O": ("G", "T")}
ANGLES_3 = {
    "G": {"base": math.pi / 2, "inc": []},
    "T": {"base": 0.0, "inc": [("G", 0, 2.4), ("G", 1, 0.8)]},
    "O": {"base": 0.3, "inc": [("G", 1, 1.0), ("T", 1, 0.6)]},
}

NODES_10 = ["A", "I", "R", "G", "T", "S", "H", "D", "O", "F"]
PARENTS_10 = {
    "A": (), "I": ("A",), "R": ("I",), "G": ("R",),
    "T": ("A", "I", "G"), "S": ("T", "A"), "H": ("S", "R"),
    "D": ("H", "G"), "O": ("A", "R", "T", "D", "H"), "F": ("O", "D"),
}
ANGLES_10 = {
    "A": {"base": 1.0, "inc": []},
    "I": {"base": 0.0, "inc": [("A", 1, 0.8)]},
    "R": {"base": 0.0, "inc": [("I", 1, 1.2)]},
    "G": {"base": 0.0, "inc": [("R", 0, 1.0)]},
    "T": {"base": 0.2, "inc": [("A", 1, 1.2), ("I", 1, 1.0), ("G", 0, 1.4)]},
    "S": {"base": 0.3, "inc": [("T", 1, 0.8), ("A", 1, 0.4)]},
    "H": {"base": 0.4, "inc": [("S", 1, 0.6), ("R", 1, 0.5)]},
    "D": {"base": 0.5, "inc": [("H", 1, 0.4), ("G", 1, 0.3)]},
    "O": {"base": 0.1, "inc": [("A", 0, 0.8), ("R", 0, 0.6), ("T", 1, 1.2),
                               ("D", 1, 0.5), ("H", 1, 0.4)]},
    "F": {"base": 0.3, "inc": [("O", 1, 0.8), ("D", 1, 0.3)]},
}

# --------------------------------------------------------- random shapes

SHAPES = {
    "confounder_3": (["Z", "T", "Y"],
                     {"Z": (), "T": ("Z",), "Y": ("Z", "T")}),
    "collider_2":   (["A", "B", "C"],
                     {"A": (), "B": (), "C": ("A", "B")}),
    "three_parent": (["A", "B", "C", "D"],
                     {"A": (), "B": (), "C": ("A",), "D": ("A", "B", "C")}),
    "four_parent":  (["A", "B", "C", "D", "E"],
                     {"A": (), "B": (), "C": (), "D": ("A",),
                      "E": ("A", "B", "C", "D")}),
}


def random_angles(nodes, parents, rng):
    """One random additive specification: base plus one increment per edge."""
    angles = {}
    for n in nodes:
        inc = [(p, int(rng.integers(0, 2)), float(rng.uniform(0.0, ANGLE_HI)))
               for p in parents[n]]
        angles[n] = {"base": float(rng.uniform(0.0, ANGLE_HI)), "inc": inc}
    return angles


# ------------------------------------------------------------------ main

def main():
    C.banner("V11 - exactness of the additive-in-angle family (V1a / V2a)")
    rng = np.random.default_rng(SEED)
    payload = {"seed": SEED, "n_random_per_shape": N_RANDOM,
               "angle_hi": ANGLE_HI, "paper_models": {}, "random": {}}

    # self-consistency: the general builder must reproduce the hand-written
    # circuits of common.py exactly (H on |0> equals RY(pi/2) on |0>)
    ref3 = C.basis_probabilities(C.build_additive_3(measure=False))
    mine3 = C.basis_probabilities(build_additive(NODES_3, PARENTS_3, ANGLES_3))
    ref10 = C.basis_probabilities(C.build_additive_10(do=None, measure=False))
    mine10 = C.basis_probabilities(build_additive(NODES_10, PARENTS_10, ANGLES_10))
    d3, d10 = float(np.abs(ref3 - mine3).max()), float(np.abs(ref10 - mine10).max())
    print("builder vs common.build_additive_3  : max |dp| = {0:.3e}".format(d3))
    print("builder vs common.build_additive_10 : max |dp| = {0:.3e}".format(d10))
    payload["builder_consistency"] = {"model_3": d3, "model_10": d10}

    print("\n--- manuscript models ---")
    for name, nodes, parents, angles in (
            ("three_variable", NODES_3, PARENTS_3, ANGLES_3),
            ("ten_variable", NODES_10, PARENTS_10, ANGLES_10)):
        obs, surg = check_model(nodes, parents, angles)
        big = max_total_angle(nodes, parents, angles)
        print("{0:15s} V1a {1:.3e}  V2a {2:.3e}  max composed angle {3:.2f}"
              " ({4})".format(name, obs, surg, big,
                              "exceeds pi" if big > math.pi else "within pi"))
        payload["paper_models"][name] = {
            "v1a_max_error": obs, "v2a_max_error": surg,
            "max_composed_angle": big, "exceeds_pi": big > math.pi}

    print("\n--- random specifications ---")
    for shape, (nodes, parents) in SHAPES.items():
        within = {"n": 0, "v1a": 0.0, "v2a": 0.0}
        beyond = {"n": 0, "v1a": 0.0, "v2a": 0.0}
        for _ in range(N_RANDOM):
            angles = random_angles(nodes, parents, rng)
            obs, surg = check_model(nodes, parents, angles)
            bucket = beyond if max_total_angle(nodes, parents, angles) > math.pi else within
            bucket["n"] += 1
            bucket["v1a"] = max(bucket["v1a"], obs)
            bucket["v2a"] = max(bucket["v2a"], surg)
        print("{0:14s} within pi: n={1:3d} V1a {2:.3e} V2a {3:.3e} | "
              "beyond pi: n={4:3d} V1a {5:.3e} V2a {6:.3e}".format(
                  shape, within["n"], within["v1a"], within["v2a"],
                  beyond["n"], beyond["v1a"], beyond["v2a"]))
        payload["random"][shape] = {"within_pi": within, "beyond_pi": beyond}

    allv = [payload["paper_models"][k][m] for k in payload["paper_models"]
            for m in ("v1a_max_error", "v2a_max_error")]
    allv += [payload["random"][s][r][m] for s in payload["random"]
             for r in ("within_pi", "beyond_pi") for m in ("v1a", "v2a")]
    payload["overall_max_error"] = max(allv)
    print("\noverall worst |dp| over everything: {0:.3e}".format(max(allv)))
    C.save_result("v11_additive_exactness", payload)


if __name__ == "__main__":
    main()
