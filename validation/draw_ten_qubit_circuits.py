"""Draw the ten-qubit circuits of the manuscript (Appendix B, Figs. B1-B2).

The circuits are built from the same angle specification that
v11_additive_exactness.py verifies (ANGLES_10, NODES_10), so the drawings
cannot drift from the specification that the exactness check covers. The
drawing style matches the three-qubit figures produced by
analysis/draw_circuits_standalone.py: Qiskit mpl drawer, "iqx" style, one
named register per variable, and a barrier after each node's mechanism
block.

Differences from the three-qubit figures, chosen for print legibility of a
ten-wire circuit:

* the final computational-basis measurement is omitted (it adds ten columns
  and carries no information about the mechanism);
* each circuit is folded into two rows at a mechanism-block boundary.

Gate counts are printed for cross-checking against the manuscript text:
observational 7 RY + 19 CRY + 8 X (depth 18), do(T=1) 6 RY + 16 CRY + 7 X
(depth 14), where the 7 X of the interventional circuit are 6 relabeling
gates plus the single X preparing the Treatment register.

Requires matplotlib and pylatexenc in addition to the validation
requirements. Writes ../figures/ten_qubit_observational_circuit.pdf and
../figures/ten_qubit_do_T1_circuit.pdf.
"""

import os

import matplotlib

matplotlib.use("Agg")

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.passes import RemoveBarriers

import v11_additive_exactness as v11

FULL_NAMES = {
    "A": "Age",
    "I": "Income",
    "R": "Region",
    "G": "Gender_bias",
    "T": "Treatment",
    "S": "Insurance",
    "H": "Hospital",
    "D": "Doctor",
    "O": "Outcome",
    "F": "Satisfaction",
}

# Fold points chosen so that each row break falls on a barrier:
# observational breaks after the Insurance block, do(T=1) after Hospital.
FOLD_OBSERVATIONAL = 20
FOLD_DO_T1 = 19

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def build_named(nodes, angles, do=None):
    """Same construction as v11.build_additive, with named registers and
    a barrier after each mechanism block; measurement omitted."""
    regs = [QuantumRegister(1, "%s (%s)" % (FULL_NAMES[n], n)) for n in nodes]
    qc = QuantumCircuit(*regs)
    q = {n: regs[i][0] for i, n in enumerate(nodes)}
    for n in nodes:
        if do is not None and n == do[0]:
            if do[1] == 1:
                qc.x(q[n])
            qc.barrier()
            continue
        spec = angles[n]
        if spec["base"] != 0:
            qc.ry(spec["base"], q[n])
        for parent, cond, theta in spec["inc"]:
            if cond == 0:
                qc.x(q[parent])
            qc.cry(theta, q[parent], q[n])
            if cond == 0:
                qc.x(q[parent])
        qc.barrier()
    return qc


def gate_summary(qc):
    """Gate counts and depth with the drawing-only barriers removed."""
    bare = RemoveBarriers()(qc)
    ops = {k: int(n) for k, n in bare.count_ops().items()}
    return ops, bare.depth()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [
        ("ten_qubit_observational_circuit", None, FOLD_OBSERVATIONAL),
        ("ten_qubit_do_T1_circuit", ("T", 1), FOLD_DO_T1),
    ]
    for name, do, fold in jobs:
        qc = build_named(v11.NODES_10, v11.ANGLES_10, do)
        ops, depth = gate_summary(qc)
        fig = qc.draw("mpl", style="iqx", fold=fold)
        path = os.path.join(OUT_DIR, name + ".pdf")
        fig.savefig(path, bbox_inches="tight")
        print("%-32s gates=%s total=%d depth=%d -> %s"
              % (name, ops, sum(ops.values()), depth, os.path.normpath(path)))


if __name__ == "__main__":
    main()
