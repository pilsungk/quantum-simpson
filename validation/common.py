"""Shared helpers for the DQS validation scripts.

This module holds the restricted additive-in-angle circuits used in the
manuscript, the exact reference distribution they are compared against, and
the provenance machinery that every result file carries. The scripts that
import it are:

    v04_additive_family_fit.py   expressivity limit of the additive family
    v11_additive_exactness.py    exactness of compilation and of surgery

Every result record stores the SHA256 of the script that produced it and of
this module, so a stored number can be checked against the sources without a
version-control system.
"""

import hashlib
import inspect
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ------------------------------------------------- angle <-> probability

def theta_from_p(p):
    """Rotation angle encoding probability p on a |0>-initialised qubit."""
    return 2.0 * math.asin(math.sqrt(p))


def p_from_theta(theta):
    """Probability produced by rotation angle theta: sin^2(theta/2)."""
    return math.sin(theta / 2.0) ** 2


# ------------------------------------------------ reference distribution

def factorization(model, assignment, do=None):
    """Reference probability of one complete assignment.

    Returns the Bayesian-network factorization prod_i P(x_i | pa_i), or the
    truncated factorization when do is given.

    Args:
        model: model dict.
        assignment: list of 0/1 in node order.
        do: None or (node_name, value).
    """
    nodes, parents, cpt = model["nodes"], model["parents"], model["cpt"]
    idx = {n: i for i, n in enumerate(nodes)}
    pr = 1.0
    for n in nodes:
        v = assignment[idx[n]]
        if do is not None and n == do[0]:
            pr *= 1.0 if v == do[1] else 0.0
            continue
        key = tuple(assignment[idx[q]] for q in parents[n])
        p = cpt[n][key]
        pr *= p if v == 1 else 1.0 - p
    return pr


def basis_probabilities(circuit):
    """Exact computational-basis distribution of a measurement-free circuit."""
    return np.abs(np.asarray(Statevector(circuit).data)) ** 2


def assignment_of(index, n_nodes):
    """Node-order assignment corresponding to a basis index (little-endian)."""
    return [(index >> i) & 1 for i in range(n_nodes)]


# ------------------------------------------- additive-family circuits

def build_additive_3(measure=False):
    """Three-variable circuit in the restricted additive-in-angle family.

    This is the construction used in the Simpson-paradox case study:
    q0 = Z, q1 = T, q2 = Y, with the outcome receiving a base rotation
    plus one rotation per incoming edge on the same target.
    """
    qc = QuantumCircuit(3, 3) if measure else QuantumCircuit(3)
    qc.h(0)
    qc.x(0)
    qc.cry(2.4, 0, 1)
    qc.x(0)
    qc.cry(0.8, 0, 1)
    qc.ry(0.3, 2)
    qc.cry(1.0, 0, 2)
    qc.cry(0.6, 1, 2)
    if measure:
        for i in range(3):
            qc.measure(i, i)
    return qc


def build_additive_10(do=None, measure=True):
    """Ten-variable healthcare-style circuit in the additive-in-angle family.

    Args:
        do: None for observational, or 0/1 to intervene on Treatment (q4).
        measure: append measurements.
    """
    qc = QuantumCircuit(10, 10)
    qc.ry(1.0, 0)
    qc.cry(0.8, 0, 1)
    qc.cry(1.2, 1, 2)
    qc.x(2)
    qc.cry(1.0, 2, 3)
    qc.x(2)
    if do is None:
        qc.ry(0.2, 4)
        qc.cry(1.2, 0, 4)
        qc.cry(1.0, 1, 4)
        qc.x(3)
        qc.cry(1.4, 3, 4)
        qc.x(3)
    elif do == 1:
        qc.x(4)
    qc.ry(0.3, 5)
    qc.cry(0.8, 4, 5)
    qc.cry(0.4, 0, 5)
    qc.ry(0.4, 6)
    qc.cry(0.6, 5, 6)
    qc.cry(0.5, 2, 6)
    qc.ry(0.5, 7)
    qc.cry(0.4, 6, 7)
    qc.cry(0.3, 3, 7)
    qc.ry(0.1, 8)
    qc.x(0)
    qc.cry(0.8, 0, 8)
    qc.x(0)
    qc.x(2)
    qc.cry(0.6, 2, 8)
    qc.x(2)
    qc.cry(1.2, 4, 8)
    qc.cry(0.5, 7, 8)
    qc.cry(0.4, 6, 8)
    qc.ry(0.3, 9)
    qc.cry(0.8, 8, 9)
    qc.cry(0.3, 7, 9)
    if measure:
        for q in range(10):
            qc.measure(q, q)
    return qc


def additive_probs(beta):
    """Four conditional probabilities of a 2-parent edge-additive node.

    theta(a, b) = beta0 + a * betaA + b * betaB, and the probability is
    sin^2(theta / 2). Returns the values in the order (a,b) =
    (0,0), (0,1), (1,0), (1,1).
    """
    b0, bA, bB = beta
    return np.array([p_from_theta(b0 + a * bA + b * bB)
                     for a in (0, 1) for b in (0, 1)])


# ------------------------------------------------------- provenance

def environment():
    """Environment metadata recorded with every result file."""
    import scipy
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qiskit": qiskit.__version__,
    }


def file_digest(path):
    """SHA256 of a file, as a hex string."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _calling_script():
    """Path of the script that invoked save_result.

    Walks out of this module so that the entry-point file is identified
    rather than common.py itself.
    """
    here = os.path.abspath(__file__)
    for frame in inspect.stack():
        candidate = os.path.abspath(frame.filename)
        if candidate != here and candidate.endswith(".py"):
            return candidate
    return os.path.abspath(sys.argv[0])


def save_result(script_name, payload):
    """Write one experiment's results to results/<script_name>.json.

    The record carries its own provenance: the SHA256 of the script that
    produced it and of this module, which holds the compiler, the
    reference models and the statistics. A result can therefore be checked
    against the sources without a version-control system - recompute the
    two digests and compare. Note that common.py matters as much as the
    script: changing a CPT here alters a result without the script itself
    changing at all.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    caller = _calling_script()
    record = {
        "experiment": script_name,
        "script": {"name": os.path.basename(caller),
                   "sha256": file_digest(caller)},
        "shared_module": {"name": os.path.basename(__file__),
                          "sha256": file_digest(os.path.abspath(__file__))},
        "environment": environment(),
        "results": payload,
    }
    path = os.path.join(RESULTS_DIR, script_name + ".json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
    print("\nresults written to {0}".format(os.path.relpath(path)))
    return path


def banner(title):
    """Print a consistent section header."""
    print("=" * 72)
    print(title)
    print("=" * 72)
