# Validation scripts for the additive-in-angle circuit family

Reproduction artifact for the exactness and expressivity claims made about
the restricted additive-in-angle circuit family used in the manuscript. Each
script runs on its own, prints a table, and writes a JSON record to
`results/`.

## What is being validated

The exactness statements are mathematical; the scripts do not test whether
they are true. They test that the circuits as built realize them, at
floating-point precision, on the manuscript's two models and on randomly
drawn specifications. The expressivity script measures a quantity the theory
states only qualitatively: how far the restricted family falls short of an
arbitrary conditional probability table.

Every measurement is made on an ideal statevector simulator. Nothing here
concerns hardware noise, and no experiment claims a quantum advantage.

## Scripts

| script | claim under test | headline result |
|---|---|---|
| `v11_additive_exactness.py` | V1a: the circuit prepares the Bayesian-network factorization of the conditional probability table its angles induce. V2a: removing the gates on an intervened register and preparing it in the intervention value gives the truncated factorization. | both manuscript models and 1,000 random specifications over four graph shapes, every non-root node, both intervention arms: worst error 5.6e-16, inside and beyond composed angle pi |
| `v04_additive_family_fit.py` | the additive family cannot represent an arbitrary table once a node has two or more parents; how large is the residual | best-found minimax error over 200 random two-parent tables: median 0.021, worst 0.079 (upper bounds on the true minimum) |

Script numbers are kept from the author's internal validation suite so that
result files can be traced back to it.

`common.py` holds the two manuscript circuits, the exact reference
distribution the circuits are compared against, and the result format.
Nothing else is shared between the scripts.

## Running

These scripts need Qiskit 2, whereas the case-study code in `../src` is pinned
to Qiskit 1.4.3 by the top-level `requirements.txt`. Use a separate virtual
environment for this directory:

    python3 -m venv .venv-validation
    source .venv-validation/bin/activate
    pip install -r requirements.txt
    python3 v11_additive_exactness.py
    python3 v04_additive_family_fit.py

Each script is seeded and deterministic. `v11` runs in a few seconds; `v04`
runs two global searches per table and takes several minutes.

## Provenance

Every `results/*.json` file records the SHA256 of the script that produced
it and of `common.py`, together with the Python, numpy, scipy and qiskit
versions in use. A stored number can therefore be checked against the
sources without a version-control system: recompute the two digests and
compare. Changing `common.py` changes every result's provenance even when
the script itself is untouched.

## Reading the numbers

`v11` reports the largest absolute difference between a basis-state
probability read from the statevector and the reference factorization. A
value of order 1e-16 is floating-point rounding; the claim is exactness, and
anything materially above that would be a failure. The script classifies each
random specification by whether its largest composed angle exceeds pi, and
reports both groups separately: exactness holds in both, because
sin^2(theta/2) is the Born probability for any theta. What is lost beyond pi
is only the monotone reading of an angle increment as a positive change in
probability, which is a limitation of the parameterization discussed in the
manuscript, not of the circuit's statistics.

`v04` reports the smallest max-entry error found when fitting a random
two-parent table with the three angles of the additive family. The two
searches used (grid-multistart Nelder-Mead and differential evolution) are
not certified global minimizers, so each reported residual is an upper bound
on the true minimum. The family never reproduces a generic table exactly; the
residual is a few percentage points.
