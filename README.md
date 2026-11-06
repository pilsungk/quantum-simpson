# Quantum-Circuit Representation of Pearl-Style Interventions

This repository contains the source code, data, and analysis scripts for the
manuscript "Quantum-Circuit Representation of Pearl-Style Interventions". An
earlier framing of the same work is on arXiv as
[2509.00744](https://arxiv.org/abs/2509.00744) (v1, v2).

## Manuscript Versions and Code State

The manuscript framing changed after arXiv v2: the arXiv versions emphasized
executable semantics on quantum hardware, while the current manuscript limits
its claim to the representational level for a restricted family of binary
SCMs. The circuit construction is numerically unchanged; comments and reported
wording were updated to match the current framing, and the validation,
hardware re-analysis, and figure scripts listed below were added for it.

| Manuscript version | Code state |
|---|---|
| arXiv 2509.00744 v1, v2 ("Implementing Pearl's DO-Calculus on Quantum Circuits: A Simpson-Type Case Study on NISQ Hardware") | tag `pre-v3-20260228` |
| Current manuscript ("Quantum-Circuit Representation of Pearl-Style Interventions") | `main` |

## Repository Structure

-   **/src**: Contains the core Python modules and experiment execution scripts.
    -   `simpson_do.py`: Three-variable model — observational and post-surgery interventional circuits.
    -   `multi_level_simpson_paradox.py`: Ten-variable network — model definition and circuit construction.
    -   `dowhy_backdoor_validation.py`: Produces the ten-variable results table — DoWhy identifies the adjustment set; every effect is then estimated nonparametrically (g-formula) with across-trial confidence intervals.
    -   `run-experiments-simpson.py`: Produces the simulated column of the three-variable results table — multi-trial runner for the three-variable model.
    -   `simpson_paradox_ionq_standalone.py`: Script to run the three-variable experiment on IonQ backends.
-   **/analysis**: Contains Python scripts used to generate the figures for the paper from the data in the `/data` directory.
    -   `ionq_trial_level_analysis.py`: Recomputes the hardware column of the three-qubit results table from the raw per-trial counts (`data/ionq_aria1_qpu_trials.json`) with Student's t intervals over the three QPU trials, and records estimator definitions, post-selection rates and recorded shot totals. Added for the revised manuscript.
-   **/data**: JSON results.
    -   `simpson_paradox_results.json` — three-variable model, simulated column of the three-variable results table.
    -   `results_10_variable_backdoor.json` — ten-variable model, simulated columns of the ten-variable results table; also the input to `validation/v12_exact_ten_variable.py`.
    -   `ionq_aria1_qpu_trials.json` — raw per-trial counts of the three IonQ Aria-1 QPU executions, with provenance (which trials of the original run are included and why the other two, produced by a local-simulator fallback after quota exhaustion, are not). This is the input to the hardware re-analysis.
    -   `ionq_aria1_qpu_trial_level_analysis.json` — output of `analysis/ionq_trial_level_analysis.py`; source of the hardware column in the revised results table.
    -   `simpson_paradox_ionq_aria-1_qpu_3trials_analysis.json` — original summary of the IonQ hardware run (point estimates with normal-approximation intervals), retained as submitted; superseded by the trial-level analysis above.
-   **/figures**: Publication-ready figures (PDFs). File names carry a figure-number prefix from an earlier version; the mapping to the current manuscript is (outputs of earlier manuscript versions are kept under tag `pre-v3-20260228`):

    | Manuscript figure | Repository file | Produced by |
    |---|---|---|
    | Fig. 1 (causal DAGs, observational / interventional) | `f1_causal_dag_3q.pdf`, `f1_intervention_dag_3q.pdf` | drawn externally |
    | Fig. 2 (three-qubit circuits) | `f2_observational_circuit_with_confounding.pdf`, `f2_interventional_circuit_implementing_do_T1.pdf` | `analysis/draw_circuits_standalone.py` |
    | Fig. 3 (three-variable, simulation vs. QPU) | `f3_comparative.pdf` | `analysis/simpson-paradox-3q_sim_vs_qpu.py` |
    | Figs. B1, B2 (ten-qubit circuits) | `ten_qubit_observational_circuit.pdf`, `ten_qubit_do_T1_circuit.pdf` | `validation/draw_ten_qubit_circuits.py` |
-   **/validation**: Exactness and expressivity checks for the additive-in-angle circuit family (added for the revised manuscript). Self-contained, with its own `README.md` and `requirements.txt`; needs Qiskit 2, unlike `/src`.
-   `requirements.txt`: A list of required Python packages for reproducibility.
-   `LICENSE`: The license for this project (MIT License).

## Reproducibility

### 1. Setup

The pinned versions in `requirements.txt` (Qiskit 1.4.3, qiskit-aer 0.17.1) are
what the reported numbers were produced with. A dedicated virtual environment is
recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Figures from the provided data

```bash
python analysis/simpson-paradox-3q_sim_vs_qpu.py       # 3-qubit sim vs QPU
python analysis/draw_circuits_standalone.py            # 3-qubit circuit diagrams
python validation/draw_ten_qubit_circuits.py           # 10-qubit circuit diagrams (validation venv)
```

### 3. Rerunning the simulations (optional)

Run from inside `/src` (the scripts import each other by module name):

```bash
cd src
python run-experiments-simpson.py          # three-variable, 30 trials
python dowhy_backdoor_validation.py        # ten-variable, 30 trials
python multi_level_simpson_paradox.py      # ten-variable model, exploratory run
```

Shot noise makes the third decimal fluctuate between runs; reported values are
means over trials with 95% confidence intervals.

### 4. Hardware re-analysis from raw counts

```bash
python analysis/ionq_trial_level_analysis.py   # hardware column, Student's t over 3 QPU trials
```

Needs only numpy and scipy. The original IonQ run script pooled the three
trials and reported point estimates; the intervals in the originally
submitted summary were computed outside this repository with a normal
approximation. The script above recomputes every row from the raw counts
with Student's t on the three trials, matching the method used for the
simulator, and writes `data/ionq_aria1_qpu_trial_level_analysis.json`.
Note that IonQ QPU counts are reconstructed by qiskit-ionq as
`round(p * shots)` per outcome from the probabilities the API returns, so
recorded totals can differ from the requested 1,024 by a shot or two; the
data file's provenance block records the totals.

### 5. Validation of the additive-in-angle family (Qiskit 2)

`/validation` checks, on an ideal statevector simulator, that the restricted
circuit family prepares exactly the distribution its angles induce and that
circuit surgery gives the truncated factorization (`v11`), measures how far
the family falls short of an arbitrary conditional probability table (`v04`),
and computes the exact values of the five contrasts in the ten-variable
results table by enumerating the induced joint distribution (`v12`). It
requires Qiskit 2 and therefore a separate environment; see
`validation/README.md`.

## Citation

Until the current manuscript is published, please cite the arXiv version:

```bibtex
@misc{kang2025simpson,
      title={Implementing Pearl's DO-Calculus on Quantum Circuits: A Simpson-Type Case Study on NISQ Hardware},
      author={Pilsung Kang},
      year={2025},
      eprint={2509.00744},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
