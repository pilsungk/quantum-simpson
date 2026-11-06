# A Circuit-Level Representation of Pearl-Style Interventions: A Simpson-Paradox Case Study for Binary Structural Causal Models

This repository contains the source code, data, and analysis scripts for the
manuscript "A Circuit-Level Representation of Pearl-Style Interventions: A
Simpson-Paradox Case Study for Binary Structural Causal Models".

**arXiv Preprint:** [https://arxiv.org/abs/2509.00744]

## Manuscript Versions and Code State

The manuscript framing changed across arXiv versions: earlier versions
emphasized executable semantics on quantum hardware, while the current version
limits its claim to the representational level for a restricted family of
binary SCMs. The circuit construction is numerically unchanged; only comments
and reported wording were updated to match the current framing.

| Manuscript version | Code state |
|---|---|
| arXiv v1, v2 | tag `pre-v3-20260228` |
| arXiv v3 (current) | `main` |

## Repository Structure

-   **/src**: Contains the core Python modules and experiment execution scripts.
    -   `simpson_do.py`: Three-variable model — observational and post-surgery interventional circuits.
    -   `multi_level_simpson_paradox.py`: Ten-variable network — model definition and circuit construction.
    -   `dowhy_backdoor_validation.py`: Produces Table 2 — DoWhy identifies the adjustment set; every effect is then estimated nonparametrically (g-formula) with across-trial confidence intervals.
    -   `run-experiments-simpson.py`: Produces Table 1 — multi-trial runner for the three-variable model.
    -   `simpson_paradox_ionq_standalone.py`: Script to run the three-variable experiment on IonQ backends.
-   **/analysis**: Contains Python scripts used to generate the figures for the paper from the data in the `/data` directory.
    -   `ionq_trial_level_analysis.py`: Recomputes the hardware column of the three-qubit results table from the raw per-trial counts (`data/ionq_aria1_qpu_trials.json`) with Student's t intervals over the three QPU trials, and records estimator definitions, post-selection rates and recorded shot totals. Added for the revised manuscript.
-   **/data**: JSON results.
    -   `simpson_paradox_results.json` — three-variable model (Table 1).
    -   `results_10_variable_backdoor.json` — ten-variable model (Table 2).
    -   `ionq_aria1_qpu_trials.json` — raw per-trial counts of the three IonQ Aria-1 QPU executions, with provenance (which trials of the original run are included and why the other two, produced by a local-simulator fallback after quota exhaustion, are not). This is the input to the hardware re-analysis.
    -   `ionq_aria1_qpu_trial_level_analysis.json` — output of `analysis/ionq_trial_level_analysis.py`; source of the hardware column in the revised results table.
    -   `simpson_paradox_ionq_aria-1_qpu_3trials_analysis.json` — original summary of the IonQ hardware run (point estimates with normal-approximation intervals), retained as submitted; superseded by the trial-level analysis above.
    -   `results_10_qubit.json` — earlier ten-variable output retained from a previous manuscript version. Its stratified rows are unweighted averages over strata and therefore differ from the population-weighted (g-formula) values reported in the current Table 2; it is not the source of any table in the present version.
-   **/figures**: Publication-ready figures (PDFs). Some correspond to earlier manuscript versions and are not used in the current text.
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
python analysis/simpson-paradox-resolution-10q.py      # 10-qubit figure (earlier manuscript version)
python analysis/draw_circuits_standalone.py            # 3-qubit circuit diagrams
python validation/draw_ten_qubit_circuits.py           # 10-qubit circuit diagrams (validation venv)
```

### 3. Rerunning the simulations (optional)

Run from inside `/src` (the scripts import each other by module name):

```bash
cd src
python run-experiments-simpson.py          # three-variable, 30 trials (Table 1)
python dowhy_backdoor_validation.py        # ten-variable, 30 trials (Table 2)
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

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{kang2025simpson,
      title={A Circuit-Level Representation of Pearl-Style Interventions: A Simpson-Paradox Case Study for Binary Structural Causal Models}, 
      author={Pilsung Kang},
      year={2025},
      eprint={2509.00744},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
