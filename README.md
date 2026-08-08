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
-   **/data**: JSON results.
    -   `simpson_paradox_results.json` — three-variable model (Table 1).
    -   `results_10_variable_backdoor.json` — ten-variable model (Table 2).
    -   `simpson_paradox_ionq_aria-1_qpu_3trials_analysis.json` — IonQ hardware run.
    -   `results_10_qubit.json` — earlier ten-variable output retained from a previous manuscript version. Its stratified rows are unweighted averages over strata and therefore differ from the population-weighted (g-formula) values reported in the current Table 2; it is not the source of any table in the present version.
-   **/figures**: Publication-ready figures (PDFs). Some correspond to earlier manuscript versions and are not used in the current text.
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
python analysis/draw_circuits_standalone.py            # circuit diagrams
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
