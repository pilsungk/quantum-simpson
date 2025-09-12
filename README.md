# Quantum Causality: Resolving Simpson's Paradox with DO-Calculus

This repository contains the source code, data, and analysis scripts for the paper "Quantum Causality: Resolving Simpson's Paradox with DO-Calculus".

**arXiv Preprint:** [https://arxiv.org/abs/2509.00744]

## Abstract

Distinguishing correlation from causation is a fundamental challenge in machine intelligence, often representing a critical barrier to building robust and trustworthy systems. While Pearl's $\mathcal{DO}$-calculus provides a rigorous, abstract framework for causal inference, a parallel challenge lies in its physical implementation. Here, we apply and experimentally validate a quantum algorithmic framework that realizes causal interventions by directly manipulating a physical system's evolution. Our approach maps causal networks onto quantum circuits where probabilistic links are encoded by controlled-rotation gates, and interventions are realized by a structural remodeling of the circuit---a physical analogue to Pearl's "graph surgery". We demonstrate the method's efficacy by resolving Simpson's Paradox in a 3-qubit model, and show its scalability by quantifying confounding bias in a 10-qubit healthcare simulation. Critically, we provide a proof-of-principle experimental validation on an IonQ Aria quantum computer, successfully reproducing the paradox and its resolution in the presence of real-world noise. This work establishes a practical pathway for quantum causal inference, offering a new computational tool to address deep-rooted challenges in algorithmic fairness and explainable AI (XAI).

## Repository Structure

-   **/src**: Contains the core Python modules and experiment execution scripts.
    -   `simpson_do.py`: Core logic for the 3-qubit Simpson's Paradox model.
    -   `multi_level_simpson_paradox.py`: Core logic for the 10-qubit scalability model.
    -   `run-experiments-simpson.py`: Script to run multiple trials of the 3-qubit simulation to generate statistical data.
    -   `simpson_paradox_ionq_standalone.py`: Script to run the 3-qubit experiment on IonQ backends (simulator or QPU).
-   **/analysis**: Contains Python scripts used to generate the figures for the paper from the data in the `/data` directory.
-   **/data**: Contains the final JSON data files from the simulation and QPU experiments.
-   **/figures**: Contains the publication-ready figures (PDFs) as they appear in the paper.
-   `requirements.txt`: A list of required Python packages for reproducibility.
-   `LICENSE`: The license for this project (MIT License).

## Reproducibility

### 1. Setup Environment

To set up the necessary environment, install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Generating Figures from Pre-computed Data

The final, processed data from our experiments is provided in the `/data` directory. To regenerate the main figures for the paper from this data, run the scripts in the `/analysis` directory:

```bash
# Generate a figure for 3-qubit sim vs. QPU results
python analysis/simpson-paradox-3q_sim_vs_qpu.py

# Generate a figure for 10-qubit results
python analysis/simpson-paradox-resolution-10q.py

# Generate circuit diagrams
python analysis/draw_circuits_standalone.py
```

### 3. Regenerating Simulation Data (Optional)

If you wish to regenerate the simulation data from scratch, you can run the scripts in the `/src` directory.

```bash
# Regenerate the 3-qubit simulation data (30 trials)
# This will take some time.
python src/run-experiments-simpson.py

# Regenerate the 10-qubit simulation data (10 trials)
# This will also take some time.
python src/multi_level_simpson_paradox.py
```
The output JSON files will be saved in the root directory.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{kang2025simpson,
      title={Quantum Causality: Resolving Simpson's Paradox with DO-Calculus}, 
      author={Pilsung Kang},
      year={2025},
      eprint={2509.00744},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
