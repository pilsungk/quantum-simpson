# Implementing Pearl's DO-Calculus on Quantum Circuits: A Simpson-Type Case Study on NISQ Hardware

This repository contains the source code, data, and analysis scripts for the paper "Implementing Pearl's DO-Calculus on Quantum Circuits: A Simpson-Type Case Study on NISQ Hardware".

**arXiv Preprint:** [https://arxiv.org/abs/2509.00744]

## Abstract

Distinguishing correlation from causation is a central challenge in machine intelligence, and Pearl's $\mathcal{DO}$-calculus provides a rigorous symbolic framework for reasoning about interventions. A complementary question is whether such intervention logic can be given *executable semantics* on physical quantum devices. Our approach maps causal networks onto quantum circuits, where nodes are encoded in qubit registers, probabilistic links are implemented by controlled-rotation gates, and interventions are realized by a structural remodeling of the circuit---a physical analogue of Pearl's ``graph surgery'' that we term *circuit surgery*. We show that, for a family of 3-node confounded treatment models (including a Simpson-type reversal), the post-surgery circuits reproduce exactly the interventional distributions prescribed by the corresponding classical $\mathcal{DO}$-calculus. We then demonstrate a proof-of-principle experimental realization on an IonQ Aria trapped-ion processor and a 10-qubit synthetic healthcare model, observing close agreement between hardware estimates and classical baselines under realistic noise. We do not claim quantum speedup; instead, our contribution is to establish a concrete pathway by which causal graphs and Pearl-style interventions can be represented, executed, and empirically tested within the formalism of quantum circuits.


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
