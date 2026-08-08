"""
dowhy_backdoor_validation.py

Validation for the ten-variable model (Result 2 of the manuscript).

Goal: check whether the circuit-surgery interventional ATE agrees with a
standard backdoor-adjustment reference. DoWhy is used ONLY to identify a
valid backdoor adjustment set from the DAG; the numeric effect is then
computed nonparametrically (empirical g-formula) on the simulated binary
data, so that no estimator (e.g. linear regression) misspecification can
enter. All variables are binary, so the empirical adjustment is exact up
to sampling noise.

Pipeline:
  1. run observational circuit -> counts -> binary DataFrame
  2. build DAG (edge list taken from the circuit construction)
  3. DoWhy identify_effect -> backdoor adjustment set Z
  4. nonparametric backdoor: sum_z [P(O=1|T=1,z) - P(O=1|T=0,z)] P(z)
  5. circuit surgery ATE from intervention circuits do(T=1), do(T=0)
  6. print comparison table rows

Requires: qiskit, qiskit-aer, pandas, numpy, dowhy, networkx
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import qiskit
from scipy.stats import t as t_dist

from multi_level_simpson_paradox import MultiLevelSimpsonsParadox


# Variable names in qubit order (index = qubit id).
# Must match MultiLevelSimpsonsParadox.qubit_names.
VAR_NAMES = [
    "Age", "Income", "Region", "Gender_bias", "Treatment",
    "Insurance", "Hospital", "Doctor", "Outcome", "Satisfaction",
]

TREATMENT = "Treatment"
OUTCOME = "Outcome"

# Directed edges of the DAG, transcribed directly from
# create_complex_confounded_circuit() in multi_level_simpson_paradox.py.
# This list is the single source of truth for the graph; do not hardcode
# an edge count elsewhere.
DAG_EDGES = [
    ("Age", "Income"),
    ("Income", "Region"),
    ("Region", "Gender_bias"),
    ("Age", "Treatment"),
    ("Income", "Treatment"),
    ("Gender_bias", "Treatment"),
    ("Treatment", "Insurance"),
    ("Age", "Insurance"),
    ("Insurance", "Hospital"),
    ("Region", "Hospital"),
    ("Hospital", "Doctor"),
    ("Gender_bias", "Doctor"),
    ("Age", "Outcome"),
    ("Region", "Outcome"),
    ("Treatment", "Outcome"),
    ("Doctor", "Outcome"),
    ("Hospital", "Outcome"),
    ("Outcome", "Satisfaction"),
    ("Doctor", "Satisfaction"),
]


def counts_to_dataframe(counts):
    """Convert Qiskit measurement counts to a binary DataFrame.

    Each measured bitstring is parsed into the ten variables using the
    same parser as the model, then replicated by its count so that the
    DataFrame reflects the empirical distribution.

    Args:
        counts: dict mapping bitstring -> integer count (from AerSimulator).

    Returns:
        pandas DataFrame with one column per variable in VAR_NAMES and one
        row per shot.
    """
    model = MultiLevelSimpsonsParadox()
    rows = []
    for outcome_str, n in counts.items():
        parsed = model.parse_measurement_outcome(outcome_str)
        # parsed order matches VAR_NAMES (see parse_measurement_outcome)
        rows.extend([parsed] * int(n))
    df = pd.DataFrame(rows, columns=VAR_NAMES)
    return df


def build_gml_graph(edges, nodes):
    """Build a GML graph string for DoWhy from an edge list.

    Args:
        edges: list of (source, target) name tuples.
        nodes: list of node names.

    Returns:
        GML format string accepted by dowhy.CausalModel(graph=...).
    """
    lines = ["graph [", "directed 1"]
    index = {}
    for i, name in enumerate(nodes):
        index[name] = i
        lines.append("node [ id {0} label \"{1}\" ]".format(i, name))
    for src, dst in edges:
        lines.append("edge [ source {0} target {1} ]".format(
            index[src], index[dst]))
    lines.append("]")
    return "\n".join(lines)


def identify_backdoor_set(df, edges, nodes, treatment, outcome):
    """Use DoWhy only to identify a valid backdoor adjustment set.

    The numeric estimate is NOT taken from DoWhy; we only read off the
    backdoor variables it identifies from the graph.

    Args:
        df: binary DataFrame of simulated data.
        edges, nodes: DAG specification.
        treatment, outcome: variable names.

    Returns:
        Sorted list of variable names in the identified backdoor set
        (treatment and outcome excluded). Returns an empty list if DoWhy
        reports no backdoor variables.
    """
    from dowhy import CausalModel

    gml = build_gml_graph(edges, nodes)
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=gml,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)

    # Collect backdoor variables across whatever backdoor sets DoWhy exposes.
    backdoor_vars = set()
    bvars = getattr(identified, "backdoor_variables", None)
    if isinstance(bvars, dict):
        for vs in bvars.values():
            backdoor_vars.update(vs)
    elif isinstance(bvars, (list, set)):
        backdoor_vars.update(bvars)

    # Fallback: parse get_backdoor_variables() if attribute is empty.
    if not backdoor_vars:
        try:
            backdoor_vars.update(identified.get_backdoor_variables())
        except Exception:
            pass

    backdoor_vars.discard(treatment)
    backdoor_vars.discard(outcome)
    return sorted(backdoor_vars)

def nonparametric_backdoor_ate(df, treatment, outcome, adjustment_set):
    """Compute the backdoor-adjusted ATE nonparametrically (g-formula).

    For binary treatment and outcome:
        ATE = sum_z [ P(O=1 | T=1, z) - P(O=1 | T=0, z) ] P(z)
    using empirical frequencies over the adjustment set z. Strata with no
    support for one of the treatment arms are skipped (their P(z) weight is
    re-normalized out), which is the standard empirical-overlap handling.

    Args:
        df: binary DataFrame.
        treatment, outcome: variable names.
        adjustment_set: list of conditioning variable names.

    Returns:
        Float ATE estimate.
    """
    n_total = len(df)
    if not adjustment_set:
        # No confounders to adjust for: ATE reduces to the difference in
        # outcome means across treatment arms.
        p1 = df.loc[df[treatment] == 1, outcome].mean()
        p0 = df.loc[df[treatment] == 0, outcome].mean()
        return float(p1 - p0)

    ate = 0.0
    used_weight = 0.0
    grouped = df.groupby(adjustment_set)
    for _, stratum in grouped:
        treated = stratum[stratum[treatment] == 1]
        control = stratum[stratum[treatment] == 0]
        if len(treated) == 0 or len(control) == 0:
            # No overlap in this stratum; cannot form the contrast.
            continue
        p1 = treated[outcome].mean()
        p0 = control[outcome].mean()
        weight = len(stratum) / n_total
        ate += (p1 - p0) * weight
        used_weight += weight

    if used_weight > 0:
        # Re-normalize over strata that had overlap.
        ate = ate / used_weight
    return float(ate)

def trial_ci(values, confidence=0.95):
    """95% CI from across-trial estimates (t-distribution).

    Matches calculate_confidence_interval in run-experiments-simpson.py:
    mean +/- t_{n-1, 0.975} * std / sqrt(n), with ddof=1.

    Args:
        values: list of per-trial point estimates.
        confidence: CI level (default 0.95).

    Returns:
        (mean, lo, hi) tuple.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    std_err = float(arr.std(ddof=1) / np.sqrt(n))
    crit = float(t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1))
    lo = mean - crit * std_err
    hi = mean + crit * std_err
    return mean, lo, hi


def run_validation(shots=15000, n_trials=30, seed=0, save_path=None):
    """Run the backdoor-vs-circuit comparison with across-trial CIs.

    Every method is evaluated once per trial on independent circuit
    executions, so all rows of the table share the same statistical
    treatment (across-trial 95% CI). DoWhy is used only to identify the
    backdoor adjustment set; the numeric backdoor effect is computed
    nonparametrically on each trial's data.

    Args:
        shots: shots per circuit execution.
        n_trials: number of independent trials (CI resolution).
        seed: base RNG seed for reproducibility of the simulator stream.
        save_path: if given, write the aggregated results to this path as
            JSON. Nothing is written when None.

    Prints the Result 2 table rows with mean and 95% CI, and returns the
    aggregated result dictionary.
    """
    model = MultiLevelSimpsonsParadox(shots=shots, n_trials=n_trials)

    obs_list = []
    age_list = []
    region_list = []
    backdoor_list = []
    circuit_list = []
    p_do1_list = []
    p_do0_list = []
    adjustment_set = None

    for t in range(n_trials):
        run_seed = seed + t

        # --- observational data from the circuit ---
        obs_circuit = model.create_complex_confounded_circuit()
        obs_counts = model.simulator.run(
            obs_circuit, shots=shots, seed_simulator=run_seed
        ).result().get_counts()
        df = counts_to_dataframe(obs_counts)

        # identify backdoor set once (graph-determined; stable across trials)
        if adjustment_set is None:
            adjustment_set = identify_backdoor_set(
                df, DAG_EDGES, VAR_NAMES, TREATMENT, OUTCOME)
            print("DoWhy-identified backdoor adjustment set:")
            print("  ", adjustment_set)

        obs_list.append(
            observational_unadjusted_ate(df, TREATMENT, OUTCOME))
        age_list.append(
            single_variable_stratified_ate(df, TREATMENT, OUTCOME, "Age"))
        region_list.append(
            single_variable_stratified_ate(df, TREATMENT, OUTCOME, "Region"))
        backdoor_list.append(
            nonparametric_backdoor_ate(
                df, TREATMENT, OUTCOME, adjustment_set))

        # --- circuit-surgery ATE from intervention circuits ---
        int1 = model.create_intervention_circuit(do_treatment=1)
        c1 = model.simulator.run(
            int1, shots=shots, seed_simulator=run_seed
        ).result().get_counts()
        df1 = counts_to_dataframe(c1)
        p1 = df1[OUTCOME].mean()

        int0 = model.create_intervention_circuit(do_treatment=0)
        c0 = model.simulator.run(
            int0, shots=shots, seed_simulator=run_seed
        ).result().get_counts()
        df0 = counts_to_dataframe(c0)
        p0 = df0[OUTCOME].mean()

        p_do1_list.append(float(p1))
        p_do0_list.append(float(p0))
        circuit_list.append(float(p1 - p0))

    # --- aggregate across trials ---
    rows = [
        ("Observational (unadjusted)", obs_list),
        ("Stratified by Age (partial)", age_list),
        ("Stratified by Region (partial)", region_list),
        ("Backdoor adjustment (DoWhy set)", backdoor_list),
        ("Circuit surgery representation", circuit_list),
    ]

    print()
    print("Result 2 comparison (ten-variable model, "
          "{0} trials, {1} shots):".format(n_trials, shots))
    print("  {0:<36s} {1:>8s}   {2:>16s}".format("Method", "ATE", "95% CI"))
    for name, vals in rows:
        mean, lo, hi = trial_ci(vals)
        print("  {0:<36s} {1:>8.3f}   [{2:.3f}, {3:.3f}]".format(
            name, mean, lo, hi))

    pm1, _, _ = trial_ci(p_do1_list)
    pm0, _, _ = trial_ci(p_do0_list)
    print()
    print("  P(O=1 | do(T=1)) = {0:.3f}".format(pm1))
    print("  P(O=1 | do(T=0)) = {0:.3f}".format(pm0))

    # --- assemble the result record (no recomputation; the same values
    #     that were just printed) ---
    methods = {}
    for name, vals in rows:
        mean, lo, hi = trial_ci(vals)
        methods[name] = {
            "ate_mean": round(float(mean), 6),
            "ci_95": [round(float(lo), 6), round(float(hi), 6)],
            "per_trial": [round(float(v), 6) for v in vals],
        }

    result = {
        "experiment_name":
            "Ten-Variable Model - Backdoor Reference vs Circuit Surgery",
        "metadata": {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "qiskit_version": qiskit.__version__,
            "n_trials": n_trials,
            "shots_per_trial": shots,
            "base_seed": seed,
            "treatment": TREATMENT,
            "outcome": OUTCOME,
            "adjustment_set_source": "DoWhy identify_effect",
            "effect_estimator": "nonparametric g-formula (empirical)",
        },
        "adjustment_set": sorted(adjustment_set),
        "methods": methods,
        "interventional_probabilities": {
            "P(outcome=1 | do(treatment=1))": round(float(pm1), 6),
            "P(outcome=1 | do(treatment=0))": round(float(pm0), 6),
        },
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as handle:
            json.dump(result, handle, indent=2)
        print()
        print("  results written to {0}".format(save_path))

    return result


def observational_unadjusted_ate(df, treatment, outcome):
    """Crude observational contrast P(O=1|T=1) - P(O=1|T=0), no adjustment."""
    p1 = df.loc[df[treatment] == 1, outcome].mean()
    p0 = df.loc[df[treatment] == 0, outcome].mean()
    return float(p1 - p0)


def single_variable_stratified_ate(df, treatment, outcome, stratifier):
    """Partial adjustment by a single variable (illustrative only)."""
    return nonparametric_backdoor_ate(
        df, treatment, outcome, [stratifier])

if __name__ == "__main__":
    run_validation(
        save_path=os.path.join(
            os.path.dirname(__file__), "..", "data",
            "results_10_variable_backdoor.json"),
    )
