"""Trial-level re-analysis of the IonQ Aria-1 hardware runs.

Input : data/ionq_aria1_qpu_trials.json  (three QPU trials, raw counts)
Output: data/ionq_aria1_qpu_trial_level_analysis.json

Why this script exists. The original run script (src/simpson_paradox_ionq_
standalone.py) pools the counts of all trials and reports point estimates
only; the confidence intervals in data/simpson_paradox_ionq_aria-1_qpu_
3trials_analysis.json were produced outside the repository with a normal
approximation. This script recomputes everything from the raw per-trial
counts so that the hardware column of the results table can be reproduced,
and replaces the normal approximation by Student's t on the three trials,
matching the method stated in the manuscript and used for the simulator.

What is computed, in order of importance.

  1. Estimator C per trial (the manuscript's estimator): within each
     interventional circuit, post-select shots whose treatment register
     reads the intended value, then standardize over G with P(G) taken
     from the observational circuit of the same trial. Observational
     subgroup and aggregate contrasts are plain conditional frequencies.
     The per-trial values reproduce the published point estimates.
  2. Trial-level Student's t 95% intervals (n = 3, 2 degrees of freedom)
     for the four rows of the hardware column.
  3. Estimator A (direct contrast, no post-selection, no standardization)
     as a sensitivity check on the choice of estimator.
  4. Pooled estimator C over the three trials with a parametric bootstrap
     interval. This treats the 3 x 1,025 shots as one sample and does
     not reflect trial-to-trial variation; it is reported as supplementary.
  5. An exploratory overdispersion check: observed between-trial variance
     of C against the shot-noise variance expected from a parametric
     bootstrap within each trial. With two degrees of freedom the test has
     little power and is recorded as a diagnostic only.
  6. Data-handling records: the fraction of shots discarded by
     post-selection, and the recorded total per circuit run (QPU counts
     are round(p * shots) per outcome, see the data file's provenance).

Nothing here uses Qiskit. The script is deterministic (seeded bootstrap).
"""

import json
import math
import os

import numpy as np
from scipy.stats import t as t_dist

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
IN_PATH = os.path.join(DATA, "ionq_aria1_qpu_trials.json")
OUT_PATH = os.path.join(DATA, "ionq_aria1_qpu_trial_level_analysis.json")

SEED = 42
N_BOOT = 20000
CELLS = [(g, t, o) for g in (0, 1) for t in (0, 1) for o in (0, 1)]


# ------------------------------------------------------------ parsing

def parse(counts):
    """Map an IonQ/Qiskit count dict to {(g, t, o): count}.

    Keys look like '010 000': the first group is the three measured bits in
    little-endian order, so the last character is q0 = G, the middle is
    q1 = T, the first is q2 = O.
    """
    out = {}
    for key, n in counts.items():
        bits = key.split()[0]
        g, t, o = int(bits[-1]), int(bits[-2]), int(bits[-3])
        out[(g, t, o)] = out.get((g, t, o), 0) + int(n)
    return out


def vec(d):
    return np.array([d.get(c, 0) for c in CELLS], dtype=float)


def frac(d, sel):
    """P(O=1) among shots satisfying sel(g, t); returns (p, n)."""
    num = sum(n for (g, t, o), n in d.items() if sel(g, t) and o == 1)
    den = sum(n for (g, t, o), n in d.items() if sel(g, t))
    return (num / den if den else float("nan")), den


# --------------------------------------------------------- estimators

def estimator_c(obs, d1, d0):
    """Manuscript estimator: post-select on T, standardize over G."""
    n_g = {g: sum(n for (gg, t, o), n in obs.items() if gg == g) for g in (0, 1)}
    tot = n_g[0] + n_g[1]
    eff = 0.0
    for g in (0, 1):
        p1, _ = frac(d1, lambda gg, t, g=g: gg == g and t == 1)
        p0, _ = frac(d0, lambda gg, t, g=g: gg == g and t == 0)
        eff += (n_g[g] / tot) * (p1 - p0)
    return eff


def estimator_a(d1, d0):
    """Direct contrast P(O=1 | do(T=1)) - P(O=1 | do(T=0)), no post-selection."""
    p1, _ = frac(d1, lambda g, t: True)
    p0, _ = frac(d0, lambda g, t: True)
    return p1 - p0


def observational_rows(obs):
    male = frac(obs, lambda g, t: g == 0 and t == 1)[0] - frac(obs, lambda g, t: g == 0 and t == 0)[0]
    female = frac(obs, lambda g, t: g == 1 and t == 1)[0] - frac(obs, lambda g, t: g == 1 and t == 0)[0]
    overall = frac(obs, lambda g, t: t == 1)[0] - frac(obs, lambda g, t: t == 0)[0]
    return male, female, overall


def discarded_by_postselection(d1, d0):
    n1 = sum(d1.values()); x1 = sum(n for (g, t, o), n in d1.items() if t != 1)
    n0 = sum(d0.values()); x0 = sum(n for (g, t, o), n in d0.items() if t != 0)
    return {"intervention_t1": {"discarded": x1, "total": n1},
            "intervention_t0": {"discarded": x0, "total": n0}}


# ---------------------------------------------------------- intervals

def t_interval(values, confidence=0.95):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    crit = float(t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1))
    return {"mean": mean, "sd": sd, "se": se, "df": n - 1,
            "critical": crit,
            "ci_95": [mean - crit * se, mean + crit * se],
            "values": [float(v) for v in arr]}


def resample(rng, v):
    n = int(v.sum())
    return dict(zip(CELLS, rng.multinomial(n, v / v.sum())))


# ---------------------------------------------------------------- main

def main():
    rng = np.random.default_rng(SEED)
    rec = json.load(open(IN_PATH))
    trials = [(parse(t["observational"]), parse(t["intervention_t1"]),
               parse(t["intervention_t0"]), t["trial"]) for t in rec["trials"]]

    # 1-2. per-trial estimates and t intervals
    per = {"male": [], "female": [], "overall_obs": [], "interventional_c": [],
           "interventional_a": []}
    for obs, d1, d0, _ in trials:
        m, f, o = observational_rows(obs)
        per["male"].append(m); per["female"].append(f); per["overall_obs"].append(o)
        per["interventional_c"].append(estimator_c(obs, d1, d0))
        per["interventional_a"].append(estimator_a(d1, d0))
    rows = {k: t_interval(v) for k, v in per.items()}

    # 4. pooled estimator C with parametric bootstrap
    P = [sum(vec(x[i]) for x in trials) for i in range(3)]
    pooled_c = estimator_c(*[dict(zip(CELLS, p)) for p in P])
    boots = [estimator_c(resample(rng, P[0]), resample(rng, P[1]), resample(rng, P[2]))
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pooled = {"estimate": float(pooled_c), "bootstrap_ci_95": [float(lo), float(hi)],
              "n_boot": N_BOOT,
              "note": ("treats all shots as one sample; does not reflect "
                       "trial-to-trial variation; supplementary only")}

    # 5. exploratory overdispersion check on estimator C
    within = []
    for obs, d1, d0, _ in trials:
        v = [vec(obs), vec(d1), vec(d0)]
        sims = [estimator_c(resample(rng, v[0]), resample(rng, v[1]), resample(rng, v[2]))
                for _ in range(N_BOOT)]
        within.append(float(np.var(sims, ddof=1)))
    shot_var = float(np.mean(within))
    obs_var = rows["interventional_c"]["sd"] ** 2
    ratio = obs_var / shot_var
    from scipy.stats import chi2
    stat = (len(trials) - 1) * ratio
    overdisp = {"between_trial_sd": rows["interventional_c"]["sd"],
                "shot_noise_sd": math.sqrt(shot_var),
                "variance_ratio": ratio, "chi2": stat, "df": len(trials) - 1,
                "p_value": float(1 - chi2.cdf(stat, len(trials) - 1)),
                "note": ("exploratory; two degrees of freedom give little power "
                         "to detect drift")}

    # 6. data-handling records
    discard = {str(k): discarded_by_postselection(d1, d0) for _, d1, d0, k in trials}
    tot_disc = sum(v[c]["discarded"] for v in discard.values() for c in v)
    tot_all = sum(v[c]["total"] for v in discard.values() for c in v)
    totals = {str(t["trial"]): t["recorded_totals"] for t in rec["trials"]}

    out = {
        "input_file": os.path.basename(IN_PATH),
        "method": {
            "primary": ("per-trial estimates, mean over 3 trials, Student's t "
                        "interval with 2 degrees of freedom"),
            "estimator_c": ("post-select shots whose treatment register reads "
                            "the intended value; standardize over G with P(G) "
                            "from the observational circuit of the same trial"),
            "estimator_a": "direct contrast on the interventional circuits, no post-selection, no standardization",
            "seed": SEED},
        "rows": rows,
        "pooled_estimator_c": pooled,
        "overdispersion_exploratory": overdisp,
        "post_selection": {"per_trial": discard,
                           "discarded_total": tot_disc, "shots_total": tot_all,
                           "fraction": tot_disc / tot_all},
        "recorded_totals_per_circuit": totals,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=1)

    # ------------------------------------------------------------- report
    print("Trial-level re-analysis of the IonQ Aria-1 runs (3 QPU trials)\n")
    print("{0:18s} {1:>9s}  {2:>22s}   {3}".format("row", "mean", "t 95% CI", "per-trial"))
    for k, label in (("male", "Obs. male (G=0)"), ("female", "Obs. female (G=1)"),
                     ("overall_obs", "Obs. overall"), ("interventional_c", "Interventional (C)"),
                     ("interventional_a", "Interventional (A)")):
        r = rows[k]
        print("{0:18s} {1:+9.4f}  [{2:+.3f}, {3:+.3f}]   {4}".format(
            label, r["mean"], r["ci_95"][0], r["ci_95"][1],
            " / ".join("{0:+.4f}".format(v) for v in r["values"])))
    print("\npooled C: {0:+.4f}  bootstrap 95% [{1:+.3f}, {2:+.3f}]  (supplementary)".format(
        pooled["estimate"], *pooled["bootstrap_ci_95"]))
    print("overdispersion (exploratory): ratio {0:.3f}, chi2 {1:.2f} on {2} df, p = {3:.3f}".format(
        overdisp["variance_ratio"], overdisp["chi2"], overdisp["df"], overdisp["p_value"]))
    print("post-selection discarded: {0}/{1} shots = {2:.2%}".format(tot_disc, tot_all, tot_disc / tot_all))
    print("recorded totals per circuit:", totals)
    print("\nwritten:", os.path.relpath(OUT_PATH))


if __name__ == "__main__":
    main()
