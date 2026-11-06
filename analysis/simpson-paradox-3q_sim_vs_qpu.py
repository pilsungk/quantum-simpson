# simpson-paradox-3q_sim_vs_qpu.py
# This script generates the comparative bar chart (manuscript Fig. 3),
# showing both ideal simulation and real QPU results side-by-side.
#
# The QPU means and 95% intervals are read from
# data/ionq_aria1_qpu_trial_level_analysis.json (Student's t over the three
# trials, produced by analysis/ionq_trial_level_analysis.py from the raw
# counts), so the figure cannot drift from the hardware re-analysis.

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QPU_JSON = os.path.join(HERE, "..", "data", "ionq_aria1_qpu_trial_level_analysis.json")

# --- 1. Final Data from Simulation (Source 11) and QPU (Source 19) ---
labels = ['Male (G=0)', 'Female (G=1)', 'Overall (Obs.)', 'Overall (Causal)']

# Ideal Simulation Data
sim_means = [0.166, 0.296, -0.061, 0.232]
sim_ci = [[0.163, 0.169], [0.292, 0.300], [-0.063, -0.058], [0.230, 0.234]]
sim_errors = [(upper - lower) / 2 for lower, upper in sim_ci]

# IonQ QPU Data (n=3 trials): per-trial means and Student's t intervals
# (2 degrees of freedom), estimator C for the interventional contrast.
with open(QPU_JSON) as f:
    qpu_rows = json.load(f)["rows"]
qpu_keys = ["male", "female", "overall_obs", "interventional_c"]
qpu_means = [qpu_rows[k]["mean"] for k in qpu_keys]
qpu_ci = [qpu_rows[k]["ci_95"] for k in qpu_keys]
qpu_errors = [(upper - lower) / 2 for lower, upper in qpu_ci]

# --- 2. Create the Grouped Bar Chart ---
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(labels))
width = 0.35

# --- MODIFIED LINES: Changed colors ---
rects1 = ax.bar(x - width/2, sim_means, width, yerr=sim_errors, capsize=4,
                label='Ideal Simulation', color='#1f77b4', edgecolor='black') # Blue
rects2 = ax.bar(x + width/2, qpu_means, width, yerr=qpu_errors, capsize=4,
                label='IonQ QPU (n=3)', color='#ff7f0e', edgecolor='black') # Orange
# --- END OF MODIFIED LINES ---

# --- 3. Add Labels, Title, and Formatting ---
ax.set_ylabel(r'Treatment Effect ($\Delta P$)', fontsize=26)
#ax.set_title("3-Qubit Simpson's Paradox: Ideal Simulation vs. Real QPU Results", fontsize=16, pad=20)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylim(-0.15, 0.6)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)
plt.yticks(fontsize=22)

ax.legend(fontsize=20)

# Value labels sit beside the error bar at the bar end: simulation labels to
# the left of their bar, QPU labels to the right, so the whisker stays
# visible and labels of the two bars in a group do not collide.
for rects, side in ((rects1, -1), (rects2, +1)):
    for rect in rects:
        h = rect.get_height()
        xc = rect.get_x() + rect.get_width() / 2
        ax.text(xc + side * 0.03, h + (0.008 if h >= 0 else -0.008), '%.3f' % h,
                ha='left' if side > 0 else 'right',
                va='bottom' if h >= 0 else 'top', fontsize=17, zorder=5)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
fig.tight_layout()

# --- 4. Save the Figure as a PDF ---
out = os.path.join(HERE, "..", "figures", "f3_comparative.pdf")
plt.savefig(out, bbox_inches='tight')

for lab, m, ci in zip(labels, qpu_means, qpu_ci):
    print("QPU %-16s mean %+.3f  95%% t-CI [%+.3f, %+.3f]" % (lab, m, ci[0], ci[1]))
print("Comparative figure saved as", os.path.normpath(out))
