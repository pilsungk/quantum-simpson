# generate_figure_1_comparative.py
# This script generates the final comparative bar chart for Figure 1,
# showing both ideal simulation and real QPU results side-by-side.

import matplotlib.pyplot as plt
import numpy as np

# --- 1. Final Data from Simulation (Source 11) and QPU (Source 19) ---
labels = ['Male (G=0)', 'Female (G=1)', 'Overall (Obs.)', 'Overall (Causal)']

# Ideal Simulation Data
sim_means = [0.166, 0.296, -0.061, 0.232]
sim_ci = [[0.163, 0.169], [0.292, 0.300], [-0.063, -0.058], [0.230, 0.234]]
sim_errors = [(upper - lower) / 2 for lower, upper in sim_ci]

# IonQ QPU Data (n=3 trials)
qpu_means = [0.116, 0.292, -0.026, 0.222]
qpu_ci = [[0.105, 0.127], [0.184, 0.400], [-0.066, 0.014], [0.195, 0.249]]
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
ax.set_ylabel('Treatment Effect (ΔP)', fontsize=26)
#ax.set_title("3-Qubit Simpson's Paradox: Ideal Simulation vs. Real QPU Results", fontsize=16, pad=20)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylim(-0.13, 0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)
plt.yticks(fontsize=22)

ax.legend(fontsize=20)

ax.bar_label(rects1, fmt='%.3f', padding=3, fontsize=19)
ax.bar_label(rects2, fmt='%.3f', padding=3, fontsize=19)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
fig.tight_layout()

# --- 4. Save the Figure as a PDF ---
plt.savefig('figure_1_comparative.pdf', bbox_inches='tight')

print("Comparative Figure 1 has been successfully generated and saved as 'figure_1_comparative.pdf'")
