# This script generates the enhanced bar chart for Figure 2 using the full
# 10-qubit simulation data from Table 2 and saves it as a PDF.

import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data from the 10-Qubit Simulation (Source 26) ---
labels = ['Overall (Obs.)', 'Stratified\nby Region', 'Stratified\nby Age', 'Causal\nIntervention']
mean_effects = [0.377, 0.406, 0.497, 0.486]

# Calculate the error values (half of the confidence interval width)
ci_95 = [
    [0.371, 0.383],  # Observational CI
    [0.396, 0.416],  # Region Stratified CI
    [0.491, 0.503],  # Age Stratified CI
    [0.481, 0.490]   # Causal CI
]
errors = [(upper - lower) / 2 for lower, upper in ci_95]

# --- 2. Create the Bar Chart ---
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(10, 6))

# Use a progressive color scheme to tell the story
colors = ['#d62728', '#ff7f0e', '#ff7f0e', '#2ca02c'] # Red -> Orange -> Green
bar_labels = ['Biased Observation', 'Partial Correction', 'Partial Correction', 'True Effect']
bars = ax.bar(labels, mean_effects, yerr=errors, capsize=5, color=colors, edgecolor='black', width=0.6)

# --- 3. Add Labels, Title, and a Reference Line for the True Effect ---
ax.set_ylabel('Treatment Effect (ΔP)', fontsize=18)
#ax.set_title("Correcting Confounding Bias in 10-Qubit Simulation", fontsize=16, pad=20)
ax.set_ylim(0.35, 0.52) # Adjust ylim to focus on the differences
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add a horizontal dashed line for the "true" causal effect
true_effect_value = 0.486
ax.axhline(true_effect_value, color='green', linestyle='--', linewidth=1.5, label=f'True Causal Effect ({true_effect_value})')
ax.legend(fontsize=14)

ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=14)
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# --- 4. Save the Figure as a PDF ---
plt.savefig('figure_2_10q_enhanced.pdf', bbox_inches='tight')

print("Enhanced Figure 2 has been successfully generated and saved as 'figure_2_10q_enhanced.pdf'")

# To display the plot directly, uncomment the line below
# plt.show()
