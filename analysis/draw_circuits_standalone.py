# draw_circuits_standalone.py 
# This script generates and saves the circuit diagrams for the 3-qubit model

import matplotlib.pyplot as plt
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np

# --- 1. Define named registers for clarity ---
q_gender = QuantumRegister(1, name="Gender (G)")
q_treatment = QuantumRegister(1, name="Treatment (T)")
q_outcome = QuantumRegister(1, name="Outcome (O)")
c_bits = ClassicalRegister(3, name="c")

# --- 2. Build and save the Observational (Confounded) Circuit ---
print("Building and drawing Observational Circuit...")
obs_qc = QuantumCircuit(q_gender, q_treatment, q_outcome, c_bits)

# Gate sequence for the observational circuit
obs_qc.h(q_gender)
obs_qc.barrier()

# -- MODIFIED SECTION: Statistically Robust Confounding Logic ---
# Males (G=0) get a high chance of treatment (e.g., ~85% prob)
obs_qc.x(q_gender) 
obs_qc.cry(2.4, q_gender, q_treatment) 
obs_qc.x(q_gender) 

# Females (G=1) get a low chance of treatment (e.g., ~15% prob)
obs_qc.cry(0.8, q_gender, q_treatment)
# --- END OF MODIFIED SECTION ---

obs_qc.barrier()
obs_qc.ry(0.3, q_outcome)
obs_qc.cry(1.0, q_gender, q_outcome)
obs_qc.cry(0.6, q_treatment, q_outcome)
obs_qc.barrier()
obs_qc.measure([q_gender[0], q_treatment[0], q_outcome[0]], [c_bits[0], c_bits[1], c_bits[2]])

# Draw the circuit and get the figure object
fig_obs = obs_qc.draw('mpl', style='iqx')
#fig_obs.suptitle("a) Observational Circuit with Confounding", fontsize=16)

# Save directly from the figure object
fig_obs.savefig("observational_circuit_with_confounding.pdf", bbox_inches='tight')
print("Saved to observational_circuit_with_confounding.pdf")
plt.close(fig_obs) # Close the figure to free up memory


# --- 3. Build and save the Interventional Circuit ---
print("\nBuilding and drawing Interventional Circuit...")
int_qc = QuantumCircuit(q_gender, q_treatment, q_outcome, c_bits)

# Gate sequence for the interventional circuit
int_qc.h(q_gender)
int_qc.barrier()
int_qc.x(q_treatment) # do(T=1)
int_qc.barrier()
int_qc.ry(0.3, q_outcome)
int_qc.cry(1.0, q_gender, q_outcome)
int_qc.cry(0.6, q_treatment, q_outcome)
int_qc.barrier()
int_qc.measure([q_gender[0], q_treatment[0], q_outcome[0]], [c_bits[0], c_bits[1], c_bits[2]])

# Draw the circuit and get the figure object
fig_int = int_qc.draw('mpl', style='iqx')
#fig_int.suptitle("b) Interventional Circuit implementing do(T=1)", fontsize=16)

# Save directly from the figure object
fig_int.savefig("interventional_circuit_implementing_do(T=1).pdf", bbox_inches='tight')
print("Saved to interventional_circuit_implementing_do(T=1).pdf")
plt.close(fig_int) # Close the figure to free up memory

print("\nPDF circuit diagram generation complete.")

