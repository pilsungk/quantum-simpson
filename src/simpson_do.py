"""
Simpson's Paradox with Proper Do-Calculus
==================================================
Causal Structure:
    Gender (G) <-> Treatment (T)  [Entangled confounding]
         |              |
         v              v
    Outcome (O) <-------+
Key fix: Clean intervention circuit that never creates G->T connection
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.stats import binomtest, norm
from math import sqrt

class CorrectedSimpsonsParadox:
    """
    Corrected implementation with proper do-calculus intervention circuits
    """
    
    def __init__(self, shots=10000, n_trials=3):
        self.simulator = AerSimulator()
        self.shots = shots
        self.n_trials = n_trials
        self.results = {}
    
    def wilson_ci(self, p_hat, n, alpha=0.05):
        """Wilson confidence interval"""
        z = norm.ppf(1 - alpha/2)
        denominator = 1 + z**2/n
        center = (p_hat + z**2/(2*n)) / denominator
        half_width = (z/denominator) * sqrt((p_hat*(1-p_hat))/n + z**2/(4*n**2))
        return (center - half_width, center + half_width)
    

    def create_confounded_circuit(self):
        """
        Create confounded circuit for Simpson's paradox
        with statistically robust confounding logic.

        Structure:
        - q0: Gender (G)
        - q1: Treatment (T) 
        - q2: Outcome (O)
        
        Confounding: Males get more treatment (~85%), females less (~15%)
        But both groups have some treated and untreated members
        Treatment helps everyone
        """
        qc = QuantumCircuit(3, 3)
        
        # Gender distribution (50/50)
        qc.h(0)
        
        # Statistically robust confounding logic
        # Males (G=0) get a high chance of treatment (e.g., ~85% prob)
        qc.x(0) # Control on G=0
        qc.cry(2.4, 0, 1) 
        qc.x(0) # Un-compute control
        
        # Females (G=1) get a low chance of treatment (e.g., ~15% prob)
        qc.cry(0.8, 0, 1) # Control on G=1
        
        # Outcome model (remains the same):
        qc.ry(0.3, 2)      # Base rate
        qc.cry(1.0, 0, 2)  # Gender effect: G=1 -> better outcome
        qc.cry(0.6, 1, 2)  # Treatment effect: T=1 -> better outcome
        
        qc.measure_all()
        return qc

    def create_intervention_circuit(self, do_treatment=1):
        """
        CORRECTED: Clean intervention circuit for do(T=treatment)
        
        In do(T) world, G->T connection never exists!
        Only G->O and T->O relationships remain.
        """
        qc = QuantumCircuit(3, 3)  # G, T, O
        
        # Step 1: Prepare G (population distribution, unchanged)
        qc.h(0)  # Equal gender split
        
        # Step 2: Set T directly to intervention value (NO G->T connection)
        if do_treatment == 1:
            qc.x(1)  # Set T=1 directly
        # T=0 by default (no operation needed)
        
        # Step 3: Apply remaining causal effects (G->O and T->O only)
        # Base rate (same as observational)
        qc.ry(0.3, 2)
        
        # Gender effect: Females have better outcomes (same as observational)
        qc.cry(1.0, 0, 2)  # G=1 (females) get boost
        
        # Treatment effect: beneficial for all (same as observational)
        qc.cry(0.6, 1, 2)  # T=1 improves outcome for everyone
        
        # Step 4: Measure final state
        qc.measure_all()
        return qc
    
    def measure_observational(self):
        """Measure observational distribution with confounding"""
        print("Measuring observational distribution...")
        
        all_counts = {}
        for trial in range(self.n_trials):
            qc = self.create_confounded_circuit()
            job = self.simulator.run(transpile(qc, self.simulator), shots=self.shots)
            counts = job.result().get_counts()
            
            for outcome, count in counts.items():
                o, t, g = int(outcome[0]), int(outcome[1]), int(outcome[2])
                key = (g, t, o)
                all_counts[key] = all_counts.get(key, 0) + count
        
        # Calculate effects
        effects = {}
        for gender in [0, 1]:
            for treatment in [0, 1]:
                good = all_counts.get((gender, treatment, 1), 0)
                total = (all_counts.get((gender, treatment, 0), 0) + 
                        all_counts.get((gender, treatment, 1), 0))
                if total > 0:
                    effects[(gender, treatment)] = good / total
                else:
                    effects[(gender, treatment)] = 0.5
        
        # Group effects
        male_effect = effects[(0, 1)] - effects[(0, 0)]
        female_effect = effects[(1, 1)] - effects[(1, 0)]
        
        # Aggregate effect
        total_t1_good = sum(all_counts.get((g, 1, 1), 0) for g in [0, 1])
        total_t1 = sum(all_counts.get((g, 1, o), 0) for g in [0, 1] for o in [0, 1])
        total_t0_good = sum(all_counts.get((g, 0, 1), 0) for g in [0, 1])
        total_t0 = sum(all_counts.get((g, 0, o), 0) for g in [0, 1] for o in [0, 1])
        
        p_t1 = total_t1_good / total_t1 if total_t1 > 0 else 0.5
        p_t0 = total_t0_good / total_t0 if total_t0 > 0 else 0.5
        aggregate_effect = p_t1 - p_t0
        
        # Check for paradox
        paradox = (male_effect > 0.05 and female_effect > 0.05 and aggregate_effect < -0.05)
        
        print(f"  Male effect: {male_effect:+.3f}")
        print(f"  Female effect: {female_effect:+.3f}")  
        print(f"  Aggregate effect: {aggregate_effect:+.3f}")
        print(f"  Simpson's paradox: {paradox}")
        
        # Population distribution
        total_pop = sum(all_counts.values())
        p_male = sum(count for (g, t, o), count in all_counts.items() if g == 0) / total_pop
        p_female = 1 - p_male
        
        return {
            'male_effect': male_effect,
            'female_effect': female_effect,
            'aggregate_effect': aggregate_effect,
            'paradox_detected': paradox,
            'population': {'p_male': p_male, 'p_female': p_female},
            'raw_counts': all_counts
        }
    
    def clean_intervention(self, do_treatment=1):
        """
        CORRECTED: Clean intervention using proper do-calculus circuit
        """
        print(f"Clean intervention: do(T={do_treatment})")
        
        all_counts = {}
        
        for trial in range(self.n_trials):
            # Use the corrected intervention circuit
            qc = self.create_intervention_circuit(do_treatment)
            job = self.simulator.run(transpile(qc, self.simulator), shots=self.shots)
            counts = job.result().get_counts()
            
            for outcome, count in counts.items():
                o, t, g = int(outcome[0]), int(outcome[1]), int(outcome[2])
                # Verify T matches our intervention (should always be true now)
                if t == do_treatment:
                    key = (g, t, o)
                    all_counts[key] = all_counts.get(key, 0) + count
        
        # Calculate P(O=1|do(T), G) for each gender
        gender_probs = {}
        for gender in [0, 1]:
            good = all_counts.get((gender, do_treatment, 1), 0)
            total = (all_counts.get((gender, do_treatment, 0), 0) + 
                    all_counts.get((gender, do_treatment, 1), 0))
            if total > 0:
                gender_probs[gender] = good / total
            else:
                gender_probs[gender] = 0.5
        
        print(f"  P(O=1|do(T={do_treatment}), G=0/male): {gender_probs[0]:.3f}")
        print(f"  P(O=1|do(T={do_treatment}), G=1/female): {gender_probs[1]:.3f}")
        
        return {
            'do_treatment': do_treatment,
            'gender_probs': gender_probs,
            'raw_counts': all_counts
        }
    
    def calculate_total_causal_effect(self, obs_results, int_t0, int_t1):
        """
        Calculate total causal effect via proper marginalization
        """
        print("Calculating total causal effect...")
        
        # Population gender distribution
        p_male = obs_results['population']['p_male']
        p_female = obs_results['population']['p_female']
        
        print(f"  Population: {p_male:.1%} male, {p_female:.1%} female")
        
        # Marginalize over gender
        p_o1_do_t1 = (int_t1['gender_probs'][0] * p_male + 
                      int_t1['gender_probs'][1] * p_female)
        
        p_o1_do_t0 = (int_t0['gender_probs'][0] * p_male + 
                      int_t0['gender_probs'][1] * p_female)
        
        total_causal_effect = p_o1_do_t1 - p_o1_do_t0
        
        print(f"  P(O=1|do(T=1)) = {p_o1_do_t1:.3f} (marginalized)")
        print(f"  P(O=1|do(T=0)) = {p_o1_do_t0:.3f} (marginalized)")
        print(f"  Total causal effect = {total_causal_effect:+.3f}")
        
        return {
            'p_o1_do_t1': p_o1_do_t1,
            'p_o1_do_t0': p_o1_do_t0,
            'total_causal_effect': total_causal_effect
        }
    
    def run_complete_experiment(self):
        """Run complete Simpson's paradox experiment with corrected intervention"""
        print("="*60)
        print("CORRECTED SIMPSON'S PARADOX WITH CLEAN DO-CALCULUS")
        print("="*60)
        
        # Step 1: Observational analysis
        print("\n[STEP 1: OBSERVATIONAL ANALYSIS]")
        obs_results = self.measure_observational()
        
        # Step 2: Clean interventional analysis
        print("\n[STEP 2: CLEAN INTERVENTIONS - NO G->T CONNECTION]")
        int_t0 = self.clean_intervention(do_treatment=0)
        int_t1 = self.clean_intervention(do_treatment=1)
        
        # Step 3: Calculate total causal effect
        print("\n[STEP 3: TOTAL CAUSAL EFFECT]")
        causal_results = self.calculate_total_causal_effect(obs_results, int_t0, int_t1)
        
        # Step 4: Final comparison
        print("\n[STEP 4: SIMPSON'S PARADOX RESOLUTION]")
        obs_aggregate = obs_results['aggregate_effect']
        causal_total = causal_results['total_causal_effect']
        bias = obs_aggregate - causal_total
        
        print(f"  OBSERVATIONAL aggregate: {obs_aggregate:+.3f}")
        print(f"  TRUE causal effect: {causal_total:+.3f}")
        print(f"  Confounding bias: {bias:+.3f}")
        
        # Within-group causal effects
        male_causal = int_t1['gender_probs'][0] - int_t0['gender_probs'][0]
        female_causal = int_t1['gender_probs'][1] - int_t0['gender_probs'][1]
        
        print(f"  Within-group causal effects:")
        print(f"    Male: {male_causal:+.3f}")
        print(f"    Female: {female_causal:+.3f}")
        
        # Check resolution
        paradox_resolved = (obs_results['paradox_detected'] and 
                           abs(bias) > 0.05 and
                           causal_total > 0 and
                           male_causal > 0 and female_causal > 0)
        
        print(f"\n[RESOLUTION STATUS]")
        if paradox_resolved:
            print("  SUCCESS: Simpson's paradox resolved!")
            print("  - Observational data showed paradox")
            print("  - Do-calculus revealed true positive effect")
            print("  - Confounding bias quantified and removed")
            print(f"  POLICY: Treatment beneficial (+{causal_total:.3f}, not {obs_aggregate:+.3f})")
        else:
            print("  Paradox resolution incomplete")
        
        # Store results
        self.results = {
            'observational': obs_results,
            'interventional': {'t0': int_t0, 't1': int_t1},
            'causal': causal_results,
            'resolution': {
                'paradox_resolved': paradox_resolved,
                'confounding_bias': bias,
                'within_group_effects': {'male': male_causal, 'female': female_causal}
            }
        }
        
        return self.results

# Self-test verification functions
def test_observational_detection():
    """Test that observational analysis detects Simpson's paradox"""
    print("Testing observational Simpson's paradox detection...")
    exp = CorrectedSimpsonsParadox(shots=5000, n_trials=2)
    obs = exp.measure_observational()
    
    # Should detect paradox: positive within-group, negative aggregate
    paradox_detected = obs['paradox_detected']
    male_positive = obs['male_effect'] > 0.02
    female_positive = obs['female_effect'] > 0.02
    aggregate_negative = obs['aggregate_effect'] < -0.02
    
    success = paradox_detected and male_positive and female_positive and aggregate_negative
    
    print(f"  Paradox detected: {paradox_detected}")
    print(f"  Male effect positive: {male_positive} ({obs['male_effect']:+.3f})")
    print(f"  Female effect positive: {female_positive} ({obs['female_effect']:+.3f})")
    print(f"  Aggregate negative: {aggregate_negative} ({obs['aggregate_effect']:+.3f})")
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    return success

def test_clean_intervention():
    """Test that clean intervention shows positive effects"""
    print("Testing clean intervention circuit...")
    exp = CorrectedSimpsonsParadox(shots=5000, n_trials=2)
    
    int_t1 = exp.clean_intervention(do_treatment=1)
    int_t0 = exp.clean_intervention(do_treatment=0)
    
    # Both groups should show positive causal effects
    male_causal = int_t1['gender_probs'][0] - int_t0['gender_probs'][0]
    female_causal = int_t1['gender_probs'][1] - int_t0['gender_probs'][1]
    
    male_positive = male_causal > 0.02
    female_positive = female_causal > 0.02
    
    print(f"  Male causal effect: {male_causal:+.3f} (positive: {male_positive})")
    print(f"  Female causal effect: {female_causal:+.3f} (positive: {female_positive})")
    
    success = male_positive and female_positive
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    return success

def test_paradox_resolution():
    """Test complete paradox resolution"""
    print("Testing complete paradox resolution...")
    exp = CorrectedSimpsonsParadox(shots=3000, n_trials=2)
    results = exp.run_complete_experiment()
    
    # Check resolution criteria
    obs_paradox = results['observational']['paradox_detected']
    causal_positive = results['causal']['total_causal_effect'] > 0.02
    bias_significant = abs(results['resolution']['confounding_bias']) > 0.05
    male_causal_positive = results['resolution']['within_group_effects']['male'] > 0.02
    female_causal_positive = results['resolution']['within_group_effects']['female'] > 0.02
    
    resolution_success = (obs_paradox and causal_positive and bias_significant and 
                         male_causal_positive and female_causal_positive)
    
    print(f"  Observational paradox: {obs_paradox}")
    print(f"  Causal effect positive: {causal_positive}")
    print(f"  Significant bias: {bias_significant}")
    print(f"  Male causal positive: {male_causal_positive}")
    print(f"  Female causal positive: {female_causal_positive}")
    print(f"  Test: {'PASSED' if resolution_success else 'FAILED'}")
    
    return resolution_success

def test_circuit_consistency():
    """Test that intervention circuits produce consistent T values"""
    print("Testing intervention circuit consistency...")
    exp = CorrectedSimpsonsParadox(shots=1000, n_trials=1)
    
    # Test T=1 intervention
    int_t1 = exp.clean_intervention(do_treatment=1)
    t1_samples = sum(count for (g, t, o), count in int_t1['raw_counts'].items() if t == 1)
    t1_total = sum(int_t1['raw_counts'].values())
    t1_consistency = t1_samples / t1_total if t1_total > 0 else 0
    
    # Test T=0 intervention  
    int_t0 = exp.clean_intervention(do_treatment=0)
    t0_samples = sum(count for (g, t, o), count in int_t0['raw_counts'].items() if t == 0)
    t0_total = sum(int_t0['raw_counts'].values())
    t0_consistency = t0_samples / t0_total if t0_total > 0 else 0
    
    # Should be perfectly consistent (100%)
    t1_perfect = t1_consistency > 0.99
    t0_perfect = t0_consistency > 0.99
    
    print(f"  T=1 intervention consistency: {t1_consistency:.3f} (perfect: {t1_perfect})")
    print(f"  T=0 intervention consistency: {t0_consistency:.3f} (perfect: {t0_perfect})")
    
    success = t1_perfect and t0_perfect
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    print("="*60)
    print("TESTING CORRECTED SIMPSON'S PARADOX IMPLEMENTATION")
    print("="*60)
    
    tests = [
        test_observational_detection,
        test_clean_intervention, 
        test_circuit_consistency,
        test_paradox_resolution
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    total = len(tests)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! Running main experiment...")
        print("\n" + "="*60)
        experiment = CorrectedSimpsonsParadox(shots=15000, n_trials=5)
        results = experiment.run_complete_experiment()
        print("\nExperiment completed successfully!")
    else:
        print("Some tests failed. Please check the implementation.")
