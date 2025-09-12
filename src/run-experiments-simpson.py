"""
Quantum Simpson's Paradox Experiment Runner
==========================================
This script runs multiple trials of the Simpson's paradox experiment
and generates structured JSON output with statistical analysis.
"""

import json
import numpy as np
from datetime import datetime
from scipy.stats import norm, t
from math import sqrt
import qiskit

# Import the original Simpson's paradox implementation
try:
    from simpson_do import CorrectedSimpsonsParadox
except ImportError:
    print("Error: Cannot import CorrectedSimpsonsParadox from simpson_do.py")
    print("Make sure simpson_do.py is in the same directory")
    exit(1)


class ExperimentRunner:
    """
    Runs multiple trials of Simpson's paradox experiment and generates
    structured JSON output with confidence intervals
    """
    
    def __init__(self, n_trials=30, shots_per_trial=15000):
        self.n_trials = n_trials
        self.shots_per_trial = shots_per_trial
        self.trial_data = []
        
    def calculate_confidence_interval(self, values, confidence=0.95):
        """
        Calculate confidence interval for a list of values
        """
        if not values or len(values) == 0:
            return {"mean": 0.0, "ci_95": [0.0, 0.0]}
        
        # Convert to numpy array for easier calculation
        values = np.array(values)
        n = len(values)
        mean_val = np.mean(values)
        std_err = np.std(values, ddof=1) / sqrt(n)
        
        # Use t-distribution for small samples, normal for large samples
        alpha = 1 - confidence
        if n > 30:
            critical_value = norm.ppf(1 - alpha/2)
        else:
            critical_value = t.ppf(1 - alpha/2, df=n-1)
        
        margin_error = critical_value * std_err
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        return {
            "mean": round(float(mean_val), 3),
            "ci_95": [round(float(ci_lower), 3), round(float(ci_upper), 3)]
        }
    
    def extract_probabilities_from_counts(self, raw_counts):
        """
        Extract conditional probabilities from raw count data
        Returns dictionary with probabilities for each gender-treatment combination
        """
        probs = {}
        
        # Calculate P(O=1|G,T) for each combination
        for gender in [0, 1]:
            for treatment in [0, 1]:
                good_outcomes = raw_counts.get((gender, treatment, 1), 0)
                total_outcomes = (raw_counts.get((gender, treatment, 0), 0) + 
                                raw_counts.get((gender, treatment, 1), 0))
                
                if total_outcomes > 0:
                    probs[(gender, treatment)] = good_outcomes / total_outcomes
                else:
                    probs[(gender, treatment)] = 0.0
        
        # Calculate overall P(O=1|T) (marginalizing over gender)
        for treatment in [0, 1]:
            good_total = sum(raw_counts.get((g, treatment, 1), 0) for g in [0, 1])
            outcome_total = sum(raw_counts.get((g, treatment, o), 0) 
                              for g in [0, 1] for o in [0, 1])
            
            if outcome_total > 0:
                probs[('overall', treatment)] = good_total / outcome_total
            else:
                probs[('overall', treatment)] = 0.0
        
        return probs
    
    def run_single_trial(self):
        """
        Run a single trial and extract all necessary probability values
        Uses the exact same logic as measure_observational() to ensure consistency
        """
        # Create experiment instance for this trial
        experiment = CorrectedSimpsonsParadox(shots=self.shots_per_trial, n_trials=1)
        
        # Run observational analysis
        obs_results = experiment.measure_observational()
        raw_counts = obs_results['raw_counts']
        
        # Use the exact same logic as measure_observational() to calculate probabilities
        probabilities = {}
        for gender in [0, 1]:
            for treatment in [0, 1]:
                good = raw_counts.get((gender, treatment, 1), 0)
                total = (raw_counts.get((gender, treatment, 0), 0) + 
                        raw_counts.get((gender, treatment, 1), 0))
                if total > 0:
                    probabilities[(gender, treatment)] = good / total
                else:
                    probabilities[(gender, treatment)] = 0.5
        
        # Calculate overall probabilities using the same aggregation logic
        total_t1_good = sum(raw_counts.get((g, 1, 1), 0) for g in [0, 1])
        total_t1 = sum(raw_counts.get((g, 1, o), 0) for g in [0, 1] for o in [0, 1])
        total_t0_good = sum(raw_counts.get((g, 0, 1), 0) for g in [0, 1])
        total_t0 = sum(raw_counts.get((g, 0, o), 0) for g in [0, 1] for o in [0, 1])
        
        overall_treated_prob = total_t1_good / total_t1 if total_t1 > 0 else 0.5
        overall_untreated_prob = total_t0_good / total_t0 if total_t0 > 0 else 0.5
        
        # Extract individual probabilities
        male_untreated_prob = probabilities[(0, 0)]
        male_treated_prob = probabilities[(0, 1)]
        female_untreated_prob = probabilities[(1, 0)]
        female_treated_prob = probabilities[(1, 1)]
        
        # Run causal interventions
        int_t0_results = experiment.clean_intervention(do_treatment=0)
        int_t1_results = experiment.clean_intervention(do_treatment=1)
        
        # Extract population gender distribution from observational data
        total_population = sum(raw_counts.values())
        male_population = sum(raw_counts.get((0, t, o), 0) for t in [0, 1] for o in [0, 1])
        female_population = sum(raw_counts.get((1, t, o), 0) for t in [0, 1] for o in [0, 1])
        
        p_male = male_population / total_population if total_population > 0 else 0.5
        p_female = female_population / total_population if total_population > 0 else 0.5
        
        # Calculate causal probabilities using gender-specific probabilities and population weights
        causal_prob_t0 = (int_t0_results['gender_probs'][0] * p_male + 
                          int_t0_results['gender_probs'][1] * p_female)
        
        causal_prob_t1 = (int_t1_results['gender_probs'][0] * p_male + 
                          int_t1_results['gender_probs'][1] * p_female)
        
        # Verify our calculations match the trial output (for debugging)
        male_effect_calc = male_treated_prob - male_untreated_prob
        female_effect_calc = female_treated_prob - female_untreated_prob
        overall_effect_calc = overall_treated_prob - overall_untreated_prob
        
        return {
            'observational': {
                'male_untreated': male_untreated_prob,
                'male_treated': male_treated_prob,
                'female_untreated': female_untreated_prob,
                'female_treated': female_treated_prob,
                'overall_untreated': overall_untreated_prob,
                'overall_treated': overall_treated_prob
            },
            'causal': {
                'prob_do_t0': causal_prob_t0,
                'prob_do_t1': causal_prob_t1
            },
            'verification': {  # For debugging - can be removed later
                'male_effect_calc': male_effect_calc,
                'female_effect_calc': female_effect_calc,
                'overall_effect_calc': overall_effect_calc,
                'male_effect_reported': obs_results['male_effect'],
                'female_effect_reported': obs_results['female_effect'],
                'aggregate_effect_reported': obs_results['aggregate_effect']
            }
        }
    
    def run_experiment(self):
        """
        Run the complete experiment with multiple trials
        """
        print("="*60)
        print("QUANTUM SIMPSON'S PARADOX EXPERIMENT")
        print("="*60)
        print(f"Running {self.n_trials} trials with {self.shots_per_trial} shots each...")
        
        # Collect data from all trials
        trial_results = []
        
        for trial in range(self.n_trials):
            if (trial + 1) % 5 == 0 or trial == 0:
                print(f"  Completed trial {trial + 1}/{self.n_trials}")
            
            try:
                trial_data = self.run_single_trial()
                trial_results.append(trial_data)
            except Exception as e:
                print(f"  Warning: Trial {trial + 1} failed: {e}")
                continue
        
        if not trial_results:
            raise RuntimeError("All trials failed. Check the implementation.")
        
        print(f"Successfully completed {len(trial_results)} trials")
        print("Calculating statistics...")
        
        # Extract data series for statistical analysis
        male_untreated = [t['observational']['male_untreated'] for t in trial_results]
        male_treated = [t['observational']['male_treated'] for t in trial_results]
        female_untreated = [t['observational']['female_untreated'] for t in trial_results]
        female_treated = [t['observational']['female_treated'] for t in trial_results]
        overall_untreated = [t['observational']['overall_untreated'] for t in trial_results]
        overall_treated = [t['observational']['overall_treated'] for t in trial_results]
        
        causal_prob_t0 = [t['causal']['prob_do_t0'] for t in trial_results]
        causal_prob_t1 = [t['causal']['prob_do_t1'] for t in trial_results]
        
        # Calculate treatment effects
        male_effects = [t - u for t, u in zip(male_treated, male_untreated)]
        female_effects = [t - u for t, u in zip(female_treated, female_untreated)]
        overall_effects = [t - u for t, u in zip(overall_treated, overall_untreated)]
        causal_effects = [t - u for t, u in zip(causal_prob_t1, causal_prob_t0)]
        
        # Build structured JSON output
        results = {
            "experiment_name": "Quantum Simpson's Paradox - 3 Qubit Model",
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                "qiskit_version": qiskit.__version__,
                "n_trials": len(trial_results),
                "shots_per_trial": self.shots_per_trial
            },
            "observational_analysis": {
                "subgroups": {
                    "male_g0": {
                        "untreated_prob_p_O1_T0": self.calculate_confidence_interval(male_untreated),
                        "treated_prob_p_O1_T1": self.calculate_confidence_interval(male_treated),
                        "effect_delta_p": self.calculate_confidence_interval(male_effects)
                    },
                    "female_g1": {
                        "untreated_prob_p_O1_T0": self.calculate_confidence_interval(female_untreated),
                        "treated_prob_p_O1_T1": self.calculate_confidence_interval(female_treated),
                        "effect_delta_p": self.calculate_confidence_interval(female_effects)
                    }
                },
                "overall": {
                    "untreated_prob_p_O1_T0": self.calculate_confidence_interval(overall_untreated),
                    "treated_prob_p_O1_T1": self.calculate_confidence_interval(overall_treated),
                    "effect_delta_p": self.calculate_confidence_interval(overall_effects)
                }
            },
            "causal_analysis": {
                "overall": {
                    "untreated_prob_p_O1_do_T0": self.calculate_confidence_interval(causal_prob_t0),
                    "treated_prob_p_O1_do_T1": self.calculate_confidence_interval(causal_prob_t1),
                    "effect_delta_p_ace": self.calculate_confidence_interval(causal_effects)
                }
            }
        }
        
        # Display summary
        print("\n[SUMMARY RESULTS]")
        print("Observational Analysis:")
        male_effect = results['observational_analysis']['subgroups']['male_g0']['effect_delta_p']['mean']
        female_effect = results['observational_analysis']['subgroups']['female_g1']['effect_delta_p']['mean']
        overall_effect = results['observational_analysis']['overall']['effect_delta_p']['mean']
        
        print(f"  Male effect: {male_effect:+.3f}")
        print(f"  Female effect: {female_effect:+.3f}")
        print(f"  Overall effect: {overall_effect:+.3f}")
        
        print("Causal Analysis:")
        causal_effect = results['causal_analysis']['overall']['effect_delta_p_ace']['mean']
        print(f"  True causal effect (ACE): {causal_effect:+.3f}")
        
        # Check for Simpson's paradox
        paradox_detected = (male_effect > 0.02 and female_effect > 0.02 and overall_effect < -0.02)
        paradox_resolved = (paradox_detected and causal_effect > 0.02)
        
        print(f"\nSimpson's Paradox:")
        print(f"  Detected: {paradox_detected}")
        print(f"  Resolved: {paradox_resolved}")
        
        return results
    
    def save_results(self, results, filename="simpson_paradox_results.json"):
        """
        Save results to JSON file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")
        return filename


# Self-test verification functions
def test_probability_extraction():
    """Test probability extraction from count data"""
    print("Testing probability extraction...")
    
    runner = ExperimentRunner(n_trials=1, shots_per_trial=1000)
    
    # Test count data (simplified example)
    test_counts = {
        (0, 0, 0): 300,  # Male, untreated, bad outcome
        (0, 0, 1): 100,  # Male, untreated, good outcome
        (0, 1, 0): 200,  # Male, treated, bad outcome  
        (0, 1, 1): 150,  # Male, treated, good outcome
        (1, 0, 0): 100,  # Female, untreated, bad outcome
        (1, 0, 1): 200,  # Female, untreated, good outcome
        (1, 1, 0): 80,   # Female, treated, bad outcome
        (1, 1, 1): 120   # Female, treated, good outcome
    }
    
    probs = runner.extract_probabilities_from_counts(test_counts)
    
    # Check calculations
    male_untreated_expected = 100 / (300 + 100)  # 0.25
    male_treated_expected = 150 / (200 + 150)    # 0.43
    female_untreated_expected = 200 / (100 + 200)  # 0.67
    female_treated_expected = 120 / (80 + 120)     # 0.60
    
    success = (abs(probs[(0, 0)] - male_untreated_expected) < 0.01 and
              abs(probs[(0, 1)] - male_treated_expected) < 0.01 and
              abs(probs[(1, 0)] - female_untreated_expected) < 0.01 and
              abs(probs[(1, 1)] - female_treated_expected) < 0.01)
    
    print(f"  Male untreated: {probs[(0, 0)]:.3f} (expected: {male_untreated_expected:.3f})")
    print(f"  Male treated: {probs[(0, 1)]:.3f} (expected: {male_treated_expected:.3f})")
    print(f"  Female untreated: {probs[(1, 0)]:.3f} (expected: {female_untreated_expected:.3f})")
    print(f"  Female treated: {probs[(1, 1)]:.3f} (expected: {female_treated_expected:.3f})")
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    return success

def test_confidence_intervals():
    """Test confidence interval calculation"""
    print("Testing confidence interval calculation...")
    
    runner = ExperimentRunner()
    
    # Test with known data
    test_values = [0.25, 0.24, 0.26, 0.25, 0.27, 0.24, 0.26]
    result = runner.calculate_confidence_interval(test_values)
    
    expected_mean = np.mean(test_values)
    has_mean = abs(result['mean'] - expected_mean) < 0.001
    has_ci = 'ci_95' in result and len(result['ci_95']) == 2
    ci_valid = result['ci_95'][0] <= result['mean'] <= result['ci_95'][1]
    
    success = has_mean and has_ci and ci_valid
    
    print(f"  Expected mean: {expected_mean:.3f}")
    print(f"  Calculated mean: {result['mean']:.3f}")
    print(f"  Confidence interval: {result['ci_95']}")
    print(f"  Mean correct: {has_mean}")
    print(f"  CI format correct: {has_ci}")
    print(f"  CI contains mean: {ci_valid}")
    print(f"  Test: {'PASSED' if success else 'FAILED'}")
    
    return success

def test_single_trial():
    """Test single trial execution"""
    print("Testing single trial execution...")
    
    runner = ExperimentRunner(n_trials=1, shots_per_trial=2000)
    
    try:
        trial_result = runner.run_single_trial()
        
        # Check structure
        has_obs = 'observational' in trial_result
        has_causal = 'causal' in trial_result
        
        # Check observational data
        obs_keys = ['male_untreated', 'male_treated', 'female_untreated', 
                   'female_treated', 'overall_untreated', 'overall_treated']
        obs_complete = all(key in trial_result['observational'] for key in obs_keys)
        
        # Check causal data
        causal_keys = ['prob_do_t0', 'prob_do_t1']
        causal_complete = all(key in trial_result['causal'] for key in causal_keys)
        
        # Check probability ranges
        all_probs = (list(trial_result['observational'].values()) + 
                    list(trial_result['causal'].values()))
        probs_valid = all(0 <= p <= 1 for p in all_probs)
        
        success = has_obs and has_causal and obs_complete and causal_complete and probs_valid
        
        print(f"  Has observational data: {has_obs}")
        print(f"  Has causal data: {has_causal}")
        print(f"  Observational complete: {obs_complete}")
        print(f"  Causal complete: {causal_complete}")
        print(f"  Probabilities valid: {probs_valid}")
        print(f"  Test: {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"  Error during trial: {e}")
        print(f"  Test: FAILED")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING EXPERIMENT RUNNER")
    print("="*60)
    
    tests = [
        test_probability_extraction,
        test_confidence_intervals,
        test_single_trial
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
        
        # Run the main experiment
        runner = ExperimentRunner(n_trials=30, shots_per_trial=15000)
        results = runner.run_experiment()
        filename = runner.save_results(results)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {filename}")
        
        # Show sample of the output
        print("\nSample of generated JSON structure:")
        sample_output = {
            "experiment_name": results["experiment_name"],
            "metadata": results["metadata"],
            "observational_analysis": {
                "subgroups": {
                    "male_g0": results["observational_analysis"]["subgroups"]["male_g0"]
                }
            }
        }
        print(json.dumps(sample_output, indent=2))
        
    else:
        print("Some tests failed. Please check the implementation.")
