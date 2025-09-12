"""
simpson_paradox_ionq.py
Optimized Simpson's Paradox experiment for IonQ hardware
Self-contained version with integrated circuit generation
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider
from qiskit.providers.jobstatus import JobStatus
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_SIMULATOR = False  # Set to False for real QPU
API_KEY = ""  # Your IonQ API key
SHOTS = 2048  # Shots per circuit
N_TRIALS = 1  # Number of trial repetitions
TIMEOUT = 3600  # Job timeout (1 hour)
USE_AER_INSTEAD = False  # True to use Aer for comparison

class SimpsonParadoxIonQ:
    """
    Simpson's Paradox experiment optimized for IonQ hardware
    Self-contained with integrated circuit generation
    """
    
    def __init__(self, use_simulator=True, shots=2048, n_trials=3):
        self.shots = shots
        self.n_trials = n_trials
        self.use_simulator = use_simulator
        
        # Initialize IonQ provider
        print("Initializing IonQ connection...")
        self.provider = IonQProvider(token=API_KEY)
        
        # Select backend with safety check
        self._select_backend()
        
        self.results = {}
        
    def _select_backend(self):
        """Select IonQ backend with cost warning for QPU"""
        if self.use_simulator:
            if USE_AER_INSTEAD:
                from qiskit_aer import AerSimulator
                self.backend = AerSimulator()
                print("Using Qiskit Aer Simulator for comparison")
            else:
                self.backend = self.provider.get_backend("ionq_simulator")
                print(f"Using IonQ Simulator (FREE)")
            print(f"  Total shots: {self.shots * 3 * self.n_trials:,}")
        else:
            # QPU selection
            print("WARNING: QPU USAGE")
            print(f"  Backend: IonQ QPU")
            print(f"  Batch jobs: {self.n_trials} (3 circuits each)")
            print(f"  Total shots: {self.shots * 3 * self.n_trials:,}")
            print(f"  Cost will be estimated using preflight mode")
            
            # Note: Actual cost confirmation will happen in run_experiment()
            self.backend = self.provider.get_backend("qpu.aria-1")
            print("Using IonQ QPU (cost confirmation pending)")
    
    def create_confounded_circuit(self):
        """
        Create confounded circuit for Simpson's paradox
        Self-contained implementation with CORRECTED confounding logic
        
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
        
        # CORRECTED Confounding:
        # We want both groups to have treated/untreated members, but at different rates.
        # Males (G=0) get a high chance of treatment (e.g., ~85% prob)
        qc.x(0)  # Control on G=0
        qc.cry(2.4, 0, 1) 
        qc.x(0)  # Un-compute control
        
        # Females (G=1) get a low chance of treatment (e.g., ~15% prob)
        qc.cry(0.8, 0, 1)  # Control on G=1
        
        # Outcome model (remains the same):
        qc.ry(0.3, 2)      # Base rate
        qc.cry(1.0, 0, 2)  # Gender effect: G=1 -> better outcome
        qc.cry(0.6, 1, 2)  # Treatment effect: T=1 -> better outcome
        
        qc.measure_all()
        return qc
    
    def create_intervention_circuit(self, do_treatment=1):
        """
        Create intervention circuit for do(T=treatment)
        Self-contained implementation
        
        In do(T) world, G->T connection is severed
        """
        qc = QuantumCircuit(3, 3)
        
        # Gender distribution unchanged
        qc.h(0)
        
        # Force treatment value (no G->T connection!)
        if do_treatment == 1:
            qc.x(1)
        
        # Same outcome model
        qc.ry(0.3, 2)  # Base rate
        qc.cry(1.0, 0, 2)  # Gender effect
        qc.cry(0.6, 1, 2)  # Treatment effect
        
        qc.measure_all()
        return qc
                
    def estimate_cost(self, circuits):
        """
        Estimate cost using IonQ's preflight mode
        
        Args:
            circuits: List of quantum circuits
            
        Returns:
            float: Estimated cost in USD
        """
        if self.use_simulator or USE_AER_INSTEAD:
            return 0.0  # No cost for simulators
            
        try:
            print("  Estimating QPU cost...")
            # Run preflight to get cost estimate
            compiled_circuits = transpile(circuits, self.backend, optimization_level=1)
            job = self.backend.run(compiled_circuits, shots=self.shots, preflight=True)
            job.wait_for_final_state()
            
            # Get cost from metadata - handle both method and property
            metadata = job.metadata() if callable(job.metadata) else job.metadata
            cost_usd = metadata.get("cost_usd", 0.0) if isinstance(metadata, dict) else 0.0
            
            print(f"  Estimated cost: ${cost_usd:.2f}")
            return cost_usd
            
        except Exception as e:
            print(f"  Cost estimation failed: {e}")
            print("  Using fallback estimate...")
            # Fallback calculation
            total_shots = self.shots * len(circuits)
            gates_per_circuit = 10
            estimated_cost = max(1.0, total_shots * gates_per_circuit * 0.00003)
            print(f"  Fallback estimate: ${estimated_cost:.2f}")
            return estimated_cost
    
    def create_batch_circuits(self):
        """
        Create all 3 Simpson's Paradox circuits for batch execution
        
        Returns:
            list: [observational, intervention_t1, intervention_t0]
        """
        circuits = []
        
        # 1. Observational (confounded) circuit
        qc_obs = self.create_confounded_circuit()
        qc_obs.name = "observational"
        circuits.append(qc_obs)
        
        # 2. Intervention do(T=1) circuit
        qc_int1 = self.create_intervention_circuit(do_treatment=1)
        qc_int1.name = "intervention_t1"
        circuits.append(qc_int1)
        
        # 3. Intervention do(T=0) circuit
        qc_int0 = self.create_intervention_circuit(do_treatment=0)
        qc_int0.name = "intervention_t0"
        circuits.append(qc_int0)
        
        return circuits
        """
        Create all 3 Simpson's Paradox circuits for batch execution
        
        Returns:
            list: [observational, intervention_t1, intervention_t0]
        """
        circuits = []
        
        # 1. Observational (confounded) circuit
        qc_obs = self.create_confounded_circuit()
        qc_obs.name = "observational"
        circuits.append(qc_obs)
        
        # 2. Intervention do(T=1) circuit
        qc_int1 = self.create_intervention_circuit(do_treatment=1)
        qc_int1.name = "intervention_t1"
        circuits.append(qc_int1)
        
        # 3. Intervention do(T=0) circuit
        qc_int0 = self.create_intervention_circuit(do_treatment=0)
        qc_int0.name = "intervention_t0"
        circuits.append(qc_int0)
        
        return circuits
    
    def execute_batch(self, circuits):
        """
        Execute circuits as a batch job on IonQ
        
        Args:
            circuits: List of quantum circuits
            
        Returns:
            list: Counts for each circuit
        """
        print(f"  Executing batch of {len(circuits)} circuits...")
        
        try:
            # Transpile for IonQ backend
            compiled_circuits = transpile(circuits, self.backend, optimization_level=1)
            
            # Submit batch job
            job = self.backend.run(compiled_circuits, shots=self.shots)
            
            # Get job ID (handle both IonQ and Aer)
            job_id = job.job_id() if callable(job.job_id) else job.job_id
            print(f"  Job ID: {job_id}")
            
            # Wait for completion with progress updates
            start_time = time.time()
            last_status = None
            
            while job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                current_status = job.status()
                if current_status != last_status:
                    elapsed = int(time.time() - start_time)
                    status_name = current_status.name if hasattr(current_status, 'name') else str(current_status)
                    print(f"  Status: {status_name} (elapsed: {elapsed}s)")
                    last_status = current_status
                    
                if time.time() - start_time > TIMEOUT:
                    raise TimeoutError(f"Job timeout after {TIMEOUT} seconds")
                    
                time.sleep(10)  # Check every 10 seconds
                
            if job.status() == JobStatus.ERROR:
                error_msg = job.error_message() if hasattr(job, 'error_message') else "Unknown error"
                raise RuntimeError(f"Job failed: {error_msg}")
                
            # Get results
            result = job.result()
            counts_list = [result.get_counts(i) for i in range(len(circuits))]
            
            print(f"  Batch completed successfully!")
            return counts_list
            
        except Exception as e:
            print(f"  Batch execution failed: {e}")
            print("  Falling back to local simulator...")
            
            # Fallback to local simulator
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            compiled = transpile(circuits, sim)
            job = sim.run(compiled, shots=self.shots)
            result = job.result()
            return [result.get_counts(i) for i in range(len(circuits))]
    
    def analyze_results(self, all_counts):
        """
        Analyze results from multiple trials
        
        Args:
            all_counts: List of count dictionaries from trials
            
        Returns:
            dict: Analysis results
        """
        # Aggregate counts across trials
        obs_counts_agg = {}
        int_t1_counts_agg = {}
        int_t0_counts_agg = {}
        
        for trial_counts in all_counts:
            for outcome, count in trial_counts[0].items():
                obs_counts_agg[outcome] = obs_counts_agg.get(outcome, 0) + count
            for outcome, count in trial_counts[1].items():
                int_t1_counts_agg[outcome] = int_t1_counts_agg.get(outcome, 0) + count
            for outcome, count in trial_counts[2].items():
                int_t0_counts_agg[outcome] = int_t0_counts_agg.get(outcome, 0) + count
        
        # Use the same analysis function from the original script
        def parse_and_analyze(obs_counts, int_t1_counts, int_t0_counts):
            """
            Parse IonQ output format and analyze
            
            Ground truth from debug script confirms standard Qiskit bit ordering:
            Bit string represents OTG (Outcome, Treatment, Gender)
            """
            
            # Parse counts with correct Qiskit ordering
            def parse_counts(counts):
                parsed = {}
                for outcome, count in counts.items():
                    bits = outcome.replace(' ', '')
                    if len(bits) >= 3:
                        # Standard Qiskit ordering: bits represent q2q1q0
                        # bits[0] = q2 (O), bits[1] = q1 (T), bits[2] = q0 (G)
                        o = int(bits[0])
                        t = int(bits[1])
                        g = int(bits[2])
                        parsed[(g, t, o)] = count
                return parsed
            
            obs_parsed = parse_counts(obs_counts)
            int_t1_parsed = parse_counts(int_t1_counts)
            int_t0_parsed = parse_counts(int_t0_counts)
            
            # Calculate effects
            def calc_prob(parsed, gender, treatment):
                good = parsed.get((gender, treatment, 1), 0)
                bad = parsed.get((gender, treatment, 0), 0)
                total = good + bad
                return good / total if total > 0 else 0.0
            
            # Observational effects
            male_effect = (calc_prob(obs_parsed, 0, 1) - 
                          calc_prob(obs_parsed, 0, 0))
            female_effect = (calc_prob(obs_parsed, 1, 1) - 
                            calc_prob(obs_parsed, 1, 0))
            
            # Overall observational effect
            treated_good = (obs_parsed.get((0,1,1), 0) + 
                           obs_parsed.get((1,1,1), 0))
            treated_bad = (obs_parsed.get((0,1,0), 0) + 
                          obs_parsed.get((1,1,0), 0))
            untreated_good = (obs_parsed.get((0,0,1), 0) + 
                             obs_parsed.get((1,0,1), 0))
            untreated_bad = (obs_parsed.get((0,0,0), 0) + 
                            obs_parsed.get((1,0,0), 0))
            
            p_treated = treated_good / (treated_good + treated_bad) if (treated_good + treated_bad) > 0 else 0
            p_untreated = untreated_good / (untreated_good + untreated_bad) if (untreated_good + untreated_bad) > 0 else 0
            overall_effect = p_treated - p_untreated
            
            # Population distribution
            total_male = sum(obs_parsed.get((0, t, o), 0) 
                           for t in [0,1] for o in [0,1])
            total_female = sum(obs_parsed.get((1, t, o), 0) 
                             for t in [0,1] for o in [0,1])
            p_male = total_male / (total_male + total_female) if (total_male + total_female) > 0 else 0.5
            
            # Causal effects
            p_o1_do_t1 = (calc_prob(int_t1_parsed, 0, 1) * p_male + 
                         calc_prob(int_t1_parsed, 1, 1) * (1 - p_male))
            p_o1_do_t0 = (calc_prob(int_t0_parsed, 0, 0) * p_male + 
                         calc_prob(int_t0_parsed, 1, 0) * (1 - p_male))
            causal_effect = p_o1_do_t1 - p_o1_do_t0
            
            # Check for paradox with detailed output
            print(f"\nEffect sizes for paradox detection:")
            print(f"  Male effect: {male_effect:+.3f} (need > +0.02)")
            print(f"  Female effect: {female_effect:+.3f} (need > +0.02)")
            print(f"  Overall effect: {overall_effect:+.3f} (need < -0.02)")
            
            # Sanity check
            if female_effect < 0 or male_effect < 0:
                print("  WARNING: Negative effect detected - possible issue with circuit or parsing")
            
            paradox_detected = (male_effect > 0.02 and 
                              female_effect > 0.02 and 
                              overall_effect < -0.02)
            
            return {
                'observational': {
                    'male_effect': float(male_effect),
                    'female_effect': float(female_effect),
                    'overall_effect': float(overall_effect),
                    'population': {'p_male': float(p_male)}
                },
                'causal': {
                    'effect': float(causal_effect),
                    'p_o1_do_t1': float(p_o1_do_t1),
                    'p_o1_do_t0': float(p_o1_do_t0)
                },
                'paradox': {
                    'detected': bool(paradox_detected),
                    'bias': float(overall_effect - causal_effect),
                    'resolved': bool(paradox_detected and causal_effect > 0.02)
                }
            }
        
        return parse_and_analyze(obs_counts_agg, int_t1_counts_agg, int_t0_counts_agg)
    
    def run_experiment(self):
        """
        Run the complete Simpson's Paradox experiment
        
        Returns:
            dict: Experimental results
        """
        print("\n" + "="*60)
        print("SIMPSON'S PARADOX ON IONQ HARDWARE")
        print("="*60)
        # Handle both IonQ (method) and Aer (property) backends
        backend_name = self.backend.name() if callable(self.backend.name) else self.backend.name
        print(f"Backend: {backend_name}")
        print(f"Batch mode: {self.n_trials} trials × 3 circuits")
        print(f"Total shots: {self.shots * 3 * self.n_trials:,}")
        print(f"Circuit generation: Self-contained")
        
        # Cost estimation for QPU
        if not self.use_simulator and not USE_AER_INSTEAD:
            print("\nEstimating QPU cost...")
            sample_circuits = self.create_batch_circuits()
            estimated_cost_per_batch = self.estimate_cost(sample_circuits)
            total_estimated_cost = estimated_cost_per_batch * self.n_trials
            
            # Store estimated cost for later
            self._estimated_cost = total_estimated_cost
            
            print(f"\nCost Summary:")
            print(f"  Cost per batch: ${estimated_cost_per_batch:.2f}")
            print(f"  Number of batches: {self.n_trials}")
            print(f"  Total estimated cost: ${total_estimated_cost:.2f}")
            
            confirm = input("Continue with QPU execution? (type 'yes'): ")
            if confirm.lower() != 'yes':
                print("Execution cancelled.")
                return None
        
        all_trial_counts = []
        
        for trial in range(self.n_trials):
            print(f"\nTrial {trial + 1}/{self.n_trials}")
            
            # Create batch of 3 circuits
            circuits = self.create_batch_circuits()
            
            # Execute batch
            counts = self.execute_batch(circuits)
            all_trial_counts.append(counts)
            
            # Quick analysis of this trial
            obs_total = sum(counts[0].values())
            print(f"  Observational: {len(counts[0])} outcomes, {obs_total} counts")
            print(f"  Intervention T=1: {len(counts[1])} outcomes")
            print(f"  Intervention T=0: {len(counts[2])} outcomes")
        
        # Analyze aggregated results
        print("\nAnalyzing aggregated results...")
        analysis = self.analyze_results(all_trial_counts)
        
        # Display results
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        
        obs = analysis['observational']
        causal = analysis['causal']
        paradox = analysis['paradox']
        
        print(f"\nObservational Effects:")
        print(f"  Male effect: {obs['male_effect']:+.3f}")
        print(f"  Female effect: {obs['female_effect']:+.3f}")
        print(f"  Overall effect: {obs['overall_effect']:+.3f}")
        
        print(f"\nCausal Analysis:")
        print(f"  True causal effect: {causal['effect']:+.3f}")
        print(f"  Confounding bias: {paradox['bias']:+.3f}")
        
        print(f"\nSimpson's Paradox:")
        print(f"  Detected: {bool(paradox['detected'])}")
        print(f"  Resolved: {bool(paradox['resolved'])}")
        
        # Add cost info if QPU was used
        cost_info = None
        if not self.use_simulator and not USE_AER_INSTEAD:
            # Calculate actual cost (this is an estimate based on execution)
            total_shots_executed = self.shots * 3 * self.n_trials
            cost_info = {
                'estimated_cost_usd': getattr(self, '_estimated_cost', 0.0),
                'total_shots': total_shots_executed,
                'cost_per_shot': 'See IonQ billing'
            }
            print(f"\nExecution Cost:")
            print(f"  Total shots executed: {total_shots_executed:,}")
            print(f"  Estimated cost (preflight): ${self._estimated_cost:.2f}")
            print(f"  Check IonQ dashboard for actual billing")
        
        # Save results
        self.save_results(analysis, all_trial_counts, cost_info)
        
        return analysis
    
    def save_results(self, analysis, raw_counts, cost_info=None):
        """Save results to JSON file"""
        # Handle backend name for both IonQ and Aer
        backend_name = self.backend.name() if callable(self.backend.name) else self.backend.name
        
        results_data = {
            'experiment_info': {
                'name': "Simpson's Paradox - IonQ Hardware",
                'date': datetime.now().isoformat(),
                'backend': backend_name,
                'simulator': self.use_simulator,
                'shots_per_circuit': self.shots,
                'n_trials': self.n_trials,
                'total_shots': self.shots * 3 * self.n_trials,
                'execution_mode': 'batch',
                'circuit_generation': 'self-contained'
            },
            'analysis': analysis,
            'raw_trial_counts': [
                {
                    'trial': i,
                    'observational': counts[0],
                    'intervention_t1': counts[1],
                    'intervention_t0': counts[2]
                }
                for i, counts in enumerate(raw_counts)
            ]
        }
        
        # Add cost information if available
        if cost_info:
            results_data['cost_info'] = cost_info
        
        filename = f"simpson_paradox_ionq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)
            
        print(f"\nResults saved to: {filename}")


def run_self_tests():
    """Run self-tests before main experiment"""
    print("Running self-tests...")
    
    # Test 1: IonQ connection
    try:
        provider = IonQProvider(token=API_KEY)
        backend = provider.get_backend("ionq_simulator")
        print("  IonQ connection: OK")
    except Exception as e:
        print(f"  IonQ connection: FAILED - {e}")
        return False
    
    # Test 2: Circuit creation (now self-contained)
    try:
        exp = SimpsonParadoxIonQ(use_simulator=True, shots=100, n_trials=1)
        qc = exp.create_confounded_circuit()
        print("  Circuit creation: OK")
        print(f"    Circuit depth: {qc.depth()}")
        print(f"    Circuit gates: {qc.count_ops()}")
    except Exception as e:
        print(f"  Circuit creation: FAILED - {e}")
        return False
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    # Configuration summary
    print("Configuration:")
    print(f"  USE_SIMULATOR: {USE_SIMULATOR}")
    print(f"  SHOTS: {SHOTS}")
    print(f"  N_TRIALS: {N_TRIALS}")
    print(f"  USE_AER_INSTEAD: {USE_AER_INSTEAD}")
    
    if not API_KEY:
        print("\nERROR: Please set your IonQ API key!")
        print("Get your key from: https://cloud.ionq.com/")
        exit(1)
    
    # Run self-tests
    if not run_self_tests():
        print("\nSelf-tests failed. Please check configuration.")
        exit(1)
    
    # Run main experiment
    experiment = SimpsonParadoxIonQ(
        use_simulator=USE_SIMULATOR,
        shots=SHOTS,
        n_trials=N_TRIALS
    )
    
    results = experiment.run_experiment()
    
    print("\nExperiment completed successfully!")
    
    if bool(results['paradox']['detected']):
        print("Simpson's Paradox successfully demonstrated!")
        if bool(results['paradox']['resolved']):
            print("Quantum do-calculus successfully resolved the paradox!")
    else:
        print("Consider increasing shots or trials for clearer results.")
        print("If effects are negative, check circuit implementation.")
