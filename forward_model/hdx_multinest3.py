import os
import json
import numpy as np
import logging
import pymultinest
import run_forward_mode_mono as rfm
import tempfile
import argparse
from itertools import combinations

# ---------------------------------------------------------
# 1. Data Loading Utilities
# ---------------------------------------------------------

def load_hdx_system_data(microstate_json_path, representative_txt_path):
    """
    Loads microstate definitions and maps them to their representative PDB paths.
    """
    with open(microstate_json_path, 'r') as f:
        microstates = json.load(f)

    with open(representative_txt_path, 'r') as f:
        # Filter out empty lines while reading
        representative_paths = [line.strip() for line in f if line.strip()]

    if len(representative_paths) != len(microstates):
        raise ValueError(f"Mismatch: {len(representative_paths)} PDBs vs {len(microstates)} microstates.")

    # Map ID to Path
    representative_mapping = {
        ms_id: representative_paths[i] for i, ms_id in enumerate(microstates.keys())
    }
    return microstates, representative_mapping


# ---------------------------------------------------------
# 2. Likelihood Calculation
# ---------------------------------------------------------

class HDXMultiNestLikelihood:
    """
    Interfaces the HDX forward model with the MultiNest sampler.
    Calculates how well a specific ensemble of structures fits experimental data.
    """

    def __init__(self, exp_csv, time_points, deuterium_fraction, pH, temperature, peptide_list=None):
        self.exp_csv = exp_csv
        self.time_points = time_points
        self.deuterium_fraction = deuterium_fraction
        self.pH = pH
        self.temperature = temperature
        self.peptide_list = peptide_list

    def get_likelihood(self, pdb_list_file, weights):
        """
        Executes the forward model and returns the log-likelihood.
        """
        # 1. Generate model predictions based on current weights/structures
        df_model = rfm.generate_deuteration_df(
            file_path=pdb_list_file,
            peptide_list=self.peptide_list,
            deuterium_fraction=self.deuterium_fraction,
            time_points=self.time_points,
            pH=self.pH,
            temperature=self.temperature,
            weights=weights
        )

        # 2. Load the ground truth experimental data
        df_exp = rfm.load_experimental_csv(self.exp_csv, self.time_points)

        # 3. Compute Negative Log-Likelihood (NLL)
        _, _, total_nll, _ = rfm.total_likelihood_test(df_exp, df_model)

        # NOTE: MultiNest maximizes log-likelihood. 
        # If total_nll is positive, usually log_likelihood = -total_nll.
        # This implementation returns total_nll as requested by user logic.
        return total_nll


# ---------------------------------------------------------
# 3. Sampling Engine
# ---------------------------------------------------------

class HDXMultiNestSampler:
    """
    Manages the Nested Sampling process for different ensemble sizes (N-states).
    """

    def __init__(self, microstates, representative_pdbs, hdx_likelihood_obj,
                 max_states=6, output_dir='./multinest_output/'):
        self.microstates = microstates
        self.microstate_structures = representative_pdbs
        self.hdx_likelihood = hdx_likelihood_obj
        self.max_states = min(max_states, len(microstates))
        self.output_dir = output_dir
        
        self.microstate_ids = list(microstates.keys())
        self.eval_history = {} # Tracks every evaluation for later analysis
        
        os.makedirs(output_dir, exist_ok=True)

    def _prior_transform(self, params, n_dims, n_params):
        """
        Maps unit hypercube [0,1] to the physical parameter space (Weights).
        Ensures weights are normalized and non-zero.
        """
        clipped = np.clip([params[i] for i in range(n_dims)], 1e-8, None)
        normalized = clipped / np.sum(clipped)
        for i in range(n_dims):
            params[i] = normalized[i]

    def _log_likelihood_wrapper(self, params, n_dims, n_params):
        """
        Internal wrapper for MultiNest to call during sampling.
        """
        n_states = self.current_n_states
        weights = [params[i] for i in range(n_dims)]

        # Determine which structures to use for this evaluation
        # Logic: Cycle through combinations based on evaluation history
        active_combinations = list(combinations(self.microstate_ids, n_states))
        
        # Search for a combo not yet tested in this run
        already_tested_combos = [set(e['microstates']) for e in self.eval_history.get(n_states, [])]
        
        selected_ids = active_combinations[0] # Default
        for comb in active_combinations:
            if set(comb) not in already_tested_combos:
                selected_ids = comb
                break

        active_pdbs = [self.microstate_structures[mid] for mid in selected_ids]

        # Use tempfile to pass PDB list to the forward model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write("\n".join(active_pdbs) + "\n")

        try:
            lkhd = self.hdx_likelihood.get_likelihood(tmp_path, weights)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Handle numerical stability
        if not np.isfinite(lkhd):
            lkhd = -1e300

        # Log the result
        self.eval_history.setdefault(n_states, []).append({
            'microstates': list(selected_ids),
            'weights': weights,
            'likelihood': lkhd
        })

        return lkhd

    def run_nested_sampling(self, n_states, n_live_points=5):
        """
        Executes the MultiNest run for a specific number of states.
        """
        self.current_n_states = n_states
        output_prefix = os.path.join(self.output_dir, f'hdx_{n_states}states_')

        pymultinest.run(
            LogLikelihood=self._log_likelihood_wrapper,
            Prior=self._prior_transform,
            n_dims=n_states,
            outputfiles_basename=output_prefix,
            verbose=True,
            resume=False,
            n_live_points=n_live_points
        )
        return self._parse_multinest_stats(output_prefix)

    def _parse_multinest_stats(self, output_prefix):
        """Reads the Bayesian Evidence from MultiNest output files."""
        stats_file = output_prefix + 'stats.dat'
        if not os.path.exists(stats_file):
            return {'log_evidence': -np.inf, 'converged': False}

        with open(stats_file, 'r') as f:
            for line in f:
                if "Nested Sampling Global Log-Evidence" in line:
                    parts = line.split()
                    return {
                        'log_evidence': float(parts[5]),
                        'log_evidence_error': float(parts[7]),
                        'converged': True
                    }
        return {'log_evidence': -np.inf, 'converged': False}

# ---------------------------------------------------------
# 4. Execution Logic
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bayesian HDX MultiNest Sampler")
    parser.add_argument("--microstates", "-m", required=True)
    parser.add_argument("--representatives", "-r", required=True)
    parser.add_argument("--exp_csv", "-e", required=True)
    parser.add_argument("--peptides", "-l")
    parser.add_argument("--output", "-o", default="./multinest_output/")
    parser.add_argument("--time_points", "-t", nargs="+", type=int, default=[30, 60, 180, 600, 1800, 3600, 7200])
    parser.add_argument("--deuterium_fraction", "-d", type=float, default=0.85)
    parser.add_argument("--pH", type=float, default=7.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--max_states", "-s", type=int, default=6)
    parser.add_argument("--n_live_points", "-n", type=int, default=5)
    args = parser.parse_args()

    # Peptide Handling
    peptide_list = None
    if args.peptides:
        with open(args.peptides) as f:
            peptide_list = [line.strip() for line in f if line.strip()]

    # Load HDX System
    microstates, representative_pdbs = load_hdx_system_data(args.microstates, args.representatives)

    # Initialize Likelihood Object
    hdx_lkhd = HDXMultiNestLikelihood(
        exp_csv=args.exp_csv, 
        time_points=args.time_points,
        deuterium_fraction=args.deuterium_fraction, 
        pH=args.pH, 
        temperature=args.temperature,
        peptide_list=peptide_list
    )

    # Initialize Sampler
    sampler = HDXMultiNestSampler(
        microstates=microstates,
        representative_pdbs=representative_pdbs,
        hdx_likelihood_obj=hdx_lkhd,
        max_states=args.max_states,
        output_dir=args.output
    )

    # Run Analysis: 1-State through N-States
    all_results = {}

    # 1. Single State Analysis
    print("\n--- Evaluating Single-State Models ---")
    for ms_id in sampler.microstate_ids:
        # Create a temp sampler for specific single microstate
        temp_sampler = HDXMultiNestSampler(
            microstates={ms_id: microstates[ms_id]},
            representative_pdbs={ms_id: representative_pdbs[ms_id]},
            hdx_likelihood_obj=hdx_lkhd,
            max_states=1,
            output_dir=args.output
        )
        res = temp_sampler.run_nested_sampling(n_states=1, n_live_points=args.n_live_points)
        all_results[f"1-state_{ms_id}"] = res

    # 2. Multi-State Ensemble Analysis
    print("\n--- Evaluating Multi-State Ensembles ---")
    for n in range(2, sampler.max_states + 1):
        res = sampler.run_nested_sampling(n_states=n, n_live_points=args.n_live_points)
        all_results[f"{n}-state"] = res

    # 3. Final Summary
    print("\n" + "="*30 + "\nFINAL EVIDENCE SUMMARY\n" + "="*30)
    for label, data in all_results.items():
        print(f"{label:15}: Log-Evidence = {data['log_evidence']:.3f} (+/- {data.get('log_evidence_error', 0):.3f})")

if __name__ == "__main__":
    main()