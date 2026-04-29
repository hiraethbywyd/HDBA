import argparse
import numpy as np
import pandas as pd
from forward_model import calc_incorporated_deuterium, calc_incorporated_deuterium_weighted, get_amino_acid_sequence
import baker_hubbard_pf as bh
import tryptic_peptides as tp
from hdx_likelihood_function import calculate_sigma, total_likelihood, add_noised_data


def parse_arguments():
    """
    Parse command line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Calculate incorporated deuterium at multiple time points.")
    
    # Add arguments 
    parser.add_argument('-d', '--deuterium_fraction', type=float, required=True, help="Fraction of deuterium incorporated (float)")
    parser.add_argument('-t', '--time_points', type=float, nargs='+', required=True, help="List of time points (comma-separated floats)")
    parser.add_argument('-p', '--pH', type=float, required=True, help="pH value for intrinsic rate calculation")
    parser.add_argument('-temp', '--temperature', type=float, required=True, help="Temperature for intrinsic rate calculation")
    parser.add_argument('-f', '--file_path', type=str, required=True, help="Path to the text file containing PDB paths")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file path to save results")
    parser.add_argument('-l', '--peptide_list', type=str, help="Text file containing list of peptides (optional)")
    parser.add_argument('-w', '--weights', type=str, help="Text file containing list of weights for each structure (optional)")
    
    return parser.parse_args()

def main():
    # Parse the arguments
    args = parse_arguments()

    # fork if peptide list is not provided, generate tryptic peptides
    if args.peptide_list:
        with open(args.peptide_list, 'r') as f:
            peptide_list = [line.strip() for line in f]
    else:
        # make tryptic peptides from the PDB file
        with open(args.file_path, 'r') as f:
            path_list = [line.strip() for line in f if line.strip()]  # Ensure only valid file paths are included
        print(f"File paths read from {args.file_path}: {path_list}")
        path_to_pdb = path_list[0]  # Take the first file path to generate the sequence
        full_sequence = get_amino_acid_sequence(path_to_pdb)
        peptide_list = tp.generate_tryptic_peptides(full_sequence)

    # Filter out peptides that are too short
    peptide_list = [pep for pep in peptide_list if len(pep) > 1]

    # Check if weights file is provided and read the weights
    if args.weights:
        with open(args.weights, 'r') as f:
            weights = [float(line.strip()) for line in f]
    else:
        weights = None

    # Print debug information for file paths and weights
    print(f"File paths: {path_list}")
    print(f"Weights: {weights}")

    # define which function to use
    if weights and path_list:
        # Ensure the number of weights matches the number of file paths
        if len(weights) != len(path_list):
            raise ValueError("The number of weights must match the number of file paths.")
        
        # Call calc_incorporated_deuterium_weighted from forward_model.py
        deuteration_df = calc_incorporated_deuterium_weighted(
            peptide_list=peptide_list,
            deuterium_fraction=args.deuterium_fraction,
            time_points=args.time_points,
            pH=args.pH,
            temperature=args.temperature,
            file_paths=path_list,
            weights=weights
        )
    else:
        # Call calc_incorporated_deuterium from forward_model.py
        deuteration_df = calc_incorporated_deuterium(
            peptide_list=peptide_list,
            deuterium_fraction=args.deuterium_fraction,
            time_points=args.time_points,
            pH=args.pH,
            temperature=args.temperature,
            file_path=args.file_path
        )

    # Add synthetic 'noised' data to the DataFrame
    deuteration_df = add_noised_data(deuteration_df, args.time_points)

    print(deuteration_df)

    # Calculate the total likelihood for each time point and add to DataFrame
    peptide_avg_likelihoods, overall_likelihood = total_likelihood(deuteration_df)
    
    # Add the average likelihoods to the DataFrame
    for peptide, avg_likelihood in peptide_avg_likelihoods.items():
        deuteration_df.loc[deuteration_df['Peptide'] == peptide, 'Avg_Likelihood'] = avg_likelihood
    
    # Add the overall likelihood to the DataFrame
    deuteration_df['Overall_Likelihood'] = overall_likelihood

    # Save the dataframe to a CSV file
    try:
        deuteration_df.to_csv(args.output, index=False)
        print(f"Results have been saved to {args.output}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")    

if __name__ == "__main__":
    main()