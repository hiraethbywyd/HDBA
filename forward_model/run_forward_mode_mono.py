eimport argparse
import numpy as np
import pandas as pd
from forward_model_mono import calc_incorporated_deuterium_with_weights, get_amino_acid_sequence
import baker_hubbard_pf_mono as bh
import tryptic_peptide_mono as tp
from hdx_likelihood_function_mono import calculate_sigma, total_likelihood, add_noised_data, total_likelihood_benchmark, likelihood, total_likelihood_test

#usage: run_forward_mode_mono.py [-h] -d DEUTERIUM_FRACTION -t TIME_POINTS [TIME_POINTS ...] -p PH -temp TEMPERATURE (--single_file SINGLE_FILE | --dual_file exp_file model_file) [-w WEIGHTS [WEIGHTS ...]] -o OUTPUT [-l PEPTIDE_LIST] run_forward_mode_mono.py: error: the following arguments are required: -d/--deuterium_fraction, -t/--time_points, -p/--pH, -temp/--temperature, -o/--output
#the way this runs right now is as follows:
    #DEUTERIUM_FRACTION = 0.5; this is the fraction of deuterium incorporated
    #TIME_POINTS = [0.0, 30.0, 60.0, 120.0, 240.0]; these are the time points at which the deuterium incorporation is measured
    #PH = 7.0; this is the pH value for intrinsic rate calculation
    #TEMPERATURE = 298.15; this is the temperature for intrinsic rate calculation

#    --single_file SINGLE_FILE; this is the path to the text file containing PDB paths for both experimental and model data (uses same data set)
#    --dual_file exp_file model_file; this is the path to the text file containing PDB paths for experimental data and model predictions
#    -w WEIGHTS [WEIGHTS ...]; this is the list of weights for the structures (optional, required if multiple PDB paths are provided)
#    -o OUTPUT; this is the output file path to save results
#    -l PEPTIDE_LIST; this is the text file containing list of peptides (optional)

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)
    
def load_experimental_csv(csv_path, time_points, peptide_list=None, fill_value=-1.0):
    """
    Load experimental HDX data from CSV and align to provided peptide list.

    - Numeric headers normalized to "<int>.0_percent".
    - All peptides in peptide_list are included; missing values filled with fill_value (-1.0 by default).
    """
    df = pd.read_csv(csv_path)
    if 'peptide' not in df.columns:
        raise KeyError("CSV must have a 'peptide' column")
    df = df.rename(columns={'peptide': 'Peptide'})
    df['Peptide'] = df['Peptide'].str.upper()

    # Normalize time columns
    rename_map = {}
    for col in df.columns:
        try:
            val = float(col)
            rename_map[col] = f"{int(val)}.0_percent"
        except ValueError:
            pass
    df = df.rename(columns=rename_map)

    # Ensure all time columns exist
    expected_cols = [f"{int(t)}.0_percent" for t in time_points]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # Optionally filter to provided peptide list
    if peptide_list:
        with open(peptide_list, 'r') as f:
            allowed = {line.strip().upper() for line in f if line.strip()}
        df['Peptide'] = df['Peptide'].str.upper()
        df = df[df['Peptide'].isin(allowed)]

    
    # Convert to numeric, fill missing
    df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors='coerce').fillna(fill_value)
    
    return df[['Peptide'] + expected_cols]

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
    #parser.add_argument('-f', '--file_path', type=str, required=True, help="Path to the text file containing PDB paths")
    #try this
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--single_file', type=str, help="Path to the text file containing PDB paths for both exp and model (uses same data set)")
        #the single_file argument takes one positional argument: 
            #SINGLE_FILE: an output data file of experimental HDX values in a CSV format
            #this file is used for both experimental and model predictions
    group.add_argument('--dual_file', nargs=2, metavar=('exp_file', 'model_file'), help="Paths to two text files containing PDB paths: first for experimental data, second for model predictions")
        #the dual_file argument takes two positional arguments: 
            #exp_file: an output data file of experimental HDX values in a CSV format)
                #the experimental data can also be generated from structure(s) if a text file with PDB path(s) is provided
                #it should be formatted as:
                    #peptide,30.0,60.0,180.0,600.0,1800.0,3600.0,7200.0
                    #LQALAQNNTETSEKIQASGILQ,13.06,21.83,41.81,48.45,75.42,90.56,98.89
            #model_file: a text file containing PDB path (single right now) for model predictions
    #
    parser.add_argument('-w', '--weights', type=float, nargs='+', help="List of weights for the structures (optional, required if multiple PDB paths are provided)")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file path to save results")
    parser.add_argument('-l', '--peptide_list', type=str, help="Text file containing list of peptides (optional)")

    return parser.parse_args()

def generate_deuteration_df(file_path, peptide_list, deuterium_fraction, time_points, pH, temperature, weights):
    if peptide_list is not None:
        # if it's a list, use it directly; if it's a string, treat as file
        if isinstance(peptide_list, str):
            with open(peptide_list, 'r') as f:
                peptides = [line.strip().upper() for line in f if line.strip()]
        elif isinstance(peptide_list, list):
            peptides = [p.strip().upper() for p in peptide_list if p.strip()]
        else:
            raise ValueError("peptide_list must be a list or a path to a file")
    else:
        with open(file_path, 'r') as f:
            path_list = [line.strip() for line in f]
        full_sequence = get_amino_acid_sequence(path_list[0])
        peptides = tp.generate_tryptic_peptides(full_sequence)

    peptides = [pep for pep in peptides if len(pep) > 1]

    # weights logic unchanged
    with open(file_path, 'r') as f:
        path_list = [line.strip() for line in f]
    num_pdb_files = len(path_list)
    if weights is None:
        weights = [1.0] * num_pdb_files
    elif len(weights) != num_pdb_files:
        raise ValueError("Number of weights must match number of PDB files.")

    return calc_incorporated_deuterium_with_weights(
        peptide_list=peptides,
        deuterium_fraction=deuterium_fraction,
        time_points=time_points,
        pH=pH,
        temperature=temperature,
        file_path=file_path,
        weights=weights
    )

def main():
    args = parse_arguments()

    peptide_list = args.peptide_list  # path to your peptide list

    # load one or two files
    if args.single_file:
        print("Single‐file mode (exp & model = same)")
        df_exp = generate_deuteration_df(
            args.single_file,
            peptide_list,
            args.deuterium_fraction,
            args.time_points,
            args.pH,
            args.temperature,
            args.weights
        )
        df_model = df_exp.copy()

    else:
        exp_file, model_file = args.dual_file
        print("Dual‐file mode (exp vs. model)")

        # Load experimental CSV aligned to peptide list
        if exp_file.lower().endswith('.csv'):
            df_exp = load_experimental_csv(
                exp_file,
                args.time_points,
                peptide_list=peptide_list,
                fill_value=-1.0  # missing peptides will be filled
            )
        else:
            df_exp = generate_deuteration_df(
                exp_file,
                peptide_list,
                args.deuterium_fraction,
                args.time_points,
                args.pH,
                args.temperature,
                args.weights
            )

        # Generate model DataFrame using same peptide list
        df_model = generate_deuteration_df(
            model_file,
            peptide_list,
            args.deuterium_fraction,
            args.time_points,
            args.pH,
            args.temperature,
            args.weights
        )

    print("Experimental DataFrame:")
    print(df_exp)
    print("Model DataFrame (head):")
    print(df_model.head())

    # run the likelihood function
    peptide_lkhd, overall_time_lkhd, total_lkhd, avg_lkhd = total_likelihood_test(df_exp, df_model)

    # map per-time likelihoods back into df_exp
    for t, lkhd_map in peptide_lkhd.items():
        df_exp[f"LogLKHD_{t}s"] = df_exp['Peptide'].map(lkhd_map)

    # add overall per-time summary
    df_exp['Total_LKHD_per_time'] = df_exp['Peptide'].map(
        lambda pep: sum(peptide_lkhd[t].get(pep, 0.0) for t in peptide_lkhd)
    )

    # save
    df_exp.to_csv(args.output, index=False)
    print(f"Results written to {args.output!r}")

# #this was for the benchmarking case with total_likelihood_test
# def main_test():
#     args = parse_arguments()

#     # load one or two files
#     if args.single_file:
#         print("Single‐file mode (exp & model = same)")
#         df_exp   = generate_deuteration_df(
#                        args.single_file, args.peptide_list,
#                        args.deuterium_fraction, args.time_points,
#                        args.pH, args.temperature, args.weights)
#         df_model = df_exp.copy()
#     else:
#         exp_file, model_file = args.dual_file
#         print("Dual‐file mode (exp vs. model)")
#         df_exp   = generate_deuteration_df(
#                        exp_file, args.peptide_list,
#                        args.deuterium_fraction, args.time_points,
#                        args.pH, args.temperature, args.weights)
#         df_model = generate_deuteration_df(
#                        model_file, args.peptide_list,
#                        args.deuterium_fraction, args.time_points,
#                        args.pH, args.temperature, args.weights)

#     print("Experimental DataFrame:")
#     print(df_exp)
#     print("Model DataFrame (head):")
#     print(df_model)

#     # run the likelihood function
#     peptide_lkhd, overall_time_lkhd, total_lkhd, avg_lkhd = \
#         total_likelihood_test(df_exp, df_model)

#     # map per-time likelihoods back into df_exp
#     for t, lkhd_map in peptide_lkhd.items():
#         df_exp[f"LogLKHD_{t}s"] = df_exp['Peptide'].map(lkhd_map)

#     # add overall per-time summary
#     df_exp['Total_LKHD_per_time'] = df_exp['Peptide'].map(lambda pep: 
#         sum(peptide_lkhd[t].get(pep, 0.0) for t in peptide_lkhd)
#     )

#     # save
#     df_exp.to_csv(args.output, index=False)
#     print(f"Results written to {args.output!r}") 

if __name__ == "__main__":
    main()