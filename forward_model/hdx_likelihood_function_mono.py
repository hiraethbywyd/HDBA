#all from Saltzberg 2016
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import forward_model_mono as fm
import scipy as sp 

def noise_model(d_exp: float, d_model: float, sigma: float, A: float, B: float):
    exp_part = (np.exp(np.power(d_exp - d_model, 2) / (2 * np.power(sigma, 2)))) / (np.sqrt(2 * np.pi) * sigma)
    erf_part = 0.5 * (sp.special.erf((A - d_exp) / (np.sqrt(2) * sigma))) - (sp.special.erf((B - d_exp) / (np.sqrt(2) * sigma)))
    result = exp_part * erf_part
        #print(result)
    return max(result, 1e-10)  # Ensure the result is always positive

def calculate_sigma(replicates):#fix this
    # Convert input to numpy array (in case it's a list)
    replicates = np.array(replicates)
    # Calculate the standard deviation using numpy's std function with ddof=1 for sample std dev
    std_dev = np.std(replicates, ddof=1)  # ddof=1 for sample standard deviation (Bessel's correction)
    return std_dev

def add_noised_data(df, time_points):
    """
    Add synthetic 'noised' data to the DataFrame.
    """
    print("Initial DataFrame columns:", df.columns)  # Debugging line to print initial columns

    for time in time_points:
        time_float = float(time)  # Ensure time is in float format
        if time_float not in df.columns:
            raise KeyError(f"Column '{time_float}' not found in DataFrame columns: {df.columns}")

        for rep in range(1, 4):
            noise_column = f'{time_float}s_noised_{rep}'
            df[noise_column] = df[time_float] + np.random.normal(loc=0, scale=0.05, size=len(df))
    return df

import numpy as np
import scipy as sp

def likelihood(d_exp, d_model, sigma, A=-0.1, B=1.2):
    """
    Robust truncated normal likelihood function for HDX data.
    Returns a small positive probability instead of raising an error when the truncation
    range has negligible probability mass.
    
    Parameters
    ----------
    d_exp : float
        Experimental deuteration fraction (0-1).
    d_model : float
        Modeled deuteration fraction (0-1).
    sigma : float
        Standard deviation of measurement noise.
    A : float, default = -0.1
        Lower truncation bound.
    B : float, default = 1.2
        Upper truncation bound.
    """
    # ---- 1) Safety checks and clamps ----
    epsilon = 1e-12
    min_sigma = 1e-6  # prevent sigma from being 0 or too small

    # Coerce to floats
    try:
        d_exp = float(d_exp)
        d_model = float(d_model)
        sigma = float(sigma)
    except Exception:
        return 1e-300  # If inputs aren't numbers, return tiny probability

    # Clamp deuteration values strictly inside [0,1]
    d_exp = min(max(d_exp, epsilon), 1.0 - epsilon)
    d_model = min(max(d_model, epsilon), 1.0 - epsilon)

    # Ensure sigma never zero or negative
    sigma = max(sigma, min_sigma)

    if A >= B:
        raise ValueError("Lower bound A must be less than upper bound B")

    # ---- 2) Compute normalization constant ----
    Z = 2 * (
        sp.special.erf((B - d_model) / (sigma * np.sqrt(2))) -
        sp.special.erf((A - d_model) / (sigma * np.sqrt(2)))
    )

    # ---- 3) Handle pathological Z ----
    if not np.isfinite(Z) or Z <= 1e-12:
        # Instead of raising an error, return tiny probability to penalize
        return 1e-300

    # ---- 4) Compute unnormalized Gaussian term ----
    leftfunction = (
        (1.0 / (sigma * np.sqrt(2 * np.pi))) *
        np.exp(-0.5 * ((d_exp - d_model) / sigma) ** 2)
    )

    # ---- 5) Final probability ----
    result = leftfunction / Z

    # Guard against NaN or negative values
    if not np.isfinite(result) or result <= 0:
        return 1e-300

    return (result)

def total_likelihood(df_exp: pd.DataFrame, df_model: pd.DataFrame, sigma: float = 0.1):
    #the 'real' likelihood function

    # 1) create "X.0_percent" columns, extract X as the string time-point
    #    Coerce each column name to str before testing for the suffix.
    percent_cols = [c for c in df_exp.columns if str(c).endswith('_percent')]
    time_points  = [str(c).split('.')[0] for c in percent_cols]

    # Build an ordered list of peptides appearing in df_exp that are also in df_model
    model_peps = set(df_model['Peptide'])
    common_peptides = [
        pep for pep in df_exp['Peptide'].tolist()
        if pep in model_peps
    ]

    # Sanity checks
    print(f"Found {len(common_peptides)} common peptides (first 10): {common_peptides[:10]}")
    print("Evaluating time points:", time_points)

    # Storage containers
    peptide_log_lkhd = {t: {} for t in time_points}
    overall_per_time = {}
    total_log_lkhd   = 0.0
    total_count      = 0

    # Loop through each time point
    for t in time_points:
        col = f"{t}.0_percent"
        time_sum = 0.0
        count    = 0

        for pep in common_peptides:
            try:
                d_e = df_exp.loc[df_exp['Peptide'] == pep, col].iloc[0] / 100
                d_m = df_model.loc[df_model['Peptide'] == pep, col].iloc[0] / 100
            except (IndexError, KeyError):
                # missing column or peptide => skip
                continue

            p = likelihood(d_e, d_m, sigma)
            loglk = -np.log(p) if (p is not None and p > 0) else np.inf

            peptide_log_lkhd[t][pep] = loglk
            time_sum += loglk
            count    += 1

        overall_per_time[t] = time_sum if count > 0 else np.inf
        total_log_lkhd    += time_sum
        total_count       += count

        print(f"• Time {t}s: {count} peptides, sum log‐lkhd = {time_sum:.2f}")

    overall_avg = total_log_lkhd / total_count if total_count else np.inf
    print(f"TOTAL log‐likelihood = {total_log_lkhd:.2f}")
    print(f"AVG   log‐likelihood = {overall_avg:.4f}")

    return peptide_log_lkhd, overall_per_time, total_log_lkhd, overall_avg

def total_likelihood_test(
    df_exp: pd.DataFrame,
    df_model: pd.DataFrame,
    sigma: float = 0.1
):
    # Ensure sigma has a minimum value to prevent numerical issues
    sigma = max(sigma, 1e-6)  # Prevent sigma from being zero or too small
    
    # ——————————————————————————————————————————————
    # 1) Dynamically pull all the "X.0_percent" columns, extract X as the string time-point
    #    Coerce each column name to str before testing for the suffix.
    percent_cols = [c for c in df_exp.columns if str(c).endswith('_percent')]
    time_points  = [str(c).split('.')[0] for c in percent_cols]
    # ——————————————————————————————————————————————

    # Build an ordered list of peptides appearing in df_exp that are also in df_model
    model_peps = set(df_model['Peptide'])
    common_peptides = [
        pep for pep in df_exp['Peptide'].tolist()
        if pep in model_peps
    ]

    # Sanity checks
    print(f"Found {len(common_peptides)} common peptides (first 10): {common_peptides[:10]}")
    print("Evaluating time points:", time_points)

    # Storage containers
    peptide_log_lkhd = {t: {} for t in time_points}
    overall_per_time = {}
    total_log_lkhd   = 0.0
    total_count      = 0

    # Loop through each time point
    for t in time_points:
        col = f"{t}.0_percent"
        time_sum = 0.0
        count    = 0

        for pep in common_peptides:
            try:
                d_e = df_exp.loc[df_exp['Peptide'] == pep, col].iloc[0] / 100
                d_m = df_model.loc[df_model['Peptide'] == pep, col].iloc[0] / 100
            except (IndexError, KeyError):
                # missing column or peptide => skip
                continue

            p = likelihood(d_e, d_m, sigma)
            loglk = -np.log(p) if (p is not None and p > 0) else np.inf

            peptide_log_lkhd[t][pep] = loglk
            time_sum += loglk
            count    += 1

        overall_per_time[t] = time_sum if count > 0 else np.inf
        total_log_lkhd    += time_sum
        total_count       += count

        print(f"• Time {t}s: {count} peptides, sum log‐lkhd = {time_sum:.2f}")

    overall_avg = total_log_lkhd / total_count if total_count else np.inf
    print(f"TOTAL log‐likelihood = {total_log_lkhd:.2f}")
    print(f"AVG   log‐likelihood = {overall_avg:.4f}")

    return peptide_log_lkhd, overall_per_time, total_log_lkhd, overall_avg

def total_likelihood_benchmark(df: pd.DataFrame, a=-0.1, b=1.2, phi=0.85):
    list_of_peptides = set(df['Peptide'])
    time_points = ['0', '30', '60', '300', '900', '3600', '14400', '84600']
    
    peptide_avg_likelihoods_per_time = {time: {} for time in time_points}
    overall_likelihood_per_time = {}
    total_log_likelihood_all_times = 0  # Initialize total likelihood across all time slices
    total_valid_peptide_count = 0  # Initialize count of valid peptide-time pairs

    sigma = 1  # Standard deviation for the Gaussian distribution test case
    
    #should this be considering peptide deuteration fraction or peptide deuteration percent?

    for time in time_points:
        # time_float = float(time)  # Ensure time is in float format
        percent_col = f"{time}.0_percent" if time == '0' else f"{time}.0_percent"

        total_log_likelihood_per_time = 0
        valid_peptide_count = 0  # To count the number of valid peptides for averaging

        for peptide in list_of_peptides:
            d_exp = df[df['Peptide'] == peptide][percent_col].iloc[0]/100
            d_model = df[df['Peptide'] == peptide][percent_col].iloc[0]/100
            # d_exp = df[df['Peptide'] == peptide][time_float].iloc[0]
            # d_model = df[df['Peptide'] == peptide][time_float].iloc[0]
            #print(f"Pep Tracking: {peptide}, Time: {time_float}, d_exp: {d_exp}, d_model: {d_model}")
            
            probability = likelihood(d_exp, d_model, sigma)
            print(f"Peptide: {peptide}, Time: {percent_col}, Likelihood: {probability}")
            
            if probability > 0:  # Avoid taking log of zero
                peptide_log_likelihood = -np.log(probability)
            else:
                peptide_log_likelihood = -np.inf  # Handle invalid values by adding negative infinity

            peptide_avg_likelihoods_per_time[time][peptide] = peptide_log_likelihood
            total_log_likelihood_per_time += peptide_log_likelihood
            valid_peptide_count += 1

        # Get the overall average likelihood for this time point
        # Get the sum of the individual peptide likelihoods for this time point
        if valid_peptide_count > 0:
            overall_sum_likelihood_per_time = total_log_likelihood_per_time
            overall_likelihood_per_time[time] = overall_sum_likelihood_per_time
            print(f"Sum of Neg. Log Likelihood: {overall_sum_likelihood_per_time}, Time: {percent_col}")
        else:
            overall_likelihood_per_time[time] = -np.inf

        total_log_likelihood_all_times += total_log_likelihood_per_time  # Add to total likelihood across all time slices
        total_valid_peptide_count += valid_peptide_count  # Add to total count of valid peptide-time pairs

    # Calculate the overall average likelihood across all time slices
    if total_valid_peptide_count > 0:
        overall_avg_likelihood_all_times = total_log_likelihood_all_times / total_valid_peptide_count
    else:
        overall_avg_likelihood_all_times = -np.inf

    print(f"Total Log Likelihood Across All Time Slices: {total_log_likelihood_all_times}")
    print(f"Overall Average Log Likelihood Across All Time Slices: {overall_avg_likelihood_all_times}")

    return peptide_avg_likelihoods_per_time, overall_likelihood_per_time, total_log_likelihood_all_times, overall_avg_likelihood_all_times
