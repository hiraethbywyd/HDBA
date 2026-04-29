#all from Saltzberg 2016
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import forward_model as fm
import scipy as sp 

def noise_model(d_exp: float, d_model: float, sigma: float, A: float, B: float):
    exp_part = (np.exp(np.power(d_exp - d_model, 2) / (2 * np.power(sigma, 2)))) / (np.sqrt(2 * np.pi) * sigma)
    erf_part = 0.5 * (sp.special.erf((A - d_exp) / (np.sqrt(2) * sigma))) - (sp.special.erf((B - d_exp) / (np.sqrt(2) * sigma)))
    result = exp_part * erf_part
        #print(result)
    return max(result, 1e-10)  # Ensure the result is always positive

def calculate_sigma(replicates):
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

def total_likelihood(df: pd.DataFrame, a=-0.1, b=1.2, phi=0.85):
    list_of_peptides = set(df['Peptide'])
    time_points = ['0', '30', '60', '300', '900', '3600', '14400', '84600']
    replicate = ['1', '2', '3']
    
    peptide_avg_likelihoods_per_time = {time: {} for time in time_points}
    overall_likelihood_per_time = {}
    
    for time in time_points:
        time_float = float(time)  # Ensure time is in float format
        total_log_likelihood_per_time = 0
        valid_peptide_count = 0  # To count the number of valid peptides for averaging
        
        for peptide in list_of_peptides:
            peptide_log_likelihood = 0
            count = 0  # To count the number of valid likelihoods for averaging
            
            for rep in replicate:
                d_exp = df[df['Peptide'] == peptide][f'{time_float}s_noised_{rep}'].iloc[0]
                d_model = df[df['Peptide'] == peptide][time_float].iloc[0]
                # Define columns using f-strings
                columns = [f'{time_float}s_noised_1', f'{time_float}s_noised_2', f'{time_float}s_noised_3']
                all_reps = np.array(df[df['Peptide'] == peptide][columns]).flatten()
                sigma = calculate_sigma(all_reps)
                A = a * phi
                B = b * phi
                noise_val = noise_model(d_exp, d_model, sigma, A, B)
                if noise_val > 0:
                    peptide_log_likelihood += -np.log(noise_val)
                    count += 1
                        #print(f"Peptide: {peptide}, Time: {time_float}, Replicate: {rep}, Noise: {noise_val}, Negative Log Likelihood: {peptide_log_likelihood}")
                else:
                    peptide_log_likelihood += -np.inf  # Handle invalid values by adding negative infinity
                        #print(f"Peptide: {peptide}, Time: {time_float}, Replicate: {rep}, Noise: {noise_val} (Invalid)")
            
            # get the average likelihood for each peptide at this time point
            if count > 0:
                avg_likelihood = peptide_log_likelihood / count
                    #print(f"Peptide: {peptide}, Time: {time_float}, Average Log Likelihood: {avg_likelihood}")
                peptide_avg_likelihoods_per_time[time][peptide] = avg_likelihood
                total_log_likelihood_per_time += avg_likelihood
                valid_peptide_count += 1
            else:
                peptide_avg_likelihoods_per_time[time][peptide] = -np.inf
        
        # get the overall average likelihood for this time point
        if valid_peptide_count > 0:
            overall_avg_likelihood = total_log_likelihood_per_time / valid_peptide_count
            overall_likelihood_per_time[time] = overall_avg_likelihood
            print(f"Overall Average Log Likelihood: {overall_avg_likelihood}, Time: {time_float}")
        else:
            overall_likelihood_per_time[time] = -np.inf
    
    return peptide_avg_likelihoods_per_time, overall_likelihood_per_time