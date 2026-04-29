import forward_model as fm 
import numpy as np
import pandas as pd

#defining standard gaussian distribution, takes in center, std, and number of points
def gaussian_noise(mean:float, sigma:float, n:int):
    return np.random.normal(mean, sigma, n)

fm.calc_percentage_deuterium_per_peptide('PEPTIDE', 0.5, 1, 7, 298, 'path_to_pdb')

fm.calc_percentage_deuterium_per_peptide()

def add_noise_to_forward_model(petptide:str, deuteration_fraction:float, time:float, pH: float, temperature: float, path_to_pdb:str, mean:float, sigma:float, n:int):
    return fm.calc_percentage_deuterium_per_peptide(petptide, deuteration_fraction, time, pH, temperature, path_to_pdb) + gaussian_noise(mean, sigma, n)


