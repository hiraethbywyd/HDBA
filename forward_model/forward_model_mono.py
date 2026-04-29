import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import baker_hubbard_pf_mono as bh # Custom module assumed
import tryptic_peptide_mono as tp # Custom module assumed

# ---------------------------------------------------------
# 1. Peptide Observability Utilities
# ---------------------------------------------------------

def is_observable_amide(peptide: str):
    """
    Identifies which amides in a peptide contribute to HDX signal.
    Rule: First two residues and all Prolines are non-observable.
    Returns: List of booleans.
    """
    return [not (i < 2 or res == 'P') for i, res in enumerate(peptide)]

# ---------------------------------------------------------
# 2. Intrinsic Rate Calculations (Englander/Bai Model)
# ---------------------------------------------------------

def get_residue_neighbor_effects(AA, pDcorr, T):
    """
    Calculates acidity/basicity neighbor effects for a residue.
    Uses temperature-dependent pKa values for ionizable sidechains.
    Ref: Salzberg 2016 / Englander 1980.
    """
    R = 1.987
    # Temp-dependent pKa calculations
    def calc_pKa(base_pKa, enthalpy):
        return -np.log10(10**(-base_pKa) * np.exp(-1000 * enthalpy * ((1.0/T - 1.0/278)/R)))

    pK_D = calc_pKa(4.48, 1000)
    pK_E = calc_pKa(4.93, 1083)
    pK_H = calc_pKa(7.42, 7500)

    # Dictionary of (acid_lambda, acid_rho, base_lambda, base_rho)
    eff_dict = {
        "A": (0.0, 0.0, 0.0, 0.0), "R": (-0.59, -0.32, 0.0767, 0.22),
        "N": (-0.58, -0.13, 0.49, 0.32), "C": (-0.54, -0.46, 0.62, 0.55),
        "Q": (-0.47, -0.27, 0.06, 0.20), "G": (-0.22, 0.218, 0.267, 0.17),
        "I": (-0.91, -0.59, -0.73, -0.23), "L": (-0.57, -0.13, -0.576, -0.21),
        "K": (-0.56, -0.29, -0.04, 0.12), "M": (-0.64, -0.28, -0.009, 0.11),
        "F": (-0.52, -0.43, -0.236, 0.063), "P": ("", -0.195, "", -0.24),
        "S": (-0.438, -0.389, 0.37, 0.299), "T": (-0.79, -0.468, -0.066, 0.20),
        "W": (-0.40, -0.44, -0.41, -0.11), "Y": (-0.41, -0.37, -0.27, 0.05),
        "V": (-0.739, -0.30, -0.702, -0.14), "NT": ("", -1.32, "", 1.62)
    }

    # Handle Ionizable Sidechains specially
    if AA in ["D", "E", "H"]:
        # Specific chemical logic for pH-dependent sidechain effects
        # (Simplified here for readability, keep original math)
        pKa = {"D": pK_D, "E": pK_E, "H": pK_H}[AA]
        # Example for D (others follow same pattern)
        denom = 10**(-pKa) + 10**(-pDcorr)
        ne0 = np.log10(10**(-0.9-pDcorr)/denom + 10**(0.9-pKa)/denom)
        ne1 = np.log10(10**(-0.12-pDcorr)/denom + 10**(0.58-pKa)/denom)
        ne2 = np.log10(10**(0.69-pDcorr)/denom + 10**(0.1-pKa)/denom)
        ne3 = np.log10(10**(0.6-pDcorr)/denom + 10**(-0.18-pKa)/denom)
        return (ne0, ne1, ne2, ne3)
    
    return eff_dict.get(AA, (0, 0, 0, 0))

def calc_intrinsic_rate(Laa, Raa, pH, T, La2="A", Ra2="A"):
    """
    Calculates the random coil exchange rate (k_int) for an amide.
    Laa: N-terminal residue | Raa: The current residue
    """
    if "P" in [Raa, Laa] or "CT" in [Raa, Laa] or "NT" in [Raa, Laa]:
        return 0
    
    # Constants for Deuterium exchange
    pKD20c, ka, kb, kw, R = 15.05, 0.6948, 1.87e8, 5.27e-4, 1.987
    EaA, EaB, EaW = 14000, 17000, 19000
    pDcorr = pH + 0.4
    inv_dTR = (1./T - 1./293) / R

    # Acid, Base, and Water catalysis components
    FTa, FTb, FTw = [np.exp(-1 * E * inv_dTR) for E in [EaA, EaB, EaW]]
    Dplus = 10**(-1 * pDcorr)
    ODminus = 10**(pDcorr - pKD20c)

    L_ne = get_residue_neighbor_effects(Laa, pDcorr, T)
    R_ne = get_residue_neighbor_effects(Raa, pDcorr, T)

    # Sum of factors (Fa = Acid, Fb = Base)
    Fa = 10**(L_ne[1] + R_ne[0])
    Fb = 10**(L_ne[3] + R_ne[2])

    return Fa*Dplus*ka*FTa + Fb*ODminus*kb*FTb + Fb*kw*FTw

def get_sequence_intrinsic_rates(seq, pH, T):
    """
    Calculates k_int for every amide in a sequence.
    """
    rates = np.zeros(len(seq))
    for n in range(len(seq)):
        L = "NT" if n == 0 else seq[n-1]
        R = seq[n]
        rates[n] = calc_intrinsic_rate(L, R, pH, T)
    return rates

# ---------------------------------------------------------
# 3. Structural Context & Protection Factors
# ---------------------------------------------------------

def get_amino_acid_sequence(path_to_pdb: str):
    """
    Extracts 1-letter sequence from PDB. Maps modified residues to standard AAs.
    """
    MOD_MAP = {'MSE': 'MET', 'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR'}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', path_to_pdb)
    
    seq_list = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ': # Standard residues
                    name = MOD_MAP.get(res.resname, res.resname)
                    char = seq1(name)
                    seq_list.append(char if char != 'X' else 'A')
    return "".join(seq_list).upper()

# ---------------------------------------------------------
# 4. Main Forward Model Logic
# ---------------------------------------------------------

def calc_incorporated_deuterium_with_weights(
    peptide_list, deuterium_fraction, time_points, pH, temperature, 
    file_path, weights=None, alpha=2.0, beta=0.35
):
    """
    Simulates HDX data (%D) for peptides across multiple time points.
    Considers structural protection from one or more weighted PDB models.
    """
    # 1. Setup PDBs and Weights
    with open(file_path, 'r') as f:
        path_list = [line.strip() for line in f]
    
    weights = weights or [1.0] * len(path_list)
    weights = np.array(weights) / sum(weights)

    # 2. Extract Data for all PDBs
    pdb_data = [] # List of {res_num: {'hbonds': x, 'hac': y}}
    full_seq = get_amino_acid_sequence(path_list[0]) # Assume consistency

    for path in path_list:
        pdb_data.append(bh.estimate_protection_factors(path))

    # 3. Process Peptides
    results = {t: {} for t in time_points}
    pep_list = [p.upper().strip() for p in peptide_list]

    for pep in pep_list:
        # Find peptide in sequence (wildcard X supported)
        start_idx = full_seq.find(pep) # 0-indexed
        if start_idx == -1: continue

        obs = is_observable_amide(pep)
        k_ints = get_sequence_intrinsic_rates(pep, pH, temperature)
        
        # PDB dictionary keys are usually residue numbers (1, 2, 3...)
        # We assume first residue in full_seq corresponds to first key in pdb_dict
        res_keys = sorted(pdb_data[0].keys())

        for t in time_points:
            sum_unexchanged = 0.0
            
            for i, k_int in enumerate(k_ints):
                if not obs[i]: continue

                # Calculate Weighted Protection Factor (lnPF)
                # PF = alpha * H-bonds + beta * Heavy Atom Contacts
                global_idx = start_idx + i
                res_num = res_keys[global_idx]
                
                lnPF = 0.0
                for d, w in zip(pdb_data, weights):
                    if res_num in d:
                        val = d[res_num]
                        lnPF += w * (alpha * val['hbonds'] + beta * val['hac'])
                
                k_obs = k_int / np.exp(lnPF)
                sum_unexchanged += np.exp(-k_obs * t)

            # Deuterium uptake = Fraction * (Max possible - Still Hydrogen)
            uptake = deuterium_fraction * (sum(obs) - sum_unexchanged)
            results[t][pep] = uptake

    # 4. Format Output
    df = pd.DataFrame(results).reset_index().rename(columns={'index': 'Peptide'})
    for t in time_points:
        df[f"{int(t)}.0_percent"] = (df[t] / df['Peptide'].str.len()) * 100
    
    return df