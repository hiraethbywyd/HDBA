import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

def get_protein_sequences_from_pdb(path_to_pdb: str):
    """
    Parses a PDB file and returns a list of primary sequences for all chains.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', path_to_pdb)
    sequences = []

    for model in structure:
        for chain in model:
            # Filters for standard amino acids (ignores heteroatoms/water)
            sequence = "".join([seq1(res.resname) for res in chain if res.id[0] == ' '])
            if sequence:
                sequences.append(sequence)
    return sequences

def generate_tryptic_peptides(sequence: str, min_len: int = 5, max_len: int = 20):
    """
    Cleaves sequence at K or R (if not followed by P).
    Returns a unique set of peptides within the specified length constraints.
    """
    # Trypsin rule: Cut after K or R, but not if P follows
    # The regex splits at the boundary where the rule matches
    raw_peptides = re.split(r'(?<=[KR])(?!P)', sequence)
    
    # Filter by length and return unique set
    return {p for p in raw_peptides if min_len <= len(p) <= max_len}

def find_peptide_locations(peptides, protein_sequence):
    """
    Locates start/end indices for a set of peptides within a protein.
    Returns a list of dictionaries for easy conversion to DataFrame.
    """
    results = []
    for pep in peptides:
        # Use re.finditer to find ALL occurrences of a peptide in case of repeats
        for match in re.finditer(f"(?={pep})", protein_sequence):
            start = match.start()
            end = start + len(pep) - 1
            results.append({
                'Peptide': pep,
                'Start': start, 
                'End': end, 
                'Length': len(pep)
            })
    return results

def plot_peptide_overlaps(path_to_pdb: str):
    """
    Visualizes where tryptic fragments align against the full protein sequence.
    """
    # 1. Data Acquisition
    sequences = get_protein_sequences_from_pdb(path_to_pdb)
    if not sequences:
        print("No sequences found.")
        return
    
    protein_seq = sequences[0] # Focus on the first chain
    fragments = generate_tryptic_peptides(protein_seq)
    peptide_data = find_peptide_locations(fragments, protein_seq)
    df = pd.DataFrame(peptide_data)

    if df.empty:
        print("No peptides matched the length criteria.")
        return

    # 2. Plot Setup
    fig, ax = plt.subplots(figsize=(15, 5))
    seq_len = len(protein_seq)
    
    ax.set_xlim(-1, seq_len)
    ax.set_ylim(-0.5, 5) # Adjusted for stacking
    ax.set_yticks([])
    
    # Display sequence on X-axis if it's short enough to be readable
    if seq_len < 100:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(list(protein_seq), fontsize=8)
    
    # 3. Plot Base Protein Line
    ax.hlines(0, 0, seq_len - 1, colors='black', lw=6, label='Protein Backbone')

    # 4. Greedy Stacking Algorithm
    # We sort peptides by start position to stack them cleanly
    df = df.sort_values('Start')
    layer_ends = [ -1 ] # Tracks the end position of the last peptide in each 'row'

    for _, row in df.iterrows():
        placed = False
        for i, end_pos in enumerate(layer_ends):
            # If this peptide starts after the last one in this row ended:
            if row['Start'] > end_pos:
                y_val = (i + 1) * 0.5
                rect = patches.Rectangle((row['Start'], y_val), row['Length'], 0.3, 
                                         color='skyblue', ec='blue', alpha=0.7)
                ax.add_patch(rect)
                layer_ends[i] = row['End']
                placed = True
                break
        
        # If no room in existing rows, create a new row
        if not placed:
            y_val = (len(layer_ends) + 1) * 0.5
            rect = patches.Rectangle((row['Start'], y_val), row['Length'], 0.3, 
                                     color='skyblue', ec='blue', alpha=0.7)
            ax.add_patch(rect)
            layer_ends.append(row['End'])

    plt.title(f'Tryptic Peptide Coverage (Chain 1 of {path_to_pdb})')
    plt.xlabel('Amino Acid Position')
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_peptide_overlaps("1f88.pdb")