from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import pandas as pd

#get protein sequences from a pdb file using biopython
def get_protein_sequences_from_pdb(path_to_pdb: str):
    parser = PDBParser()
    structure = parser.get_structure('protein', path_to_pdb)
    sequences = []
    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if residue.id[0] == ' ':
                    three_letter_code = residue.resname
                    one_letter_code = seq1(three_letter_code)
                    sequence += one_letter_code
            sequences.append(sequence)
    return sequences

#finds cut sites in a protein sequence for trypsin
def find_cut_sites(sequence):
    cut_sites = []
    for i in range(len(sequence) - 1):
        if sequence[i] in 'KR' and (i + 1 >= len(sequence) or sequence[i + 1] != 'P'):
            cut_sites.append(i + 1)  # Cut after the K or R
    return cut_sites

def generate_tryptic_peptides(sequence):
    """
    Generate tryptic peptides from a protein sequence.

    Parameters:
    - sequence (str): The full amino acid sequence of the protein.

    Returns:
    - List of tryptic peptides
    """
    import re
    # Split the sequence at K or R, but not if followed by P
    peptides = re.split(r'(?<=[KR])(?!P)', sequence)
    return peptides

# generate all possible tryptic peptides from a protein sequence, but it is not efficient for large proteins so I didn't use it

# def generate_fragments(path_to_pdb:str):
#     sequence = get_protein_sequences_from_pdb(path_to_pdb)[0]
#     cut_sites = find_cut_sites(sequence)
#     n = len(cut_sites)
#     fragments = []

#     # Generate all possible combinations of cut sites
#     for i in range(n + 1):  # +1 to include the case with no cuts
#         for combo in combinations(cut_sites, i):
#             start = 0
#             fragment = []
#             for cut in combo:
#                 fragment.append(sequence[start:cut])
#                 start = cut
#             fragment.append(sequence[start:])  # Add the last fragment
#             fragments.extend([frag for frag in fragment if 5 <= len(frag) <= 20])
#     #this removes all redundant fragments
#     return set(fragments)

# def generate_fragments(path_to_pdb: str):
#     sequence = get_protein_sequences_from_pdb(path_to_pdb)[0]
#     cut_sites = find_cut_sites(sequence)
#     fragments = set()

#     # Generate fragments directly
#     start = 0
#     for cut in cut_sites:
#         fragment = sequence[start:cut]
#         if 5 <= len(fragment) <= 20:
#             fragments.add(fragment)
#         start = cut

#     # Add the last fragment
#     fragment = sequence[start:]
#     if 5 <= len(fragment) <= 20:
#         fragments.add(fragment)

#     return fragments



#takes in a list of peptide sequences and a protein sequence and outputs the start and end indices of the peptides in the protein sequence
def find_peptide_indices(peptides, protein_sequence):
    indices = []
    for peptide in peptides:
        start = protein_sequence.find(peptide)
        end = start + len(peptide) - 1
        indices.append((start, end))
    return indices

#takes in a single peptide sequence and a protein sequence and outputs the start and end indices of the peptide in the protein sequence
def find_peptide_indices_single(peptide, protein_sequence):
    indices = []
    start = protein_sequence.find(peptide)
    end = start + len(peptide) - 1
    indices.append((start, end))
    return indices

def peptide_df(sequence,peptides):
    peptide_df = pd.DataFrame(peptides, columns=['Peptide'])
    peptide_df['Length'] = peptide_df['Peptide'].apply(len)
    peptide_df['Start'] = peptide_df['Peptide'].apply(lambda x: sequence.find(x))
    peptide_df['End'] = peptide_df['Start'] + peptide_df['Length'] - 1
    return peptide_df

# this fucntion takes in a native protein protein sequence and a list of peptide sequences and outputs a plot showing the overlap of the peptides with the protein sequence
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#for visualization of petide coverage 
def plot_peptide_overlaps(path_to_pdb:str):
    # Get the protein sequence from the PDB file
    sequence = get_protein_sequences_from_pdb(path_to_pdb)[0]
    #need to uncomment the generate_fragments function to use this
    fragments = generate_fragments(path_to_pdb)
    peptides = find_peptide_indices(fragments, sequence)
    
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.set_xlim(0, len(sequence))
    ax.set_ylim(0, 1)  # Set y-axis limit to 1 since we will plot non-overlapping fragments at the same height
    ax.set_yticks([])
    ax.set_xticks(range(len(sequence)))
    ax.set_xticklabels(sequence)  # Set x-tick labels to the amino acid sequence
    
    # Plot the native sequence as a line
    ax.plot(range(len(sequence)), [0.05] * len(sequence), color='black', lw=4)
    
    # Track the y-coordinates for each peptide
    y_positions = [0.1]  # Start with the first y-coordinate
    peptide_y_positions = []  # List to store y-coordinates of plotted peptides
    
    for start, end in peptides:
        # Find a y-coordinate that does not overlap with existing peptides
        for y in y_positions:
            if all(not (start < e and end > s) for (s, e), y_pos in zip(peptides, peptide_y_positions) if y == y_pos):
                break
        else:
            y = y_positions[-1] + 0.1  # Increment y-coordinate if no non-overlapping position is found
            y_positions.append(y)
        
        peptide_y_positions.append(y)
        rect = patches.Rectangle((start, y), end - start, 0.05, linewidth=1, edgecolor='r', facecolor='r', alpha=0.5)
        ax.add_patch(rect)
    
    plt.title('Peptide Overlaps with Native Sequence')
    plt.xlabel('Sequence')
    plt.ylabel('Unique Tryptic Peptides')