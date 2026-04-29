from __future__ import print_function
import matplotlib.pyplot as plt
import itertools
import mdtraj as md
import mdtraj.testing
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch

#function to load a pdb file using mdtraj
def load_pdb(pdb_file):
    return md.load_pdb(pdb_file)

# Function to extract residues from hydrogen bonds that takes in a trajectory and hydrogen bonds
def get_residues(t, hbond):
    res1 = t.topology.atom(hbond[0]).residue.index
    res2 = t.topology.atom(hbond[2]).residue.index
    return [res1, res2]

#defining a function to calculate number of hydrogen bonds for each residue using the baker-hubbard method. output is a dictio
def calculate_hbond_number(path_to_pdb):
    # Load the trajectory
    t = load_pdb(path_to_pdb)
    # Calculate hydrogen bonds
    hbonds = md.wernet_nilsson(t)[0]

    # Count occurrences of each unique residue
    # Count occurrences of each unique residue
    residue_counts = {}
    for hbond in hbonds:
        residues = get_residues(t, hbond)
        for res in residues:
            if res not in residue_counts:
                residue_counts[res] = 0
            residue_counts[res] += 1
    #add zeros for residues without hydrogen bonds
    all_residues = set(range(t.topology.n_residues))
    residues_with_hbonds = set(residue_counts.keys())
    residues_without_hbonds = all_residues - residues_with_hbonds
    for res in residues_without_hbonds:
        residue_counts[res] = 0
    #add 1 to all of the keys 
    residue_counts = {k+1: v for k, v in residue_counts.items()}
    return residue_counts

def count_heavy_atom_contacts(file_name, distance_threshold=5):
    """Count heavy-atom contacts for each residue."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_name)
    atom_list = [atom for atom in structure.get_atoms() if atom.element not in ['H']]
    ns = NeighborSearch(atom_list)
    contact_counts = {}
    for chain in structure.get_chains():
        for residue in chain:
            if residue.id[0] != ' ':  # Skip heteroatoms
                continue
            residue_atoms = [atom for atom in residue if atom.element not in ['H']]
            contacts = set()
            for atom in residue_atoms:
                neighbors = ns.search(atom.coord, distance_threshold, level='A')
                for neighbor in neighbors:
                    if neighbor.get_parent() != residue:  # Exclude same residue
                        contacts.add(neighbor)
            contact_counts[residue] = len(contacts)
    #simplifying contact counts to be a dictionary of residue number and contact count
    contact_counts = {residue.id[1]: count for residue, count in contact_counts.items()}
    return contact_counts

#function to return the protection factors. input is a pdb file and the bc and bh values, output is a dictionary of residue number and protection factor 
def estimate_protection_factors(pdb_file, bc=0.35, bh=2.0, distance_threshold=5):
    structure = load_pdb(pdb_file)
    contact_counts = count_heavy_atom_contacts(pdb_file, distance_threshold)
    h_bond_counts = calculate_hbond_number(pdb_file)
    protection_factors = {}
    for residue in contact_counts:
        h_bond_count = h_bond_counts[residue]
        heavy_atom_count = contact_counts[residue]
        protection_factor = bh * h_bond_count + bc * heavy_atom_count
        protection_factors[residue] = protection_factor    
    return protection_factors
