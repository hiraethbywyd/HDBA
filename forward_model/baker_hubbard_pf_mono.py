from __future__ import print_function
import mdtraj as md
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch

# ---------------------------------------------------------
# 1. Structural Loading Utilities
# ---------------------------------------------------------

def load_pdb_bio(pdb_filename):
    """Loads a PDB file using BioPython's PDBParser."""
    parser = PDBParser(QUIET=True)
    try:
        return parser.get_structure('protein', pdb_filename)
    except Exception as e:
        print(f"Error loading {pdb_filename} via BioPython: {e}")
        raise

# ---------------------------------------------------------
# 2. Hydrogen Bond Analysis (Baker-Hubbard)
# ---------------------------------------------------------

def calculate_hbond_number(path_to_pdb):
    """
    Identifies backbone hydrogen bonds using the Baker-Hubbard criteria:
    - Distance (Donor-Acceptor) < 2.5 A
    - Angle (NH-O) > 120 degrees
    Returns: {residue_id: hbond_count}
    """
    # MDTraj is used here for its optimized Baker-Hubbard implementation
    traj = md.load_pdb(path_to_pdb)
    hbonds = md.baker_hubbard(traj, periodic=False)
    
    # We focus on the first chain (standard for single-protein models)
    chain = traj.topology.chain(0)
    residue_counts = {res.index: 0 for res in chain.residues}

    for hbond in hbonds:
        # hbond[0] is the Donor atom index
        donor_res_idx = traj.topology.atom(hbond[0]).residue.index
        if donor_res_idx in residue_counts:
            residue_counts[donor_res_idx] += 1

    # Shift index by +1 to match standard PDB residue numbering
    return {k + 1: v for k, v in residue_counts.items()}

# ---------------------------------------------------------
# 3. Packing Analysis (Heavy Atom Contacts)
# ---------------------------------------------------------

def sigmoid_weight(distance, k=1.0, d0=0.0):
    """Sigmoid function used to weight contacts based on distance."""
    return 1 / (1 + np.exp(-k * (distance - d0)))

def count_heavy_atom_contacts_sigmoid(structure, k=1, d0=0.0, radius=6.5):
    """
    Calculates the 'local density' around each amide Nitrogen.
    - Excludes Hydrogens.
    - Excludes Prolines (no amide H).
    - Excludes immediate sequence neighbors (i +/- 2).
    """
    # Filter for heavy atoms (non-Hydrogen) in standard residues
    atom_list = [
        atom for atom in structure.get_atoms()
        if atom.element != 'H' and atom.get_parent().id[0] == ' '
    ]
    ns = NeighborSearch(atom_list)
    contact_counts = {}

    for chain in structure.get_chains():
        residues = list(chain)
        for i, residue in enumerate(residues):
            # Filtering: Standard AA, not Proline, must have backbone N
            if residue.id[0] != ' ' or residue.resname == 'PRO' or 'N' not in residue:
                continue

            target_n_coord = residue['N'].coord
            # Find all heavy atoms within the cutoff radius
            neighbors = ns.search(target_n_coord, level='A', radius=radius)

            contacts = 0
            for neighbor in neighbors:
                neigh_res = neighbor.get_parent()
                if neigh_res not in residues:
                    continue
                
                # Logical Constraint: Exclude self and 2 residues away in sequence
                neigh_idx = residues.index(neigh_res)
                if abs(neigh_idx - i) > 2:
                    dist = np.linalg.norm(neighbor.coord - target_n_coord)
                    contacts += sigmoid_weight(dist, k, d0)

            contact_counts[residue.id[1]] = contacts

    return contact_counts

# ---------------------------------------------------------
# 4. Final Protection Factor Estimation
# ---------------------------------------------------------

def estimate_protection_factors(file_input, bc=0.35, bh=2.0, radius=5.0):
    """
    The main driver function. Calculates averaged HBonds and HAC 
    across one or multiple PDB structures (an ensemble).

    Parameters:
    - file_input: single PDB path, list of paths, or a text file listing paths.
    - bc: Weight for heavy atom contacts (default 0.35).
    - bh: Weight for hydrogen bonds (default 2.0).
    """
    # 1. Normalize file input into a list of paths
    if isinstance(file_input, str):
        if file_input.lower().endswith('.pdb'):
            pdb_files = [file_input]
        else:
            with open(file_input, 'r') as f:
                pdb_files = [l.strip() for l in f if l.strip()]
    else:
        pdb_files = file_input

    # 2. Process Ensemble
    hbond_accumulator = {}
    hac_accumulator = {}
    obs_count = {}

    for pdb in pdb_files:
        structure = load_pdb_bio(pdb)
        
        # Calculate features for this specific PDB
        hac_data = count_heavy_atom_contacts_sigmoid(structure, radius=radius)
        hbond_data = calculate_hbond_number(pdb)

        # Aggregate for averaging
        for res_id in hac_data:
            hbond_accumulator[res_id] = hbond_accumulator.get(res_id, 0) + hbond_data.get(res_id, 0)
            hac_accumulator[res_id] = hac_accumulator.get(res_id, 0) + hac_data.get(res_id, 0)
            obs_count[res_id] = obs_count.get(res_id, 0) + 1

    # 3. Compute Averages
    final_results = {}
    for res_id in obs_count:
        final_results[res_id] = {
            "hbonds": hbond_accumulator[res_id] / obs_count[res_id],
            "hac": hac_accumulator[res_id] / obs_count[res_id]
        }

    return final_results