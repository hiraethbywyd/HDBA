import re
import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
reference_pdb = "/Users/kehuang/Documents/projects/nsp2/analysis/models/expt_complete.pdb"
MDmodels_dir = "/Users/kehuang/Documents/projects/nsp2/analysis/models/MDmodels"
AFmodels_dir = "/Users/kehuang/Documents/projects/nsp2/analysis/models/AFmodels/filter"

# All possible time points in the data
time_points = [30, 60, 180, 600, 1800, 3600, 7200]

# Only plot these
plot_time_points = [30, 60, 180, 600]

# ------------------------------------------------------------
# STEP 1: READ AND DEBUG data.txt
# ------------------------------------------------------------
with open("data.txt") as f:
    raw_data = f.read()

print("First 20 lines of data.txt:")
for i, line in enumerate(raw_data.splitlines()):
    print(line)
    if i > 20:
        break

samples = raw_data.strip().split("------------------------------")
print(f"\nTotal samples parsed: {len(samples)}")

data_by_time = {t: [] for t in time_points}
expt_values = {}  # experimental log-likelihoods
model_loglikelihoods = {}  # {model_name_time: value}
model_sources = {}  # Track which directory (MD or AF) the model came from

# ------------------------------------------------------------
# Function to find the PDB file for a given sample_name
# ------------------------------------------------------------
def find_model_file(sample_name):
    # Search in MDmodels directory
    for file in os.listdir(MDmodels_dir):
        if sample_name in file and file.endswith(".pdb"):
            return os.path.join(MDmodels_dir, file), "MD"

    # Search in AFmodels directory
    for file in os.listdir(AFmodels_dir):
        if sample_name in file and file.endswith(".pdb"):
            return os.path.join(AFmodels_dir, file), "AF"

    return None, None

# ------------------------------------------------------------
# STEP 2: PARSE data.txt AND MATCH TO FILES
# ------------------------------------------------------------
for sample in samples:
    sample = sample.strip()
    if not sample:
        continue

    lines = sample.split("\n")
    sample_name = lines[0].replace("Results for ", "").strip()
    print(f"\nProcessing sample: {sample_name}")

    # If this is the experimental data block
    if sample_name == "expt_nsp2":
        for line in lines[1:]:
            match = re.search(r'Time\s+(\d+)s:.*?sum log.*?=\s*([0-9]+\.[0-9]+)', line)
            if match:
                time = int(match.group(1))
                value = float(match.group(2))
                if time in time_points:
                    expt_values[time] = value
                    print(f"  Experimental value at time {time}: {value}")
        continue  # Skip PDB matching for experimental data

    # Locate PDB file
    model_file, source = find_model_file(sample_name)
    if not model_file:
        print(f"  WARNING: No PDB found for sample '{sample_name}'")
        continue

    # Iterate through time-value pairs
    for line in lines[1:]:
        match = re.search(r'Time\s*?(\d+).*?([0-9]+\.[0-9]+)', line)
        if match:
            time = int(match.group(1))
            value = float(match.group(2))

            # Ensure time point is valid
            if time in time_points:
                model_name = f"{sample_name}_{time}"
                model_loglikelihoods[model_name] = value
                model_sources[model_name] = source
                data_by_time[time].append(value)

# ------------------------------------------------------------
# Check counts
# ------------------------------------------------------------
print("\nData counts per time point:")
for t in time_points:
    print(f"  Time {t}: {len(data_by_time[t])} models")

# Exit early if no models
all_values = [v for t in plot_time_points for v in data_by_time[t]]
if not all_values:
    raise RuntimeError("No models found! Check model directories or data.txt names.")

xmin, xmax = min(all_values), max(all_values)

# ------------------------------------------------------------
# HISTOGRAM PLOTS FOR SELECTED TIMES
# ------------------------------------------------------------
fig, axs = plt.subplots(1, len(plot_time_points), figsize=(12, 5), sharey=True)

for i, t in enumerate(plot_time_points):
    md_values = [model_loglikelihoods[mn] for mn in model_loglikelihoods if mn.endswith(f"_{t}") and model_sources[mn] == "MD"]
    af_values = [model_loglikelihoods[mn] for mn in model_loglikelihoods if mn.endswith(f"_{t}") and model_sources[mn] == "AF"]

    # MDmodels histogram
    if md_values:
        counts, bins = np.histogram(md_values, bins=100, range=(xmin, xmax))
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        axs[i].plot(bin_centers, counts, drawstyle="steps-mid", color="blue", label="MD")

    # AFmodels histogram
    if af_values:
        counts, bins = np.histogram(af_values, bins=100, range=(xmin, xmax))
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        axs[i].plot(bin_centers, counts, drawstyle="steps-mid", color="green", label="ColabFold")

    axs[i].set_title(f'{t}s', fontsize=18, fontweight='bold')
    #axs[i].set_xlabel("Score", fontsize=18, fontweight='bold')
    fig.supxlabel("Score", fontsize=18, fontweight='bold')

    if i == 0:
        axs[i].set_ylabel("Number of models", fontsize=18, fontweight='bold')
    axs[i].tick_params(axis='both', labelsize=12)

    # Red vertical line for experimental score
    if t in expt_values:
        axs[i].axvline(expt_values[t], color='red', linestyle='--', linewidth=1, label='cryo-EM')

    #axs[i].legend(fontsize=12)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# RMSD CALCULATION
# ------------------------------------------------------------
u_ref = mda.Universe(reference_pdb)
ref_CA = u_ref.select_atoms("name CA")
ref_coords = ref_CA.positions.copy()

def compute_rmsd(model_name):
    base_name = model_name.rsplit("_", 1)[0]
    model_file, _ = find_model_file(base_name)
    if not model_file:
        return None

    try:
        u_model = mda.Universe(model_file)
        model_CA = u_model.select_atoms("name CA")
        if len(model_CA) != len(ref_CA):
            print(f"CA mismatch: {model_file}")
            return None

        rmsd_value = rms.rmsd(model_CA.positions, ref_coords, superposition=True)
        return (model_name, rmsd_value)
    except Exception as e:
        print(f"Error processing {model_file}: {e}")
        return None

# Parallel RMSD computation
model_rmsd = {}
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(compute_rmsd, mn) for mn in model_loglikelihoods.keys()]
    for future in as_completed(futures):
        result = future.result()
        if result:
            model_name, rmsd_value = result
            model_rmsd[model_name] = rmsd_value

# ------------------------------------------------------------
# SCATTER PLOTS PER TIME SLICE (RMSD X, log-likelihood Y)
# ------------------------------------------------------------
fig, axs = plt.subplots(1, len(plot_time_points), figsize=(12, 5), sharey=True)

for i, t in enumerate(plot_time_points):
    # Separate data by source
    md_rmsd = [model_rmsd[mn] for mn in model_rmsd if mn.endswith(f"_{t}") and model_sources[mn] == "MD"]
    md_loglik = [model_loglikelihoods[mn] for mn in model_rmsd if mn.endswith(f"_{t}") and model_sources[mn] == "MD"]

    af_rmsd = [model_rmsd[mn] for mn in model_rmsd if mn.endswith(f"_{t}") and model_sources[mn] == "AF"]
    af_loglik = [model_loglikelihoods[mn] for mn in model_rmsd if mn.endswith(f"_{t}") and model_sources[mn] == "AF"]

    # Scatter plot
    if md_rmsd:
        axs[i].scatter(md_rmsd, md_loglik, alpha=0.6, color='blue', label="MD")
    if af_rmsd:
        axs[i].scatter(af_rmsd, af_loglik, alpha=0.6, color='green', label="ColabFold")

    # Red dashed horizontal line for experimental score
    if t in expt_values:
        axs[i].axhline(expt_values[t], color='red', linestyle='--', linewidth=1, label='cryo-EM')

    axs[i].set_title(f'{t}s', fontsize=18, fontweight='bold')
    #Remove individual X labels
    #axs[i].set_xlabel("RMSD (Å)", fontsize=18, fontweight='bold')

    # After plotting all subplots, set a shared X label
    fig.supxlabel("RMSD (Å)", fontsize=18, fontweight='bold')

    if i == 0:
        axs[i].set_ylabel("Score", fontsize=18, fontweight='bold')
    axs[i].tick_params(axis='both', labelsize=12)
    #axs[i].legend(fontsize=12)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# PRINT TOP 100 LOWEST RMSD STRUCTURES
# ------------------------------------------------------------
if model_rmsd:
    print("\nTop 100 lowest RMSD structures:")
    sorted_rmsd = sorted(model_rmsd.items(), key=lambda x: x[1])  # sort by RMSD ascending

    print(f"{'Rank':<5} {'Model':<40} {'RMSD(Å)':<10} {'Score':<12} {'Source':<8}")
    print("-" * 80)

    for rank, (model_name, rmsd_value) in enumerate(sorted_rmsd[:100], start=1):
        score = model_loglikelihoods.get(model_name, "N/A")
        source = model_sources.get(model_name, "Unknown")
        print(f"{rank:<5} {model_name:<40} {rmsd_value:<10.3f} {score:<12.3f} {source:<8}")
else:
    print("No RMSD values calculated. Cannot display top structures.")
