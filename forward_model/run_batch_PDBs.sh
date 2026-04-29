#!/bin/bash

# Directory containing PDBs
PDB_DIR="/Users/kehuang/Documents/projects/nsp2/analysis/models/colabfold"

# Other fixed paths
DUAL_FILE="/Users/kehuang/Documents/projects/nsp2/analysis/hdx/nsp2_hdx.csv"
OUTPUT_DIR="/Users/kehuang/Documents/projects/nsp2/analysis/hdx"
PDBLIST="$OUTPUT_DIR/pdblist.dat"

# Parameters
D_VAL=0.85
TIMES="30.0 60.0 180.0 600.0 1800.0 3600.0 7200.0"
P_VAL=7
TEMP=300

# Iterate over all PDB files in directory
for pdb in "$PDB_DIR"/*.pdb; do
    # Update pdblist.dat with the current PDB
    echo "$pdb" > "$PDBLIST"

    # Set output file for this PDB
    out_csv="$OUTPUT_DIR/$(basename "$pdb" .pdb)_crystal.csv"
    log_file="$OUTPUT_DIR/$(basename "$pdb" .pdb).log"

    # Run the python command
    python run_forward_mode_mono.py -d $D_VAL -t $TIMES -p $P_VAL -temp $TEMP --dual_file "$DUAL_FILE" "$PDBLIST" -o "$out_csv" > "$log_file"

    # Parse the log and print results
    echo "Results for $(basename "$pdb")"
    grep -E "Time|TOTAL|AVG" "$log_file" | awk -F'=' '
        /Time/ { 
            gsub(/•|s| /,"",$1); 
            printf "%s %s\n", $1, $2
        }
        /TOTAL/ {print "Total: " $2}
        /AVG/ {print "Avg: " $2}
    '
    echo "------------------------------"
done
