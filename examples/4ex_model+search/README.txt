conda activate hdx_model
export LD_LIBRARY_PATH=~/MultiNest/lib:$LD_LIBRARY_PATH

python ../../sampling/dbscan_microstate.py "/Users/kehuang/Documents/GitHub/bayesian_hdx/examples/4ex_model+search/models/*.pdb" #needs full path


python hdx_multinest3.py --microstates ../examples/4ex_model+search/combined_output/microstates.json --representatives ../examples/4ex_model+search/combined_output/representative_microstates.txt --exp_csv ../examples/4ex_model+search/0.5ab.csv --n_live_points 5

python hdx_multinest3.py --microstates ../examples/4ex_model+search/combined_output/microstates.json --representatives ../examples/4ex_model+search/combined_output/representative_microstates.txt --exp_csv ../examples/4ex_model+search/0.5ab.csv --peptides ../examples/4ex_model+search/pep_list.dat --n_live_points 5


mpirun -n 15 python hdx_multinest3.py --microstates ../examples/4ex_model+search/combined_output/microstates.json --representatives ../examples/4ex_model+search/combined_output/representative_microstates.txt --exp_csv ../examples/4ex_model+search/0.5ab.csv --n_live_points 2500 > test25.log

mpirun -n 10 python hdx_multinest3.py --microstates ../examples/4ex_model+search/combined_output/microstates.json --representatives ../examples/4ex_model+search/combined_output/representative_microstates.txt --exp_csv ../examples/4ex_model+search/0.5ab.csv --peptides ../examples/4ex_model+search/pep_list.dat --n_live_points 3000 > test3k_n=2.log 

mpirun -n 10 python hdx_multinest3.py --microstates ../examples/4ex_model+search/combined_output/microstates.json --representatives ../examples/4ex_model+search/combined_output/representative_microstates.txt --exp_csv ../examples/4ex_model+search/0.5ab.csv --peptides ../examples/4ex_model+search/pep_list.dat --n_live_points 3000 > test3k_n=3.log 

mpirun -n 10 python hdx_multinest3.py --microstates ../examples/4ex_model+search/combined_output/microstates.json --representatives ../examples/4ex_model+search/combined_output/representative_microstates.txt --exp_csv ../examples/4ex_model+search/0.5ab.csv --peptides ../examples/4ex_model+search/pep_list.dat --n_live_points 3000 > test3k_n=4.log 

mpirun -n 10 python hdx_multinest3.py --microstates ../examples/4ex_model+search/combined_output/microstates.json --representatives ../examples/4ex_model+search/combined_output/representative_microstates.txt --exp_csv ../examples/4ex_model+search/0.5ab.csv --peptides ../examples/4ex_model+search/pep_list.dat --n_live_points 3000 > test3k_n=5.log 