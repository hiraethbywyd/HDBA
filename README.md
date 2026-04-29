# bayesian_hdx
Fall 2024 Sali Lab Rotation. Conformation ensemble refinement with HDX-MS data. 

## _calculate_protection_factors_
The python script **baker_hubbard_pf.py** is the one I use in the forward model. Earlier in the quarter, I played around with the **wernet_nilsson.py** script, which uses slightly different criteria for defining hydrogen bonds. I've included the jupyter notebooks I made to explore the different methods of defining hydrogen bonds and heavy atom contacts. They might not run perfectly, since I mostly used them to debug the working protection factor script I was using. However, I've included (I believe) all the files needed to run the notebooks in the _extra_files_ folder. 

The **baker_hubbard_pf.py** contains the _estimate_protection_factor_ function used in the forward model. It takes in a text file that contains the file paths to all of the models of a structure. Then, it calculates the average protection protection for all of the models. It outputs a dictionary of residue numbers and average protection factors. 

## _forward_model_

The script **forward_model.py** contains all of the functions to calculate the forward model, including calculating the intrinsic rates from Daniel's paper in 2016. The _calc_incorporated_deuterium_ function is the main forward model function. Currently, it takes in a list of peptides and then searches the full sequence for the indices of that peptide, which may be slow, so maybe we change this to take in a list of indices instead. Also, there are quite a few peptides that aren't found in the full sequence, which might be fixed by a list of peptide indices. 

The execution file is called **run_foward_model.py** and runs the _calc_incorporated_deuterium_ function. Running from the command line to get %D for all unique tryptic peptides: 

```

python run_forward_model.py -d 0.8 -t 0 30 60 3600 14000 100000 -p 7 -temp 300 -l ./peptide_list.txt -f ./example.txt -o ./output.csv 

```

where -d is the deuteration fraction in the buffer, -t are the time points, -p is pH, -temp is the temperature in kelvin, -f is the path to the text file with all of the paths to models you want to include (hydrogens must be added already), -f is a text file containing peptides for which you want to calculate %D, and -o is where you want to save the csv output. **baker_hubbard_pf.py**, **forward_model.py**, and **run_foward_model.py** should all be in the same directory. This forward model is noiseless. 

The **tryptic_peptides.py** script can generate all possible tryptic peptides given a sequence. However, for large proteins, the combination of possible peptides becomes too large to compute, so it is easier to just pass the known list of peptides in the forward model. 

The file **forward_model_noise.py** is incomplete, but it contains functions to add Gaussian noise to the forward model. The **forward_model.ipynb** and **nist_fab.ipynb** notebooks contain some analysis of experimental noise in the NIST-Fab example, and show how the forward model estimates deuterium uptake. The **old_imp.ipynb** shows the performance of the forward model compared to experimental data for NIST Fab using the prior method to calculate protection factors. 

## _likelihood_function_

My attempted implementation of Daniel's score function from his 2016 paper is in **hdx_likelihood_function.py**. The issue is coming from the truncated Gaussian term - right now the difference term erf(X)-erf(Y) is often producing 0, which is causing the likelihood function to be 0. A= -0.1*sigma and B=1.2*sigma, where I calculate sigma to be the standard deviation of the three replicates of a peptide at one time point. Also, Daniel’s code seems to have a slightly different formulation, where the sqrt(2) is replaced with sqrt(pi). 

The **likelihood_function.ipynb** notebook was for my understanding of the likelihood function and error functions. 

## _slides_

Contains two small project updates as PowerPoint slides, as well as my group meeting as a PDF file.

## _synthetic_dataset_

Contains a CSV called **noiseless_full_df.csv** that contains predicted %D for all possible tryptic peptides (**fold_switch_peptides.txt** in _extra_files_) at these parameters: -d 0.8 -t 0 30 60 3600 14000 100000 -p 7 -temp 300. This dataset contains predictions for both states of the model fold switch protein (**stateH.pdb** and **stateU.pdb** in _extra_files_). The column "state" describes if the data corresponds to U state or H state. There is no added noise to these values and the dataset represents full coverage of the protein. 
 
