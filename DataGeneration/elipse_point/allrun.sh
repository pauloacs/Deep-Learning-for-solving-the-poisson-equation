#!/bin/bash

python3 make_dataset.py #creates the blockMeshDict for each simulation and places it in the respective folder in simulation_data (the folder must exist)

bash sim_cmd.sh #runs all the cases present in simulation_data folder

python3 dl_data_generation.py #extracts Ux Uy T Ma > all_data.hdf5


#RUN WITH ---> bash allrun.sh 
#use python instead of python3 if necessary
