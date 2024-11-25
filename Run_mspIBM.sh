#!/bin/bash

module purge
module load stack/.2024-06-silent  intel-oneapi-compilers/2023.2.0
module load python/3.8.18-c3ikxoi

source /cluster/home/ursho/kryo_work/ursho/py3_euler/bin/activate

#provide some paramters to define the model run, control and version (for reproducibility), runtime threshold (model run is restarted from the last available day if the runtime threshold is reached), remove the effects of dissolution (1: True, 0: False)
control=1
version=5
time_threshold=23
dissolution_flag=0

#provide the start and end year or only a subset
for i in {2011..2011}
	do

       sbatch -n 1 --time=24:00:00 --mem-per-cpu=110000 --wrap "python ./main.py --year ${i} --version $version --control $control --config_file configuration_files/IBM_config_parameters_example.yaml --time_threshold $time_threshold --dissolution $dissolution_flag" --output=slurm-Pteropod_Extremes_V_${version}_C_${control}_year_${i}_Hindcast_%j.out

        done


