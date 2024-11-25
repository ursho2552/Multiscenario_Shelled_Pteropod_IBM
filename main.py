#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:14:12 2022

Main file to calculate results for publication

Multiscenario Shelled Pteropod IBM

The impact of aragonite saturation variability on shelled pteropods: An attribution study in the
California Current System. Urs Hofmann Elizondo, Meike Vogt, Nina Bednarsek, Matthias Münnich, and Nicolas Gruber, 2024.
Global Change Biology

@author: Urs Hofmann Elizondo
"""

import sys

import csv
import os
import datetime
import logging
import time

from dataclasses import asdict

import numpy as np

sys.path.insert(1,"/nfs/kryo/work/ursho/PhD/Projects/Pteropod_Extremes/Scripts/")
import spIBM
import project_functions


def main():
    '''
    This is the main function that runs the IBM, and prepares all the
    steps before and after the IBM is run.
    '''
    # =========================================================================
    # Read in user input from terminal
    # =========================================================================
    year, version, control, restart_day, main_flag, config_file, time_threshold, dissolution, displacement = spIBM.parse_inputs()

    # =========================================================================
    # Read YAML file with all parameter values and fill in the control and version field
    # =========================================================================
    my_config = spIBM.read_config_files(config_file)
    my_config.control = control
    my_config.version = version
    my_config.restart_day = restart_day
    my_config.main_flag = main_flag
    my_config.config_file = config_file

    if '{}' in my_config.output_dir_initialization:
        my_config.output_dir_initialization = my_config.output_dir_initialization.format(version)
        if '{}' in my_config.output_dir_initialization:
            my_config.output_dir_initialization = my_config.output_dir_initialization.format(control)

    if '{}' in my_config.output_dir_physics:
        my_config.output_dir_physics = my_config.output_dir_physics.format(version)
        if '{}' in my_config.output_dir_physics:
            my_config.output_dir_physics = my_config.output_dir_physics.format(control)

    np.random.seed(seed=my_config.seed + version)
    daily_seed = np.random.randint(low=0, high=4000, size=1100).astype(int)

    # =========================================================================
    # Read environment
    # =========================================================================
    fieldset = spIBM.read_environment(my_config, year)

    # =========================================================================
    # Calculate initial idealized population
    # =========================================================================
    if my_config.main_flag:
        if not os.path.exists(my_config.output_dir_initialization):
            os.makedirs(my_config.output_dir_initialization)
        np.random.seed(seed=my_config.seed)

        number_of_individuals = 1500
        start_generation = 0
        initial_population_dictionary = {'0': np.arange(number_of_individuals),
                            '1': start_generation,
                            '2': 0,
                            '3': 0.15,
                            '4': 0,
                            '5': 1,
                            '6': 0,
                            '7': np.random.uniform(low=-1, high=1, size=number_of_individuals),
                            '8': 0,
                            '9': -1,
                            '10': -1,
                            '11': -1,
                            '12': 0,
                            '13': 0,
                            '14': 0,
                            '15': 0,
                            '16': 1,
                            '17': 0,
                            '18': np.nan,
                            '19': 0,
                            '20': 0.15,
                            '21': 0}


        my_pteropods = spIBM.define_initial_population_dynamic(number_of_individuals=number_of_individuals,
                                                               number_of_attributes=22,
                                                               dictionary_of_values=initial_population_dictionary)

        if my_config.flag_calculate_initial_population:
            spIBM.run_ibm_idealized(my_config, my_pteropods, start_gen=0, time=5000,
                                    L_t=None, save_population=True, save_abundance=True,
                                    ensure_survival=True)

    # =========================================================================
    # Determine starting day given the abundances calculated above
    # This part requires external validation data (e.g. from MAREDAT)
    # =========================================================================
    ref_data_file = my_config.reference_abundance_data
    daily_abundance_maredat, std_abundance_maredat = project_functions.get_daily_maredat_obs(ref_data=ref_data_file)

    output_dir = my_config.output_dir_initialization
    gen0_file = my_config.gen0_file
    gen1_file = my_config.gen1_file

    my_config.start_day = spIBM.determine_starting_day(output_dir, gen0_file, gen1_file,
                                                       daily_abundance_maredat,
                                                       std_abundance_maredat, start=None)

    # =========================================================================
    # Read initial idealized population at the start day
    # =========================================================================
    initial_population = np.genfromtxt(f'{output_dir}/Pteropods_Day_{my_config.start_day}.csv',
                                       delimiter=',')

    factor = 3000//initial_population.shape[0] + 1

    initial_population = np.repeat(initial_population,factor,axis=0)
    num_init = initial_population.shape[0]

    initial_population = np.hstack((initial_population,initial_population[:,14].reshape((-1,1))))
    #reset IDs --> this is now done in the initialization function
    initial_population[:,0] = np.arange(num_init)

    # =========================================================================
    # Get the initial random positions (only calculate once for the first year)
    # =========================================================================
    if my_config.main_flag:

        distance_file = my_config.distance_file
        outfile = my_config.output_dir_initialization+my_config.initial_positions_file
        reseed_file = my_config.reseed_file

        np.random.seed(seed=my_config.version*5)

    #Ideally this is done once for the very first year, then only read from file later on
        if my_config.flag_calculate_initial_positions:
            latlon_list = project_functions.get_initial_positions_fct_distance_coast(num=num_init,
                                                                                     reseed_file=reseed_file,
                                                                                     distance_file=distance_file,
                                                                                     outfile=outfile)

        #Use a while loop in case many processes are trying to access the same file at the same time
        shape_file = 0
        while shape_file == 0:
            with open(outfile, mode='r', encoding='utf-8') as file:
                latlon_list = np.array([np.array(line).astype(float) for line in csv.reader(file)])
            shape_file = latlon_list.shape[0] if latlon_list.shape[0] > 0 else 0

    # =========================================================================
    # Initialize particles and kernel
    # =========================================================================
        pclass = spIBM.PteropodParticle
        pset_ptero = spIBM.initialize_particles(fieldset,pclass,initial_population,latlon_list)

        kernel = spIBM.pteropod_kernel

    # =========================================================================
    # Run physics only initialization, and reset times
    # =========================================================================
        total_runtime = 5
        if my_config.flag_run_physics_only:
            pset_ptero = spIBM.run_physics_only(my_config, pset_ptero, kernel,
                                                year, total_runtime=total_runtime, dt=1.0,
                                                outputdt=1.0)

        #always read from file. On the first year calculate the value and then read from file
        my_file = my_config.output_dir_physics.format(my_config.version) + my_config.physics_only_file.format(total_runtime - 1)
        pset_ptero, _, current_gen = spIBM.read_attributes_from_file(filename_day_essential=my_file,
                                                                     fieldset=fieldset,pclass=pclass)

        reset_attributes_dictionary = {'time': 0}
        pset_ptero = spIBM.reset_particle_attributes(pset_ptero,reset_attributes_dictionary)

    # =========================================================================
    # Run coupled model
    # =========================================================================
        logging.warning('Starting simulation')
        next_id = max(initial_population[:,0])+1
        logging.warning(f'Shape of initial population: {initial_population.shape}')
        logging.warning(f'Next Id {next_id}')
        current_gen = np.nanmax(initial_population[np.squeeze(np.argwhere((initial_population[:,2]==3) | (initial_population[:,3] == max(np.unique(initial_population[:,3]))))).astype(int),1])

    else:

        logging.warning('Reloading saved file')
        output_dir = f'{my_config.output_dir_simulation}year_{year}_V_{my_config.version}_control_{my_config.control}/'

        filename_day_essential = f'{output_dir}JitPtero_Day_{restart_day}_reduced.nc'

        pclass = spIBM.PteropodParticle

        pset_ptero, next_id, current_gen = spIBM.read_attributes_from_file(filename_day_essential,
                                                                           fieldset, pclass)

        if not displacement:
            kernel = spIBM.pteropod_no_displacement_kernel

        else:
            kernel = spIBM.pteropod_kernel

        gen0_file = my_config.gen0_file
        gen1_file = my_config.gen1_file

    #define time for which the model should work
    day_start = datetime.date(year,1,1) + datetime.timedelta(days=my_config.restart_day-1)
    day_end = datetime.date(year,12,31)
    time_mat = np.empty((3,(day_end - day_start).days))
    for i in range(time_mat.shape[1]):
        time_mat[0,i] = (day_start + datetime.timedelta(days=i)).year
        time_mat[1,i] = (day_start + datetime.timedelta(days=i)).day
        time_mat[2,i] = (day_start + datetime.timedelta(days=i)).timetuple().tm_yday

    start_clock = time.time()

    #add random seed for each day
    if not displacement:
        kernel = spIBM.pteropod_no_displacement_kernel

    day_counter = spIBM.run_ibm_coupled(my_config, pset_ptero, fieldset,
                                        pclass, kernel, time_mat, next_id,
                                        current_gen, length_function=None,
                                        start_clock=start_clock,
                                        time_threshold=time_threshold,
                                        dissolution=dissolution,
                                        displacement=displacement,
                                        daily_seed=daily_seed)

    logging.warning(f'{day_counter} days simulated')
    logging.warning('Successfully completed')
    # =========================================================================
    # Save model parameters used for the year
    # =========================================================================
    parameters_dict = asdict(my_config)
    parameter_file = f'{my_config.output_tables}/Parameters_{year}_lastday_{day_counter}.csv'

    if not os.path.exists(my_config.output_tables):
        os.makedirs(my_config.output_tables)
    with open(parameter_file, 'w', encoding='utf-8') as file:
        writer = csv.DictWriter(file, parameters_dict.keys())
        writer.writeheader()
        writer.writerow(parameters_dict)

    sys.exit()

# ======================
# Main Function
# ======================
if __name__ in "__main__":
    logging.basicConfig(level=logging.WARNING)

    main()

