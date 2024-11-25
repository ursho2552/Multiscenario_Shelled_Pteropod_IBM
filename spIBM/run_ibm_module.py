#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:47:18 2022

@author: ursho
"""
import datetime
import time
import logging
import os

from pathlib import Path
from tqdm import tqdm, trange

import numpy as np

import spIBM.initialization_module as initialization_module
import spIBM.population_module as population_module
import spIBM.parcels_module as parcels_module
import spIBM.coupler_module as coupler_module


from parcels import ErrorCode

def run_ibm_idealized(config_param,my_pteropods,start_gen=0,time_steps=5000,length_function=None,
                      save_population=False,save_abundance=False,ensure_survival=False):
    """This function runs the shelled pteropod IBM using idealized conditions and without advection
    and DVM.

    Keyword arguments:
    config_param -- dataclass object containing the paths for idealized environmental conditions,
                    and model parameters
    my_pteropods -- initial pteropod population
    start_gen -- generation of the initial pteropod population (default: 0)
    time_steps -- number of days to run the IBM (default: 5000)
    length_function -- size of the pteropod (in mm) as a function of time (default: None). For the
                        default value we use the growth function from Wang et al. 2017
    save_population -- boolean flag indicating if the pteropod population of each simulation day
                        should be saved (default: False)
    save_abundance -- boolean flag indicating if the abundance time series should be saved at the
                        end of the simulation (default: False)
    """
    sst_file = config_param.dir_env + config_param.sst_file
    food_file = config_param.dir_env + config_param.food_file

    assert Path(sst_file).is_file(), 'File containing idealized Temperature does not exist'
    assert Path(food_file).is_file(), 'File containing idealized Chlorophyll does not exist'

    daily_sst = np.genfromtxt(sst_file,delimiter=',')
    daily_food = np.genfromtxt(food_file,delimiter=',')

    assert daily_sst.shape[0] == 365, 'Temperature data has the wrong size. Should be 365 days'
    assert daily_food.shape[0] == 365, 'Chlorophyll data has the wrong size. Should be 365 days'

    stage_0 = np.zeros((9, time_steps))
    stage_1 = np.zeros((9, time_steps))

    number_of_individuals = my_pteropods.shape[0]
    current_gen = start_gen
    next_id = number_of_individuals

    if start_gen == 0:
        stage_0[0,0] = number_of_individuals
        stage_0[4,0] = 1.0
    elif start_gen == 1:
        stage_1[0,0] = number_of_individuals
        stage_1[4,0] = 1.0

    current_gen = start_gen
    next_id = number_of_individuals

    #spring generation
    rate_g0_0 = config_param.rate_g0_0
    rate_g0_1 = config_param.rate_g0_1
    rate_g0_2 = config_param.rate_g0_2
    rate_g0_3 = config_param.rate_g0_3

    #overwintering generation
    rate_g1_0 = config_param.rate_g1_0
    rate_g1_1 = config_param.rate_g1_1
    rate_g1_2 = config_param.rate_g1_2
    rate_g1_3 = config_param.rate_g1_3

    mynumeggs = config_param.num_eggs
    delta_err = config_param.delta_err
    day_start = config_param.day_start_initial_eggs
    temperature_0 = config_param.temperature_0
    half_saturation = config_param.half_saturation
    temperature_max = config_param.temperature_max
    arag_optimal = config_param.arag_optimal
    temp_optimal = config_param.temp_optimal

    length_function = length_function or population_module.calculate_growth_fct()

    tbar = trange(1, time_steps, leave=True)
    for i in tbar:
        #define optimal conditions
        my_pteropods[:,13] = 4
        my_pteropods[:,18] = 4
        temperature = daily_sst[(day_start+i)%365]
        food = daily_food[(day_start+i)%365]
        #mortality
        _,my_pteropods = population_module.mortality(my_pteropods,rate_g0_0,rate_g0_1,
                                                            rate_g0_2,rate_g0_3,rate_g1_0,
                                                            rate_g1_1,rate_g1_2,rate_g1_3,
                                                            arag_optimal,temp_optimal,
                                                            ensure_survival=ensure_survival)

        #growth
        my_pteropods = population_module.shell_growth(my_pteropods,length_function,arag=4,arag_x=4,
                                                      temperature=temperature,
                                                      food=food,
                                                      temperature_0=temperature_0,
                                                      half_saturation=half_saturation,
                                                      temperature_max=temperature_max)
        #development
        my_pteropods = population_module.development(my_pteropods,length_function)
        #spawning events
        my_pteropods,next_id,current_gen = population_module.spawning(my_pteropods,current_gen,
                                                                      next_id,num_eggs=mynumeggs,
                                                                      delta_err=delta_err)

        #set food sum to any number that is not zero (Otherwise the particle is recognized as
        # "beached" and removed
        my_pteropods[:,16] = 1
        #accounting
        stage_0[0,i] = np.argwhere((my_pteropods[:,2] == 0) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[1,i] = np.argwhere((my_pteropods[:,2] == 1) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[2,i] = np.argwhere((my_pteropods[:,2] == 2) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[3,i] = np.argwhere((my_pteropods[:,2] == 3) & (my_pteropods[:,1]%2 == 0) & (my_pteropods[:,5] == 1)).shape[0]
        stage_0[4:8,i] = stage_0[:4,i]/np.sum(stage_0[:4,i],axis=0) if np.sum(stage_0[:4,i],axis=0)  > 0 else 0
        tmp = my_pteropods[np.argwhere(my_pteropods[:,1]%2 == 0),3]
        if tmp.size:
            stage_0[8,i] = np.nanmedian(tmp)

        stage_1[0,i] = np.argwhere((my_pteropods[:,2] == 0) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[1,i] = np.argwhere((my_pteropods[:,2] == 1) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[2,i] = np.argwhere((my_pteropods[:,2] == 2) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[3,i] = np.argwhere((my_pteropods[:,2] == 3) & (my_pteropods[:,1]%2 == 1) & (my_pteropods[:,5] == 1)).shape[0]
        stage_1[4:8,i] = stage_1[:4,i]/np.sum(stage_1[:4,i],axis=0) if np.sum(stage_1[:4,i],axis=0) > 0 else 0
        tmp = my_pteropods[np.argwhere(my_pteropods[:,1]%2 == 1),3]
        if tmp.size:
            stage_1[8,i] = np.nanmedian(tmp)
        my_pteropods[:,12] = i

        individuals = np.sum(stage_0[:4,i],axis=0) + np.sum(stage_1[:4,i],axis=0)
        tbar.set_description(f'{individuals} Individuals')
        tbar.refresh

        if save_population:
            if not os.path.exists(config_param.output_dir_initialization):
                os.makedirs(config_param.output_dir_initialization)

            np.savetxt(config_param.output_dir_initialization + config_param.out_ptero_file.format(i),
                        my_pteropods, delimiter=',')

    if save_abundance:
        if not os.path.exists(config_param.output_dir_initialization):
            os.makedirs(config_param.output_dir_initialization)

        np.savetxt(config_param.output_dir_initialization+config_param.gen0_file,
                   stage_0, delimiter=',')
        np.savetxt(config_param.output_dir_initialization+config_param.gen1_file,
                   stage_1, delimiter=',')


        return


def run_ibm_coupled(config_param, pset, fieldset, pclass, kernel, time_mat,
                    next_id, current_gen, length_function=None, start_clock=None,
                    time_threshold=None, dissolution=None, displacement=True, daily_seed=None):
    """This function runs the shelled pteropod IBM using modeled/observed environmental conditions
    and with a defined kernel for movement and interation with the environment.

    Keyword arguments:
    config_param -- dataclass object containing the paths for idealized environmental conditions,
                    and model parameters
    pset -- Ocean Parcels particle object containing the initial population with initialized
            attributes
    fieldset -- Ocean Parcels fieldset object defining the environmental conditions
    pclass -- Ocean Parcels particle class
    kernels -- Ocean Parcels kernel. Defines how the particels move and interact with the
                environment
    time_mat -- array containing the year on the first row, the day on the second row, and the
                number of days after the beginning of the simulation period
    next_id -- the unique identifier for the next ID
    current_gen -- identifier for the current generation
    length_function -- size of the pteropod (in mm) as a function of time (default: None). For the default value
             we use the growth function from Wang et al. 2017
    """

    kernels = pset.Kernel(kernel)
    start_clock = start_clock or time.time()

    time_threshold = time_threshold or 9999

    #spring generation
    rate_g0_0 = config_param.rate_g0_0
    rate_g0_1 = config_param.rate_g0_1
    rate_g0_2 = config_param.rate_g0_2
    rate_g0_3 = config_param.rate_g0_3

    #overwintering generation
    rate_g1_0 = config_param.rate_g1_0
    rate_g1_1 = config_param.rate_g1_1
    rate_g1_2 = config_param.rate_g1_2
    rate_g1_3 = config_param.rate_g1_3

    dir_env = config_param.dir_env
    food_file = config_param.food_file
    daily_food = np.genfromtxt(dir_env+food_file,delimiter=',')
    food_max = max(daily_food)
    food_min = min(daily_food)
    delta_err = config_param.delta_err
    num_eggs = config_param.num_eggs
    half_saturation = config_param.half_saturation
    temperature_0 = config_param.temperature_0
    temperature_max = config_param.temperature_max
    arag_optimal = config_param.arag_optimal
    temp_optimal = config_param.temp_optimal
    size_threshold = config_param.size_threshold

    length_function = length_function or population_module.calculate_growth_fct()

    flag_init = config_param.main_flag

    tbar = trange(time_mat.shape[1], leave=True)
    for i in tbar:

        year = np.squeeze(time_mat[0,i]).astype(int)
        day_counter = np.squeeze(time_mat[2,i].astype(int))

        np.random.seed(seed=daily_seed[day_counter])

        output_dir = f'{config_param.output_dir_simulation}year_{year}_V_{config_param.version}_control_{config_param.control}/'
        output_dir_scratch = f'{config_param.output_dir_simulation_scratch}year_{year}_V_{config_param.version}_control_{config_param.control}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(output_dir_scratch):
            os.makedirs(output_dir_scratch)

        filename_day = f'{output_dir_scratch}JitPtero_Day_{day_counter}'
        if flag_init:
            tbar.set_description('Initializing for the first time')
            tbar.refresh
            pset = coupler_module.prepare_particles(pset,year)
            flag_init = False

        tbar.set_description(f'Day {day_counter}: Advection')
        tbar.refresh
        if config_param.main_flag:
            pset.execute(kernels,runtime=datetime.timedelta(days=1),dt=datetime.timedelta(hours=1.0),\
                output_file=pset.ParticleFile(name=filename_day, outputdt=datetime.timedelta(hours=1.0)),\
                verbose_progress=False,recovery={ErrorCode.ErrorThroughSurface: parcels_module.ReturnToSurface,ErrorCode.ErrorOutOfBounds: parcels_module.PushToWater})

            filename_day_last = f'{output_dir}JitPtero_Day_{day_counter}_reduced.nc'
            filename_day_env = f'{output_dir}JitPtero_Day_{day_counter}_env.nc'

            initialization_module.save_essential_variables(filename_day,filename_day_last,filename_day_env)

        if (time.time() - start_clock)/60/60 > time_threshold:
            logging.warning('Simulation will be restarted from the last day')

            str_sim = 'Hindcast' if 'Hindcast' in config_param.config_file else 'ConstantCO2'
            str_run = str_sim + '_NoX' if 'noextremes' in config_param.config_file else str_sim

            outfile_slurm = f"--output=slurm-Pteropod_Extremes_V_{config_param.version}_C_{config_param.control}_year_{year}_restart_{day_counter}_{str_run}_%j.out"

            os.system('module purge')
            os.system('module load new gcc/4.8.2 python/3.7.1')
            dissolution = 0 if dissolution is None else 1
            os.system(f'sbatch -n 1 --time=24:00:00 --mem-per-cpu=120000 --wrap "python ./main.py --year {year} --version {config_param.version} --control {config_param.control} --restart_day {day_counter} --config_file {config_param.config_file} --time_threshold 23 --dissolution {dissolution} --displacement {displacement}" {outfile_slurm}')
            return day_counter


        reseed_vector = pset.particle_data['reseed_area'][:] == 0
        if reseed_vector.sum() > 0:
            tbar.set_description(f'Day {day_counter}: Reseeding')
            tbar.refresh
            num = reseed_vector.sum()
            buffer = config_param.control == 1
            reseed_file = config_param.reseed_file
            distance_file = config_param.distance_file
            reseed_lon, reseed_lat = initialization_module.random_reseed(num,reseed_file,
                                                                         distance_file,
                                                                         buffer=buffer)

            pset = initialization_module.reseed_particles(pset,reseed_vector,reseed_lon,
                                                          reseed_lat,fieldset,pclass=pclass)

        tbar.set_description(f'Day {day_counter}: Mortality')
        tbar.refresh
        my_data, exposure_weight_factor = coupler_module.convert_to_mat(pset)

        day_vec = np.array([year,config_param.version,day_counter,config_param.control]).astype(int)
        outfile_mort = config_param.outfile_mort

        die_list,my_data = population_module.mortality(my_data, rate_g0_0, rate_g0_1,
                                                       rate_g0_2, rate_g0_3, rate_g1_0,
                                                       rate_g1_1, rate_g1_2, rate_g1_3,
                                                       arag_optimal, temp_optimal,
                                                       weight_factor=exposure_weight_factor,
                                                       day=day_vec, outfile=outfile_mort,
                                                       dissolution=dissolution)

        exposure_weight_factor[die_list] = np.nan
        exposure_weight_factor = exposure_weight_factor[np.isnan(exposure_weight_factor)==False]

        tbar.set_description(f'Day {day_counter}: Deletion')
        tbar.refresh
        pset = coupler_module.get_dead_particles(pset,die_list)

        tbar.set_description(f'Day {day_counter}: Growth')
        tbar.refresh
        if my_data.shape[0] < 1:
            logging.warning('All pteropods are dead')
            return day_counter

        mean_food = np.squeeze(my_data[:,16])
        food_scaled = food_min + (mean_food - 0.05)*(food_max - food_min) / (0.9 - 0.05)
        outfile_growth = config_param.outfile_growth

        arag = np.squeeze(my_data[:,13]).astype(np.float32) if dissolution is None else 4

        arag_x = np.squeeze(my_data[:,18]).astype(np.float32)

        my_data = population_module.shell_growth(my_data,length_function,arag=arag,
                               arag_x=arag_x,
                               temperature=np.squeeze(my_data[:,15]),
                               food=food_scaled, temperature_0=temperature_0,
                               half_saturation=half_saturation, temperature_max=temperature_max,
                               weight_factor=exposure_weight_factor,
                               size_threshold=size_threshold, day=None, outfile=outfile_growth)

        tbar.set_description(f'Day {day_counter}: Development')
        tbar.refresh
        my_data = population_module.development(my_data,length_function)

        #spawning events
        tbar.set_description(f'Day {day_counter}: Spawning')
        tbar.refresh
        my_data,next_id,current_gen = population_module.spawning(my_data,current_gen,next_id,
                                                                 num_eggs=num_eggs,
                                                                 delta_err=delta_err)

        #update pset_ptero
        tbar.set_description(f'Day {day_counter}: Updating')
        tbar.refresh

        pset = coupler_module.update_particleset(my_data,pset,fieldset,pclass,year)

        #save my_Data as csv file
        tbar.set_description(f'Day {day_counter}: Saving')
        tbar.refresh

        config_param.main_flag = True

        tbar.set_description(f'Day {day_counter}: Done')
        tbar.refresh

    return day_counter


def run_physics_only(config_param, pset, kernel, year, total_runtime=3, time_step=1.0,
                     outputdt=None):
    """This function runs the movement and interaction with the environment (kernel) of particles
    without the mortality, growth, development and spawing functions.
    The function return an Ocean Parcels particle object with adapted attributes

    Keyword arguments:
    config_param -- dataclass object containing the paths for idealized environmental conditions,
                    and model parameters
    pset -- Ocean Parcels particle object containing the initial population with initialized
            attributes
    fieldset -- Ocean Parcels fieldset object defining the environmental conditions
    kernels -- Ocean Parcels kernel. Defines how the particels move and interact with the
                environment
    year -- year of the simulation
    total_runtime -- number of days to run the model using only physics and without population
                     dynamics (default: 3 days)
    time_step -- sub-time-step to run the kernel (default: 1 hour)
    outputdt -- time-step at which the physics only run is saved (default: None, and uses dt)
    """

    assert time_step <= 24, "The sub time-step should be smaller than the model time-step (1 day)"

    outputdt = outputdt or time_step
    kernel = kernel = pset.Kernel(kernel)

    if total_runtime < 5:
        logging.warning('If you are running aragonite scenarios with 5 day averages, the initialization should be at least 5 days.')

    physics_only_dir = config_param.output_dir_physics.format(config_param.version)
    if not os.path.exists(physics_only_dir):
        os.makedirs(physics_only_dir)

    for i in tqdm(range(total_runtime), desc='Physics only progress'):

        filename_day = physics_only_dir+config_param.physics_only_file.format(i)

        pset = coupler_module.prepare_particles(pset,year)

        outfile = None if not outputdt else pset.ParticleFile(name=filename_day,
                                                              outputdt=datetime.timedelta(hours=outputdt))


        pset.execute(kernel,runtime=datetime.timedelta(days=1.0),dt=datetime.timedelta(hours=time_step),\
                       recovery={ErrorCode.ErrorThroughSurface: parcels_module.ReturnToSurface,ErrorCode.ErrorOutOfBounds: parcels_module.PushToWater},\
                       verbose_progress=False,output_file=outfile)

    return pset
