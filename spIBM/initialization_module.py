#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:50:58 2022
Scripts used to read the user input parameters for the shelled pteropod IBM, and to initialize the
model run
@author: ursho
"""
import os
import logging
import argparse

from dataclasses import dataclass

import yaml

import scipy.stats

import numpy as np
import xarray as xr

from parcels import FieldSet, Field, ParticleSet

@dataclass
class ConfigParameters():
    """Class to define paths and model parameters used in the model

    Keyword arguments:
    None

    """

    config_file: str
    data_path: str
    mesh_file: str

    directory_mort: str
    similarity_file: str

    output_dir_initialization: str
    gen0_file: str
    gen1_file: str
    out_ptero_file: str
    initial_positions_file: str

    reference_abundance_data: str

    output_dir_physics: str
    physics_only_file: str

    output_dir_simulation_scratch: str
    outfile_mort_scratch: str
    outfile_growth_scratch: str
    output_tables_scratch: str

    output_dir_simulation: str
    outfile_mort: str
    outfile_growth: str
    output_tables: str

    dir_env: str
    sst_file: str
    food_file: str

    velocity_file: str
    aragonite_file: str
    aragonite_hind_file: str
    scenario_arag_file: str
    extreme_file: str
    oxygen_file: str
    temperature_file: str
    chlorophyll_file: str
    depth_file: str
    unbeach_file: str
    mask_file: str
    distance_file: str
    reseed_file: str

    velocity_u_variable_name: str
    velocity_v_variable_name: str
    velocity_w_variable_name: str
    aragonite_variable_name: str
    aragonite_hind_variable_name: str
    scenario_arag_variable_name: str
    scenario_extreme_variable_name: str
    temperature_variable_name: str
    oxygen_variable_name: str
    chlorophyll_variable_name: str
    depth_variable_name: str
    mask_variable_name: str
    unbeach_lat_variable_name: str
    unbeach_lon_variable_name: str
    distance_variable_name: str
    reseed_variable_name: str

    lon_name: str
    lat_name: str
    depth_name: str
    time_name: str

    flag_calculate_initial_population: bool
    flag_calculate_initial_positions: bool
    flag_run_physics_only: bool
    main_flag: bool

    #control and version should not be here
    control: int
    start_year: int
    version: int
    restart_day: int
    start_day: int
    num_init: int
    day_start_initial_eggs: int
    seed: int
    rate_g0_0: float
    rate_g0_1: float
    rate_g0_2: float
    rate_g0_3: float
    rate_g1_0: float
    rate_g1_1: float
    rate_g1_2: float
    rate_g1_3: float
    num_eggs: int
    delta_err: float
    half_saturation: float
    temperature_0: float
    temperature_max: float
    arag_optimal: float
    temp_optimal: float
    size_threshold: float


def read_config_files(config_file,config_class=ConfigParameters):
    """This function reads a configuration file, and fills in the attributes of the dataclass with
    the respective entries in the configuratio file

    Keyword arguments:
    config_file -- path to configuration file. Should be a yaml file
    config_class -- dataclass (default: ConfigParameters dataclass defined above)
    """
    assert '.yaml' in config_file.lower(), "The configuration file should be a '.yaml' file"

    with open(config_file, encoding='utf-8') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    config = config_class(**config_list)

    return config


def parse_inputs():
    """This reads user input from the terminal to start running the shelled pteropod IBM

    """

    parser = argparse.ArgumentParser(description="Run shelled pteropod IBM")
    parser.add_argument("--year", required=True, type=int,
                        help="Year of the simulation. Year should coincide with name of file.")
    parser.add_argument("--version", required=True, type=int,
                        help="Version of the run. This integer is used to set the random seed.")
    parser.add_argument("--control", required=False, nargs='?',
                        const=0, default=0, type=int,
                        help="Which scenario is used (0: with extremes; 1: without extremes).")
    parser.add_argument("--restart_day", required=False, nargs='?', const=1, default=1, type=int,
                        help="Determine the restart day for the simulation.")
    parser.add_argument("--config_file", required=False, nargs='?',
                        const="IBM_config_parameters.yaml",
                        default="IBM_config_parameters.yaml", type=str,
                        help="Yaml file containing the paths and parameters needed for the IBM.")
    parser.add_argument("--time_threshold", required=False, nargs='?',
                        const=23,
                        default=23, type=int,
                        help="Number of hours to wait for job before restarting.")
    parser.add_argument("--dissolution", required=False, nargs='?',
                        const=0,
                        default=0, type=int,
                        help="Switch to turn dissolution on (0) or off (1). Default value is 0.")
    parser.add_argument("--displacement", required=False, nargs='?',
                        const=1,
                        default=1, type=int,
                        help="Turn displacement by currents on (1) or off (0). Default value is 1.")

    args = parser.parse_args()

    restart_day = args.restart_day
    main_flag = True if restart_day <= 1 else False
    restart_day = 1 if restart_day < 1 else restart_day
    dissolution = None if args.dissolution == 0 else args.dissolution
    displacement = True if args.displacement > 0 else False

    return args.year, args.version, args.control, restart_day, main_flag, args.config_file, args.time_threshold, dissolution, displacement


def read_environment(config_param, year):
    """This function defines the environmental conditions for the coupled simulation.
    The function defines an Ocean Parcels fieldset with all environmental conditions needed for
    the shelled pteropod IBM to run

    Keyword arguments:
    config_param -- dataclass containing all paths and parameters
    year -- year of the simulation. Should correspond to the names of the files

    """

    mesh_file = config_param.mesh_file

    velocity_file = config_param.velocity_file.format(year)
    aragonite_hind_file = config_param.aragonite_hind_file.format(year)
    aragonite_scenario_file = config_param.scenario_arag_file.format(year)
    aragonite_file = config_param.aragonite_file.format(year)

    extreme_file = config_param.extreme_file.format(year)

    oxygen_file = config_param.oxygen_file.format(year)
    temperature_file = config_param.temperature_file.format(year)
    chlorophyll_file = config_param.chlorophyll_file.format(year)
    depth_file = config_param.depth_file

    unbeach_file = config_param.unbeach_file
    reseed_file = config_param.reseed_file
    distance_file = config_param.distance_file

    filenames = {'U': {'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': velocity_file},
                 'V': {'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': velocity_file},
                 'W': {'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': velocity_file},
                 'arag':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': aragonite_scenario_file},
                 'aragX':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': aragonite_file},
                 'araghind':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': aragonite_hind_file},
                 'extremes_arag':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': extreme_file},
                 'temp':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': temperature_file},
                 'O2':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': oxygen_file},
                 'Chl':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': chlorophyll_file},
                 'Mydepth':{'lon': mesh_file, 'lat': mesh_file, 'depth': depth_file, 'data': depth_file},
                 'mask': {'lon': mesh_file, 'lat': mesh_file, 'data': unbeach_file},
                 'unBeach_lat': {'lon': mesh_file, 'lat': mesh_file, 'data': unbeach_file},
                 'unBeach_lon': {'lon': mesh_file, 'lat': mesh_file, 'data': unbeach_file},
                 'distance': {'lon': mesh_file, 'lat': mesh_file, 'data': distance_file},
                 'reseed_area': {'lon': mesh_file, 'lat': mesh_file, 'data': reseed_file}}

    variables = {'U': config_param.velocity_u_variable_name,
                 'V': config_param.velocity_v_variable_name,
                 'W': config_param.velocity_w_variable_name,
                 'arag': config_param.scenario_arag_variable_name,
                 'aragX': config_param.aragonite_variable_name,
                 'araghind': config_param.aragonite_hind_variable_name,
                 'extremes_arag': config_param.scenario_extreme_variable_name,
                 'temp': config_param.temperature_variable_name,
                 'O2': config_param.oxygen_variable_name,
                 'Chl': config_param.chlorophyll_variable_name,
                 'Mydepth': config_param.depth_variable_name,
                 'mask': config_param.mask_variable_name,
                 'unBeach_lat': config_param.unbeach_lat_variable_name,
                 'unBeach_lon': config_param.unbeach_lon_variable_name,
                 'distance': config_param.distance_variable_name,
                 'reseed_area': config_param.reseed_variable_name}

    lon_name = config_param.lon_name
    lat_name = config_param.lat_name
    depth_name = config_param.depth_name
    time_name = config_param.time_name

    dimensions = {'U': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'V': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'W': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'arag': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'aragX': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'araghind': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'extremes_arag': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'temp': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'O2': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'Chl': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name, 'time': time_name},
                  'Mydepth': {'lon': lon_name, 'lat': lat_name, 'depth': depth_name},
                  'mask': {'lon': lon_name, 'lat': lat_name},
                  'unBeach_lat': {'lon': lon_name, 'lat': lat_name},
                  'unBeach_lon': {'lon': lon_name, 'lat': lat_name},
                  'distance': {'lon': lon_name, 'lat': lat_name},
                  'reseed_area': {'lon': lon_name, 'lat': lat_name}}

    fieldset = FieldSet.from_c_grid_dataset(filenames, variables, dimensions,
                                            allow_time_extrapolation=False)

    fieldset.add_field(Field('bottom_depth', fieldset.Mydepth.depth[-1, :, :],
                             lon=fieldset.Mydepth.grid.lon, lat=fieldset.Mydepth.grid.lat))
    fieldset.add_field(Field('top_depth', fieldset.Mydepth.depth[0, :, :],
                             lon=fieldset.Mydepth.grid.lon, lat=fieldset.Mydepth.grid.lat))

    return fieldset


def define_initial_population(number_of_individuals, start_generation, number_of_attributes=22):
    """This function defines a starting population of eggs

    Keyword arguments:
    number_of_individuals -- number of indidivuals in the starting population
    start_generation -- start of the generation
    number_of_attributes -- number of attributes to characterize each individual

    """

    initial_population = np.random.rand(number_of_individuals, number_of_attributes)
    #ID
    initial_population[:,0] = np.arange(number_of_individuals)
    #generation
    initial_population[:,1] = start_generation
    #stage
    initial_population[:,2] = 0
    #shell_size
    initial_population[:,3] = 0.15
    #days_of_growth
    initial_population[:,4] = 0
    #survive
    initial_population[:,5] = 1
    #num_spawning_events
    initial_population[:,6] = 0
    #ERR
    initial_population[:,7] = np.random.uniform(low=-1, high=1, size=(number_of_individuals))
    #spawned
    initial_population[:,8] = 0
    #Parent_ID
    initial_population[:,9] = -1
    #Parent_shell_size
    initial_population[:,10] = -1
    #time_of_birth
    initial_population[:,11] = -1
    #current_time
    initial_population[:,12] = 0
    #aragonite
    initial_population[:,13] = 0
    #accumulated damage
    initial_population[:,14] = 0
    #average temp
    initial_population[:,15] = 0
    #average food
    initial_population[:,16] = 1
    #shell thickness
    initial_population[:,17] = 0
    #aragonite X scenario
    initial_population[:,18] = np.nan
    #would be dead flag
    initial_population[:,19] = 0
    #shell size extreme
    initial_population[:,20] = 0.15
    #Damage extremes
    initial_population[:,21] = 0

    return initial_population


def define_initial_population_dynamic(number_of_individuals, number_of_attributes,
                                      dictionary_of_values):
    """This function defines a starting population with attributes defined in a dictionary

    Keyword arguments:
    number_of_individuals -- number of indidivuals in the starting population
    number_of_attributes -- number of attributes to characterize each individual
    dictionary_of_values -- dictionary containing values or functions for each attribute
    """

    assert len(dictionary_of_values) == number_of_attributes, "Dictionary must contain values for each attribute"
    assert all([int(k) for k in dictionary_of_values.keys()] == np.arange(number_of_attributes)), "Dictionary keys should be the indeces for the columns (attributes) given as integers"

    #used to get the same population to test different implementations
    #np.random.rand(1500, 17)
    initial_population = np.ones((number_of_individuals, number_of_attributes))

    for key in dictionary_of_values:

        initial_population[:,int(key)] = dictionary_of_values[key]

    return initial_population


def determine_starting_day(output_dir,gen0_file,gen1_file,observations,observations_std,
                           start=None,return_rolling=False):
    """This function determines the starting day given daily simulated abundances and observations
    and range in observed abundances

    Keyword arguments:
    output_dir -- directory of files with modeled abundances
    gen0_file -- file with abundances for the first generation
    gen1_file -- file with abundances for the second generation
    observations -- daily abundance observations
    observations_std -- daily abundance ranges (here the standard deviation is used as an example)
    start -- first day in the modeled abundance that should be considered in the comparison
    """

    stage_0 = np.genfromtxt(output_dir + gen0_file, delimiter=',')
    stage_1 = np.genfromtxt(output_dir + gen1_file, delimiter=',')
    stage_0 = np.nan_to_num(stage_0)
    stage_1 = np.nan_to_num(stage_1)

    cycle1 = stage_1[1:4,:]
    cycle0 = stage_0[1:4,:]
    data = np.sum(cycle1,axis=0) + np.sum(cycle0,axis=0)

    logging.info('Matching to observations')
    best_mean,best_rolling,start_day,max_pearson,max_spearman,min_rmse,outside_range = match_to_observations(data,observations,observations_std,start=start)

    logging.info('The following metrics were found:')
    logging.info(f'{np.round(start_day)}, {np.round(max_spearman,2)}, {np.round(max_pearson,2)}, {np.round(min_rmse,2)}, {np.round(outside_range,2)}')

    logging.info(f'Start day is: {start_day}')
    if return_rolling:
        return start_day, best_mean, best_rolling
    return start_day

def save_essential_variables(filename_day, filename_day_last, filename_day_env,
                             names_keep=['lat', 'lon', 'z', 'temp', 'food', 'oxygen','arag', 'extreme_arag', 'extreme_arag_flag']):

    '''This function removes some attributes from the saved files, as they are only needed
    internally and not for the analysis, or are derived metrics from essential attributes.

    Keyword arguments:
    filename_day -- path to netcdf file to reduce attributes
    filename_day_essential -- path to new file with reduced attributes
    filename_day_env -- path to new file containing all sub-time-steps for the
        environmental conditions
    names_keep -- variables were all sub-time-steps should be stored
     '''


    data_set = xr.open_dataset(filename_day+'.nc')

    #only keep the last time step
    names_delete = ['arrival','departure_from_depth','departure','up','down','down_wings',
                    'max_depth','min_depth','departure','flag_down','flag_up','next_max_depth',
                    'productivity','O2_level','lon_chl_max','lat_chl_max','chl_ascent','chl_max',
                    'depth_chl_max','reseed_lat','reseed_lon','reseed_depth','reseed_flag',
                    'prev_lat','prev_lon','prev_depth','rng_surf','rng_bot']

    data_set_slice = data_set.sel(**{'obs': slice(-1,None)})
    data_set_slice = data_set_slice.drop(names_delete)

    #end new code to save updated pteropods
    data_set_slice.to_netcdf(filename_day_last)

    #remove all except for arag, temp, oxygen, food, z, lat, lon, extreme_arag_flag
    variables = list(data_set.keys())
    for name in names_keep:
        variables.remove(name)

    data_set_env = data_set.drop(variables).sel(**{'obs': slice(1,None)})
    data_set_env.to_netcdf(filename_day_env)

    delete_command = 'rm '+ filename_day + '.nc'
    os.system(delete_command)



def read_attributes_from_file(filename_day_essential,fieldset,pclass):
    """This function reads in the attributes of particels stored as xarray.
    The function is very specific to the project, and should be adapted if the project changes, or
    a dynamic implementation is needed

    Keyword arguments:
    filename_day_essential -- path to file containing all essential information for the particles
    fieldset -- Ocean Parcesl fieldset defining the environmental conditions
    pclass -- Ocean Parcels particle class
    """

    data_set = xr.open_dataset(filename_day_essential)

    time = data_set.time[:,-1].values
    lat = data_set.lat[:,-1].values
    lon = data_set.lon[:,-1].values
    z = data_set.z[:,-1].values
    distance = data_set.distance[:,-1].values
    temp = data_set.temp[:,-1].values
    temp_sum = data_set.temp_sum[:,-1].values
    food = data_set.food[:,-1].values
    food_sum = data_set.food_sum[:,-1].values
    oxygen = data_set.oxygen[:,-1].values
    oxygen_sum = data_set.oxygen_sum[:,-1].values
    arag = data_set.arag[:,-1].values
    arag_sum = data_set.arag_sum[:,-1].values
    damage = data_set.damage[:,-1].values
    shell_thickness = data_set.shell_thickness[:,-1].values
    generation = data_set.generation[:,-1].values
    stage = data_set.stage[:,-1].values
    survive = data_set.survive[:,-1].values
    flag_would_die = data_set.flag_would_die[:,-1].values
    num_spawning_event = data_set.num_spawning_event[:,-1].values
    shell_size = data_set.shell_size[:,-1].values
    extreme_shell_size = data_set.extreme_shell_size[:,-1].values
    extreme_damage = data_set.extreme_damage[:,-1].values

    step_counter = data_set.step_counter[:,-1].values
    step_counter_arag = data_set.step_counter_arag[:,-1].values

    days_of_growth = data_set.days_of_growth[:,-1].values
    err = data_set.ERR[:,-1].values
    spawned = data_set.spawned[:,-1].values
    my_id = data_set.MyID[:,-1].values
    parent_id = data_set.Parent_ID[:,-1].values
    parent_shell_size = data_set.Parent_shell_size[:,-1].values

    extreme_arag_flag = data_set.extreme_arag_flag[:,-1].values
    extreme_arag = data_set.extreme_arag[:,-1].values
    extreme_arag_sum = data_set.extreme_arag_sum[:,-1].values

    reseed_area = data_set.reseed_area[:,-1].values

    max_id = np.max(my_id)+1
    current_gen = np.nanmax(generation[np.squeeze(np.argwhere((stage==3) | (shell_size == max(np.unique(shell_size))))).astype(int)])

    pset = ParticleSet(fieldset=fieldset, pclass=pclass,\
                            time=time,\
                            lat=lat,\
                            lon=lon,\
                            depth=z,\
                            distance=distance,\
                            temp=temp,\
                            temp_sum=temp_sum,\
                            food=food,\
                            food_sum=food_sum,\
                            oxygen=oxygen,\
                            oxygen_sum=oxygen_sum,\
                            arag=arag,\
                            arag_sum=arag_sum,\
                            damage=damage,\
                            shell_thickness=shell_thickness,\
                            generation=generation,\
                            stage=stage,\
                            survive=survive,\
                            flag_would_die=flag_would_die,\
                            num_spawning_event=num_spawning_event,\
                            shell_size=shell_size,\
                            extreme_shell_size=extreme_shell_size,\
                            extreme_damage=extreme_damage,\
                            days_of_growth=days_of_growth,\
                            ERR=err,\
                            spawned=spawned,\
                            MyID=my_id,\
                            Parent_ID=parent_id,\
                            Parent_shell_size=parent_shell_size,\
                            extreme_arag_flag=extreme_arag_flag,\
                            extreme_arag=extreme_arag,\
                            extreme_arag_sum=extreme_arag_sum,\
                            reseed_area=reseed_area,\
                            step_counter=step_counter,\
                            step_counter_arag=step_counter_arag,\
                            lonlatdepth_dtype=np.float32)

    return pset, max_id, current_gen


def reset_particle_attributes(pset,dictionary):
    """This function resets the attributes of a particle set to those provided in a matrix.

    Keyword arguments:
    pset -- Ocean Parcels particleset
    dictionary -- dictionary containing the attributes of pset to change (key) and values to change
    """

    for key in dictionary:
        pset.particle_data[key][:] = dictionary[key]

    return pset

def random_reseed(num, reseed_file, file_distance, buffer=True):
    """
    This function randomly reseeds particles in the EBUS. The probability of reseeding is inversely
    proportional to the distance to the coast.

    Keyword arguments:
    num -- number of particles to reseed
    reseed_file -- path to file containing the reseed area
    file_distance -- path to file containing the distance to the coast
    buffer -- boolean indicating whether to reseed in the buffer area (True) or in the EBUS (False)
    """
    distance_coast = xr.open_dataset(file_distance)
    map_distance = distance_coast.distance_coast.values
    distance_list = np.reshape(map_distance,(-1,1))

    data_set_ebus = xr.open_dataset(reseed_file)
    ebus_mask = np.reshape(data_set_ebus.reseed_area.values,(-1,1))

    lat_list = np.reshape(data_set_ebus.lat_rho.values,(-1,1))
    lon_list = np.reshape(data_set_ebus.lon_rho.values,(-1,1))

    indeces = np.argwhere(ebus_mask == 2) if buffer else np.argwhere(ebus_mask >= 1)

    distance = distance_list[indeces[:,0].squeeze()]

    prob_dist = (1/distance) / np.sum(1/distance)

    choice = np.random.choice(indeces[:,0],int(num),replace=True,p=prob_dist.squeeze())

    lon = lon_list[choice,0].squeeze()
    lat = lat_list[choice,0].squeeze()

    return lon, lat

def reseed_particles(pset, reseed_vector, reseed_lon, reseed_lat, fieldset, pclass):
    """This function reseeds certain particles and keeps all other values equal

    Keyword arguments:
    pset -- Ocean Parcels particleset
    reseed_vector -- boolean vector to determine which particles have to be reseeded
    reseed_lon -- new longitudes
    reseed_lat -- new latitudes
    fieldset -- Ocean Parcels FieldSet
    pclass -- Ocean Parcels particle class
    """

    #replace lon and depth
    bool_vec = reseed_vector.astype(bool)

    time =pset.particle_data['time'][:]
    lat = pset.particle_data['lat'][:]
    lon = pset.particle_data['lon'][:]
    z = pset.particle_data['depth'][:]

    lat[bool_vec.squeeze()] = reseed_lat
    lon[bool_vec.squeeze()] = reseed_lon

    z[bool_vec.squeeze()] -= np.random.uniform(low=1.0, high=20.0, size=reseed_lat.shape)

    distance = pset.particle_data['distance'][:]
    temp = pset.particle_data['temp'][:]
    temp_sum = pset.particle_data['temp_sum'][:]
    food = pset.particle_data['food'][:]
    food_sum = pset.particle_data['food_sum'][:]
    oxygen = pset.particle_data['oxygen'][:]
    oxygen_sum = pset.particle_data['oxygen_sum'][:]
    arag = pset.particle_data['arag'][:]
    arag_sum = pset.particle_data['arag_sum'][:]
    damage = pset.particle_data['damage'][:]
    shell_thickness = pset.particle_data['shell_thickness'][:]
    generation = pset.particle_data['generation'][:]
    stage = pset.particle_data['stage'][:]
    survive = pset.particle_data['survive'][:]
    flag_would_die = pset.particle_data['flag_would_die'][:]
    num_spawning_event = pset.particle_data['num_spawning_event'][:]
    shell_size = pset.particle_data['shell_size'][:]
    extreme_shell_size = pset.particle_data['extreme_shell_size'][:]
    extreme_damage = pset.particle_data['extreme_damage'][:]

    step_counter_arag = pset.particle_data['step_counter_arag'][:]
    step_counter = pset.particle_data['step_counter'][:]

    days_of_growth = pset.particle_data['days_of_growth'][:]
    err = pset.particle_data['ERR'][:]
    spawned = pset.particle_data['spawned'][:]
    my_id = pset.particle_data['MyID'][:]
    parent_id = pset.particle_data['Parent_ID'][:]
    parent_shell_size = pset.particle_data['Parent_shell_size'][:]

    extreme_arag_flag = pset.particle_data['extreme_arag_flag'][:]
    extreme_arag = pset.particle_data['extreme_arag'][:]
    extreme_arag_sum = pset.particle_data['extreme_arag_sum'][:]

    reseed_area = pset.particle_data['reseed_area'][:]

    pset = ParticleSet(fieldset=fieldset, pclass=pclass,\
                            time=time,\
                            lat=lat,\
                            lon=lon,\
                            depth=z,\
                            distance=distance,\
                            temp=temp,\
                            temp_sum=temp_sum,\
                            food=food,\
                            food_sum=food_sum,\
                            oxygen=oxygen,\
                            oxygen_sum=oxygen_sum,\
                            arag=arag,\
                            arag_sum=arag_sum,\
                            damage=damage,\
                            shell_thickness=shell_thickness,\
                            generation=generation,\
                            stage=stage,\
                            survive=survive,\
                            flag_would_die=flag_would_die,\
                            num_spawning_event=num_spawning_event,\
                            shell_size=shell_size,\
                            extreme_shell_size=extreme_shell_size,\
                            extreme_damage=extreme_damage,\
                            days_of_growth=days_of_growth,\
                            ERR=err,\
                            spawned=spawned,\
                            MyID=my_id,\
                            Parent_ID=parent_id,\
                            Parent_shell_size=parent_shell_size,\
                            extreme_arag_flag=extreme_arag_flag,\
                            extreme_arag=extreme_arag,\
                            extreme_arag_sum=extreme_arag_sum,\
                            reseed_area=reseed_area,\
                            step_counter=step_counter,\
                            step_counter_arag=step_counter_arag,\
                            lonlatdepth_dtype=np.float32)

    return pset

def initialize_particles(fieldset, pclass, initial_population, locations):
    """This function initializes a particle set. Function is specific to the project, and should be
    adapted for other projects

    Keyword arguments:
    fieldset -- Ocean Parcels particleset
    pclass -- Ocean Parcels particle class
    initial_population -- Initial values for particel attributes
    locations -- locations of pteorpods, lons in first column, lats in second column, depth in the
                 third column for each pteropod

    """
    assert locations.shape[0] == initial_population.shape[0], "Number of entries in the initial population and the locations is not the same"

    depths = locations[:,2].astype(np.float32)
    lats = locations[:,1].astype(np.float32)
    lons = locations[:,0].astype(np.float32)
    ids = np.arange(initial_population.shape[0])

    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=lons,\
                   lat=lats,depth=depths,\
                   time=0.0,stage=initial_population[:,2],survive=initial_population[:,5],\
                   num_spawning_event=initial_population[:,6], generation=initial_population[:,1],\
                   shell_size=initial_population[:,3], days_of_growth=initial_population[:,4],\
                   ERR=initial_population[:,7], spawned=initial_population[:,8],\
                   Parent_ID=initial_population[:,9], Parent_shell_size=initial_population[:,10],\
                   MyID=ids,extreme_shell_size=initial_population[:,3],lonlatdepth_dtype=np.float32)

    return pset


def moving_average(data_set, window=30):
    """
    Function to calculate the moving average of a data set. The moving average is calculated
    using a window of size window. The window is centered around the data point for which the
    moving average is calculated. The moving average is calculated using a symmetric window.

    Keyword arguments:
    data_set -- data set for which the moving average is calculated
    window -- size of the window used for the moving average calculation
    """

    assert window > 0
    assert window%2 == 0

    num_obs = data_set.shape[0]
    moving_av = np.ones((num_obs,))*np.nan

    before_after = int(window/2)
    start_position = int(window/2)
    end_position = int(data_set.shape[0]-before_after)
    for i in range(start_position,end_position):
        moving_av[i] = np.nanmean(data_set[i-before_after:i+before_after])

    return moving_av

def rmse(predictions, targets):
    """
    Function to calculate the root mean square error between two data sets
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def match_to_observations(data,observations,observations_std,start=None):
    """This function calculates the optimal pattern match up between the data from the model and
    obsevations data. The similarity is calculated based on the Pearson, Spearman correlation
    coefficients, the Manhattan distance, and the range of modeled abundances outside of the range
    of observed abundances. Function returns the mean, rolling mean, the start day, and similarity
    metrics

    Keyword arguments:
    data -- modeled abudances
    observations -- observed abundances
    observations_std -- range of observed abundances
    start -- first day to consider in data for the comparison (default: None, the comparison is
             only done for the last third of data)
    """

    assert len(observations) == len(observations_std), "Observations and the observations_std should have the same size"

    start = start or int(data.shape[0]*2/3)

    std = abs(observations - observations_std)
    min_daily_abundance_unit = (observations - std)/max(observations)
    max_daily_abundance_unit = (observations + std)/max(observations)

    daily_abundance_unit = observations/max(observations)
    max_pearson = 0
    max_spearman = 0
    min_start = start
    min_manhattan = 100000000
    min_outside_range = 100000000
    sum_factors = 0
    corrected_min_start = min_start
    best_mean = np.zeros((365,))
    best_rolling = np.zeros((365,))

    for i in range(400):
        my_data = data[min_start+i:]
        mean_data = np.zeros((365,))
        counter = np.zeros((365,))
        if min_start+i+365*3 < data.shape[0]:
            for j in range(my_data.shape[0]):

                j_recurring = j%365
                mean_data[j_recurring] += my_data[j]
                counter[j_recurring] += 1
        mean_data = mean_data/counter/np.nanmax(mean_data)
        #calculate rolling mean
        rolling_mean = moving_average(np.hstack((mean_data,mean_data,mean_data)),30)[365:365*2]
        rolling_mean_unit = rolling_mean/max(rolling_mean)

        pearson_r = scipy.stats.pearsonr(rolling_mean_unit,daily_abundance_unit)[0]
        spearman_r = scipy.stats.spearmanr(rolling_mean_unit,daily_abundance_unit)[0]

        manhattan = np.sum(abs(rolling_mean_unit-daily_abundance_unit))
        outside_range = len(np.argwhere((rolling_mean_unit > max_daily_abundance_unit) | (rolling_mean_unit < min_daily_abundance_unit)))/len(daily_abundance_unit)

        if sum_factors <  1 - (manhattan/365) + pearson_r + spearman_r - outside_range:
            min_manhattan = manhattan
            min_outside_range = outside_range
            corrected_min_start = min_start + i
            max_pearson = pearson_r
            max_spearman = spearman_r
            sum_factors =  1 - (manhattan/365) + pearson_r + spearman_r - outside_range
            best_mean = mean_data.copy()
            best_rolling = rolling_mean.copy()

    return [best_mean,best_rolling,corrected_min_start,max_pearson,max_spearman,min_manhattan,min_outside_range]
