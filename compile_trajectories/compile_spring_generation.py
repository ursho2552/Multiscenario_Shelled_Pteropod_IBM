#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:47:47 2022

Script used to extract individuals of the spring generation, and their
attributes throughout their life-time

The script uses the following five scenarios (1984-2019):
    1. NoDiss:= No dissolution scenario used as baseline
    2. ConstCo2 := Hindcast simulation with constant CO2 forcing
    3. ConstCO2_NoX:= Hindcast simulation with constant CO2 forcing and no
       extreme events
    4. Hindcast_NoX:= Hindcast simulation without extreme events
    5. Hindcast:= Hindcast simulation

The first scenario is always less damaging than the other three. The hindcast
contains the natural variability component (N), the long-term trend (T), and
the extremes (E)

The constCO2 scenario contains the N and E without T. The constantCO2_NoX
scenario contains the N without T or E. The Hindcast_NoX scenario contains the
N, T without E.

The NoDiss scenario is alsow the baseline of the multi-verse simulation:
    - The simulation with scenario 1 & 2 gives the effect N, and the effect of
      N on E
    - The simulation with scenario 1 & 3 gives the effect of N
    - The simulation with scenrio 1 & 4 gives the effect of N and T
    - The simulation with sceario 1 & 5 gives the effect of N, T, and E

The cross-comparison between runs can then be used to isolate the effects of N,
T, and E. In addition, we can separate the effect of N on E and of T on E.


@author: ursho
"""

import os
import xarray as xr
import numpy as np
import time
import calendar
from tqdm import trange
from multiprocessing import Process
import logging

def determine_array_size(year,directory,folder,file_attrs, outdir):

    final_day = 366 if calendar.isleap(year) else 365
    max_ID = 0
    ref_ID = None
    day_min = None

    my_dir = directory + folder.format(year)

    day_trange = trange(1,final_day)
    for day in day_trange:

        file_ptero = my_dir + file_attrs.format(day,'reduced')
        if os.path.exists(file_ptero):

            ds_ptero = xr.open_dataset(file_ptero)
            IDs = ds_ptero.MyID.values
            generation = ds_ptero.generation.values

            possible_IDs = IDs[generation%2==0]

            if possible_IDs.shape[0]>0:

                ref_ID = ref_ID if ref_ID else possible_IDs.min()
                day_min = day_min if day_min else day
                day_max = day
                max_ID = max(max_ID,possible_IDs.max())

    np.save(outdir+f'obs_per_year_{year}.npy', np.array([year, day_min, day_max, max_ID, ref_ID]))
    return

def zombie_turning_points(vector,longevity):
    '''
    Function sets position to 1 if the pteropod would have died, 0 if it would have survived,
    -1 if it died in all scenarios, and nan afterwards. Thus, a 1 flags the
    turning point for a pteropod
    '''
    current_value = 1
    copy_vector = vector.copy()
    for i,val in enumerate(copy_vector):

        if val == current_value:
            current_value += 1
            if i == 0:
                copy_vector[i] = 1
            else:
                #put the previous step to one, since the "turn" was due to the conditions of the previous day
                copy_vector[i-1] = 1
                copy_vector[i] = 0
        else:
            copy_vector[i] = 0

        if i == longevity:
            copy_vector[i] = -1
            copy_vector[i+1::] = np.nan
            return copy_vector

    return copy_vector

def find_impacted_individuals(year, day_min, day_max, directory_NTE, folder, file_attrs):

    final_day = 366 if calendar.isleap(year) else 365

    day_trange = trange(int(day_min), int(day_max+1), leave=True)
    for day in day_trange:

        # define file
        file_ptero_NTE = directory_NTE + folder.format(year) + file_attrs.format(day,'reduced')

        # load dataset
        ds_ptero_NTE = xr.open_dataset(file_ptero_NTE)

        #extract ID attribute, and step counter for aragonite
        myIDs = ds_ptero_NTE.MyID
        step_counter_arag = ds_ptero_NTE.step_counter_arag

        #calculate the average undersaturation
        arag_sum_NTE = xr.where(step_counter_arag>0, ds_ptero_NTE.extreme_arag_sum/step_counter_arag, 1.5)
        arag_sum_NTE = xr.where(step_counter_arag==0, 1.5, arag_sum_NTE)

        mask_corrosive = xr.where(arag_sum_NTE<1.5, True, False).squeeze()


        #get IDs that experience corrosive conditions
        if mask_corrosive.sum() > 0:
            IDs_corrosive = myIDs[mask_corrosive].values.squeeze() if day==day_min else np.union1d(IDs_corrosive, myIDs[mask_corrosive].values.squeeze())

    return IDs_corrosive

def compile_observations(year, directory_N, directory_NE, directory_NT,
                         directory_NTE ,folder,file_attrs, max_ID, ref_ID ,
                         day_min, day_max, IDs_corrosive):
    '''
    Function compiles the differences between the scenarios.

    Keyword arguments:
    year -- specifies the year that will be summarized/compiled
    directory_X -- paths to each scenario
    folder -- folder name containing the data
    file_attrs -- file name containing the data
    extremes_only -- flag to determine if only the extremes should be considered
    num_traj_par -- vector containing the maximum ID and minimum ID in the entire population

    returns compiled data in matrix form
    '''

    num_traj = int(max_ID-ref_ID)+1
    spring_IDs = np.arange(ref_ID,max_ID+1)

    # ================================================
    # Allocate memory for compiled data
    # ================================================
    trajectories_extreme = np.ones((num_traj,300))*np.nan
    trajectories_corrosive = np.ones((num_traj,300))*np.nan

    trajectories_size_ref = np.ones((num_traj,300))*np.nan
    trajectories_size_NTE = np.ones((num_traj,300))*np.nan
    trajectories_damage_ref = np.ones((num_traj,300))*np.nan
    trajectories_damage_NTE = np.ones((num_traj,300))*np.nan

    trajectories_spawned = np.ones((num_traj,300))*np.nan
    trajectories_would_die_N = np.ones((num_traj,300))*np.nan
    trajectories_would_die_NE = np.ones((num_traj,300))*np.nan
    trajectories_would_die_NT = np.ones((num_traj,300))*np.nan
    trajectories_would_die_NTE = np.ones((num_traj,300))*np.nan
    trajectories_stage = np.ones((num_traj,300))*np.nan

    trajectories_doy = np.ones((num_traj,300))*np.nan
    trajectories_lat = np.ones((num_traj,300))*np.nan
    trajectories_lon = np.ones((num_traj,300))*np.nan
    trajectories_depth = np.ones((num_traj,300))*np.nan
    trajectories_distance = np.ones((num_traj,300))*np.nan

    trajectories_temp = np.ones((num_traj,300))*np.nan
    trajectories_chl = np.ones((num_traj,300))*np.nan
    trajectories_aragonite = np.ones((num_traj,300))*np.nan
    trajectories_aragonite_ref = np.ones((num_traj,300))*np.nan
    longevity = np.ones((num_traj,))*np.nan

    trajectories_arag_X_mean = np.ones((num_traj,300))*np.nan
    trajectories_arag_NoX_mean = np.ones((num_traj,300))*np.nan

    trajectories_duration_extremes = np.ones((num_traj,300))*np.nan
    trajectories_duration_corrosive = np.ones((num_traj,300))*np.nan
    # ===============================================
    # Loop over each day
    # ================================================
    day_trange = trange(int(day_min), int(day_max+1), leave=True)
    for day in day_trange:

        # ================================================
        # Define paths to scenarios
        # ================================================
        file_ptero_N = directory_N + folder.format(year) + file_attrs.format(day,'reduced')
        file_ptero_NE = directory_NE + folder.format(year) + file_attrs.format(day,'reduced')
        file_ptero_NT = directory_NT + folder.format(year) + file_attrs.format(day,'reduced')
        file_ptero_NTE = directory_NTE + folder.format(year) + file_attrs.format(day,'reduced')

        file_env_NTE = directory_NTE + folder.format(year) + file_attrs.format(day,'env')

        flag_files_exist = os.path.exists(file_ptero_N) * os.path.exists(file_ptero_NE) * os.path.exists(file_ptero_NT) * os.path.exists(file_ptero_NTE)
        if flag_files_exist:
            # =================================================================
            # Load reduced and env data for each simulation set
            # =================================================================
            ds_ptero_N = xr.open_dataset(file_ptero_N)
            ds_ptero_NE = xr.open_dataset(file_ptero_NE)
            ds_ptero_NT = xr.open_dataset(file_ptero_NT)
            ds_ptero_NTE = xr.open_dataset(file_ptero_NTE)
            ds_env_NTE = xr.open_dataset(file_env_NTE)

            # =================================================================
            # get IDs and produce indeces for trajectory tables
            # comm1 are all the indeces in common of the current day and
            # =================================================================
            index_ID = ds_ptero_NTE.MyID.values.astype(int)

            start_time = time.time()

            indeces, comm1, comm2 = np.intersect1d(index_ID, spring_IDs, return_indices=True)

            logging.info(f'Intersection done in {time.time()-start_time} seconds')
            # =================================================================
            # produce indeces for storing data
            # =================================================================
            indeces_store = index_ID - int(ref_ID)

            # =================================================================
            # get information whether the individuals died in one scenario
            # =================================================================
            start_time = time.time()

            zombie_N = ds_ptero_N.flag_would_die.values[comm1]
            zombie_NE = ds_ptero_NE.flag_would_die.values[comm1]
            zombie_NT = ds_ptero_NT.flag_would_die.values[comm1]
            zombie_NTE = ds_ptero_NTE.flag_would_die.values[comm1]

            logging.info(f'zombie done in {time.time()-start_time} seconds')
            # =================================================================
            # determine if individuals exerienced extremes on the current day
            # =================================================================
            start_time = time.time()

            arag_X_flag = ds_env_NTE.extreme_arag_flag
            arag_X = ds_env_NTE.extreme_arag
            arag_NoX = ds_env_NTE.arag

            mean_arag_X = arag_X.mean(dim='obs', skipna=True)
            mean_arag_NoX = arag_NoX.mean(dim='obs', skipna=True)

            step_counter_arag = ds_ptero_NTE.step_counter_arag

            mask_extremes = xr.where(arag_X_flag == 1, True, False).squeeze()
            duration_extremes = mask_extremes.sum(dim='obs')
            mask_corrosive = xr.where(arag_X < 1.5, True, False).squeeze()
            duration_corrosive = mask_corrosive.sum(dim='obs')

            mask_extremes_sum = xr.where(duration_extremes > 0, 1, 0).values[comm1]

            logging.info(f'corrosiv done in {time.time()-start_time} seconds')
            # =================================================================
            # determine if individuals experienced corrosive conditions on the
            # current day
            # =================================================================
            start_time = time.time()

            arag_sum_NTE = xr.where(step_counter_arag>0, ds_ptero_NTE.extreme_arag_sum/step_counter_arag, 1.5)
            arag_sum_NTE = xr.where(step_counter_arag==0, 1.5, arag_sum_NTE) #this feels redundant XD

            mask_corrosive = xr.where(arag_sum_NTE < 1.5, 1, 0).values[comm1]

            logging.info(f'arag done in {time.time()-start_time} seconds')
            # =================================================================
            # determine if individuals produced eggs
            # =================================================================
            start_time = time.time()

            num_spawning_event = ds_ptero_NTE.num_spawning_event.values[comm1]

            logging.info(f'spawning done in {time.time()-start_time} seconds')
            # =================================================================
            # determine life-stage
            # =================================================================
            start_time = time.time()

            stage = ds_ptero_NTE.stage.values[comm1]

            logging.info(f'stage done in {time.time()-start_time} seconds')
            # =================================================================
            # Get location of individuals
            # =================================================================
            start_time = time.time()

            days_of_growth_all = ds_ptero_NTE.days_of_growth.values.astype(int)[comm1]
            lat = ds_ptero_NTE.lat.values[comm1]
            lon = ds_ptero_NTE.lon.values[comm1]
            depth = ds_ptero_NTE.z.values[comm1]
            distance = ds_ptero_NTE.distance.values[comm1]

            step_counter_arag = ds_ptero_NTE.step_counter_arag.values[comm1]
            temperature = ds_ptero_NTE.temp_sum.values[comm1]/24
            chlorophyll = ds_ptero_NTE.food_sum.values[comm1]/24
            aragonite_avg = ds_ptero_NTE.extreme_arag_sum.values[comm1]/step_counter_arag
            aragonite = np.where(step_counter_arag>0, aragonite_avg, 1.5)

            aragonite_avg = ds_ptero_NTE.arag_sum.values[comm1]/step_counter_arag
            aragonite_ref = np.where(step_counter_arag>0, aragonite_avg, 1.5)

            L_base = ds_ptero_NTE.shell_size.values[comm1].squeeze()
            L_NTE = ds_ptero_NTE.extreme_shell_size.values[comm1].squeeze()
            damage_base = ds_ptero_NTE.damage.values[comm1].squeeze()
            damage_NTE = ds_ptero_NTE.extreme_damage.values[comm1].squeeze()

            logging.info(f'location done in {time.time()-start_time} seconds')
            # =================================================================
            # write data at the appropriate position
            # =================================================================

            start_time = time.time()
            trajectories_extreme[indeces_store[comm1], days_of_growth_all] = mask_extremes_sum.reshape((-1,1))
            trajectories_corrosive[indeces_store[comm1], days_of_growth_all] = mask_corrosive.reshape((-1,1))

            trajectories_spawned[indeces_store[comm1], days_of_growth_all] = num_spawning_event.reshape((-1,1))
            trajectories_would_die_N[indeces_store[comm1], days_of_growth_all] = zombie_N.reshape((-1,1))
            trajectories_would_die_NE[indeces_store[comm1], days_of_growth_all] = zombie_NE.reshape((-1,1))
            trajectories_would_die_NT[indeces_store[comm1], days_of_growth_all] = zombie_NT.reshape((-1,1))
            trajectories_would_die_NTE[indeces_store[comm1], days_of_growth_all] = zombie_NTE.reshape((-1,1))
            trajectories_stage[indeces_store[comm1], days_of_growth_all] = stage.reshape((-1,1))

            trajectories_doy[indeces_store[comm1], days_of_growth_all] = day#days_of_growth_all.reshape((-1,1))
            trajectories_lat[indeces_store[comm1], days_of_growth_all] = lat.reshape((-1,1))
            trajectories_lon[indeces_store[comm1], days_of_growth_all] = lon.reshape((-1,1))
            trajectories_depth[indeces_store[comm1], days_of_growth_all] = depth.reshape((-1,1))
            trajectories_distance[indeces_store[comm1], days_of_growth_all] = distance.reshape((-1,1))

            trajectories_temp[indeces_store[comm1], days_of_growth_all] = temperature.reshape((-1,1))
            trajectories_chl[indeces_store[comm1], days_of_growth_all] = chlorophyll.reshape((-1,1))
            trajectories_aragonite[indeces_store[comm1], days_of_growth_all] = aragonite.reshape((-1,1))
            trajectories_aragonite_ref[indeces_store[comm1], days_of_growth_all] = aragonite_ref.reshape((-1,1))
            longevity[indeces_store[comm1],] = days_of_growth_all.reshape((-1,1))

            trajectories_size_ref[indeces_store[comm1], days_of_growth_all] = L_base.reshape((-1,1))
            trajectories_size_NTE[indeces_store[comm1], days_of_growth_all] = L_NTE.reshape((-1,1))
            trajectories_damage_ref[indeces_store[comm1], days_of_growth_all] = damage_base.reshape((-1,1))
            trajectories_damage_NTE[indeces_store[comm1], days_of_growth_all] = damage_NTE.reshape((-1,1))

            trajectories_arag_X_mean[indeces_store[comm1], days_of_growth_all] = mean_arag_X.values[comm1].reshape((-1,1))
            trajectories_arag_NoX_mean[indeces_store[comm1], days_of_growth_all] = mean_arag_NoX.values[comm1].reshape((-1,1))

            trajectories_duration_extremes[indeces_store[comm1], days_of_growth_all] = duration_extremes.values[comm1].reshape((-1,1))
            trajectories_duration_corrosive[indeces_store[comm1], days_of_growth_all] = duration_corrosive.values[comm1].reshape((-1,1))

            logging.info(f'Saving done in {time.time()-start_time} seconds')


    return longevity, trajectories_extreme, trajectories_corrosive, trajectories_spawned, \
           trajectories_would_die_N, trajectories_would_die_NE, trajectories_would_die_NT, \
           trajectories_would_die_NTE, trajectories_stage, trajectories_doy, \
           trajectories_lat, trajectories_lon, trajectories_depth, trajectories_distance, \
           trajectories_temp, trajectories_chl, trajectories_aragonite, trajectories_aragonite_ref, \
           trajectories_size_ref, trajectories_size_NTE, trajectories_damage_ref, trajectories_damage_NTE, \
           trajectories_duration_extremes, trajectories_duration_corrosive, trajectories_arag_X_mean, trajectories_arag_NoX_mean



def main(year, max_ID, ref_ID, day_min, day_max, directory_N, directory_NE, directory_NT, directory_NTE, folder, file_attrs, outdir):

    logging.info(year, max_ID, ref_ID)
    year = int(year)

    IDs_corrosive = find_impacted_individuals(year, day_min, day_max, directory_NTE, folder, file_attrs)


    longevity, extremes, corrosive, spawned, would_die_N,\
    would_die_NE, would_die_NT, would_die_NTE, \
    stage, doy, lat, lon, depth, distance, \
    temperature, food_availability, aragonite, aragonite_ref, \
    size_ref, size_NTE, damage_ref, damage_NTE, \
    duration_extremes, duration_corrosive, \
    mean_arag_NTE, mean_arag_NoX  = compile_observations(year, directory_N,
                                                       directory_NE, directory_NT,
                                                       directory_NTE ,folder,file_attrs,
                                                       max_ID, ref_ID, day_min, day_max, IDs_corrosive)

    # ================================================
    # Save the compiled data
    # ================================================
    np.save(f'{outdir}temp_all_year_{year}.npy', temperature)
    np.save(f'{outdir}food_all_year_{year}.npy', food_availability)
    np.save(f'{outdir}arag_all_year_{year}.npy', aragonite)
    np.save(f'{outdir}arag_all_ref_year_{year}.npy', aragonite_ref)
    np.save(f'{outdir}Longevity_year_{year}.npy', longevity)
    np.save(f'{outdir}Extremes_year_{year}.npy', extremes)
    np.save(f'{outdir}Corrosive_year_{year}.npy', corrosive)
    np.save(f'{outdir}Spawned_year_{year}.npy',spawned)

    np.save(f'{outdir}Would_die_N_year_{year}.npy', would_die_N)
    np.save(f'{outdir}Would_die_NE_year_{year}.npy', would_die_NE)
    np.save(f'{outdir}Would_die_NT_year_{year}.npy', would_die_NT)
    np.save(f'{outdir}Would_die_NTE_year_{year}.npy', would_die_NTE)

    np.save(f'{outdir}Stage_year_{year}.npy', stage)
    np.save(f'{outdir}Day_of_year_{year}.npy', doy)
    np.save(f'{outdir}Lat_year_{year}.npy', lat)
    np.save(f'{outdir}Lon_year_{year}.npy', lon)
    np.save(f'{outdir}Depth_year_{year}.npy', depth)
    np.save(f'{outdir}Distance_year_{year}.npy', distance)

    np.save(f'{outdir}Size_ref_year_{year}.npy', size_ref)
    np.save(f'{outdir}Size_NTE_year_{year}.npy', size_NTE)
    np.save(f'{outdir}Damage_ref_year_{year}.npy', damage_ref)
    np.save(f'{outdir}Damage_NTE_year_{year}.npy', damage_NTE)

    np.save(f'{outdir}Mean_arag_NTE_year_{year}.npy', mean_arag_NTE)
    np.save(f'{outdir}Mean_arag_ref_year_{year}.npy', mean_arag_NoX)

    np.save(f'{outdir}Duration_extremes_year_{year}.npy', duration_extremes)
    np.save(f'{outdir}Duration_corrosive_year_{year}.npy', duration_corrosive)

    print(f'Done with {year}')
    return

'''
Main Function
'''
if __name__ in "__main__":
    logging.basicConfig(level=logging.WARNING) #DEBUG

    dir_root = '/nfs/meso/work/ursho/PhD/Projects/Pteropod_Extremes/Analysis/'

    directory_N = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_noExtremes_14_V4/output/'
    directory_NE = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_14_V4/output/'
    directory_NT = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_Hindcast_noExtremes_14_V4/output/'
    directory_NTE = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_Hindcast_14_V4/output/'

    directory_N = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_noExtremes_14_V4/output/'
    directory_NE = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_14_V4/output/'
    directory_NT = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_Hindcast_noExtremes_14_V4/output/'
    directory_NTE = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_Hindcast_14_V4/output/'

    outdir = f'{directory_NTE}characteristics_spring_generation/'

    folder = 'netcdf/year_{}_V_5_control_1/'
    file_attrs = 'JitPtero_Day_{}_{}.nc'

    flag_calculate_obs_per_year = not os.path.exists(outdir+'obs_per_year.npy')

    years = np.arange(1984,2020)

    if flag_calculate_obs_per_year:
        array_sizes = np.ones((len(years),5))*np.nan
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        jobs = []
        for year in years:

            p = Process(target=determine_array_size, args=(year,directory_NTE,folder,file_attrs, outdir))
            jobs.append(p)
    #
        for process in jobs:
            process.start()
        for process in jobs:
            process.join()

        for i, year in enumerate(years):
            array_sizes[i,:] = np.load(outdir + f'obs_per_year_{year}.npy')

        np.save(outdir+'obs_per_year.npy', array_sizes)

    else:
        array_sizes = np.load(outdir+'obs_per_year.npy')

    for i_rep in range(0,12):
        jobs = []
        for part in array_sizes[int(3*i_rep):int(3*(i_rep+1)),:]:

            year, day_min, day_max, max_ID, ref_ID = part

            p = Process(target=main, args=(year, max_ID, ref_ID, day_min, day_max, directory_N, directory_NE, directory_NT, directory_NTE, folder, file_attrs, outdir))
            jobs.append(p)

        for process in jobs:
            process.start()
        for process in jobs:
            process.join()
