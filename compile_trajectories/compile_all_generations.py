#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:46:09 2022

Script used to extract responses to damaging conditiond. The script can also extract the
information for the long-term acidification events

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
import sys
import calendar
from tqdm import trange
from multiprocessing import Process
sys.path.insert(1,"/home/ursho/PhD/Projects/Pteropod_Extremes/Scripts/")
import spIBM


def normalize_food(F):

    daily_food = np.genfromtxt("/home/ursho/PhD/Projects/Pteropod_Extremes/Data/daily_food.csv",delimiter=',')
    b = max(daily_food)
    a = min(daily_food)

    return a + (F - 0.05)*(b-a) / (0.9 - 0.05)

def shell_growth(L,L_X,damage,damage_X,T,Food,Arag,Arag_X,growth_fct_gen0,growth_rate, weight_factor, Topt=18,Tmax=31,Tmin=0.6):
    """This function determines the net shell growth given the aragonite saturation state, current size, and generation. Returns array
    containing updated attributes characterizing the pteropods. UHE 25/09/2020

    Keyword arguments:
    pteorpod_list -- Array containing all state variables characterizing the pteropods
    growth_fct_gen0 -- Shell size as function of time for spring (X=0) and winter (X=1) generation
    Arag -- Aragonite saturation state experiences by each pteropod on one day
    T -- Temperature. Default value was set to 16 to simulate optimal conditions
    F -- Food/Phytoplankton carbon available. Default value 7 was chosen to simulate optimal conditions
    T0 -- Refernce temperature for the growth rate. Default value set to 14.5 according to Wang et al. 2017
    Ks -- Food/Phytoplankton carbon half-saturation constant. The default value is set to 2.6
    Tmax -- Maximum temperature for growth
    Tmin -- Minimum temperature for growth
    day -- list containing year, version, day and control for saving all growth rates for each day (default None)
    outfile -- output directory

    """
    L = np.asarray([L]) if np.isscalar(L) or L.ndim == 0 else np.asarray(L)
    L_X = np.asarray([L_X]) if np.isscalar(L_X) or L_X.ndim == 0 else np.asarray(L_X)
    T0 = 14.5
    Ks = 0.418
    # Tmax = 18.0
    F = normalize_food(Food)

    #ensure the structure of L is correct if there is only one pteropod or multiple pteropods
    if L.shape[0] != 1:
        L = np.squeeze(L)
    if L_X.shape[0] != 1:
        L_X = np.squeeze(L_X)
    #calculate distance to reference and find index with minimum distance
    pos_idx = np.array([np.squeeze(np.argwhere(abs(growth_fct_gen0-i) == abs(growth_fct_gen0-i).min())) for i in np.around(L,4)])
    pos_idx_X = np.array([np.squeeze(np.argwhere(abs(growth_fct_gen0-i) == abs(growth_fct_gen0-i).min())) for i in np.around(L_X,4)])
    food_effect = F/(Ks+F)

    alpha_CTMI_T_NoX = spIBM.cardinal_temperature_model_inflection(T, Topt, Tmax, Tmin)

    rate = growth_rate[pos_idx]*1.3**((T - T0)/10) * food_effect * alpha_CTMI_T_NoX
    rate_X = growth_rate[pos_idx_X]*1.3**((T - T0)/10) * food_effect * alpha_CTMI_T_NoX

    delta_L = rate*L
    delta_L_X = rate_X*L_X

    L,damage  = spIBM.calculate_dissolution_calcification(L,damage,delta_L,Arag, weight_factor)
    L_X,damage_X = spIBM.calculate_dissolution_calcification(L_X,damage_X,delta_L_X,Arag_X, weight_factor)

    return L, L_X, damage, damage_X

def zombie_state(vector,longevity):
    '''
    Function marks the degree of zombie for each time-step up to the natural death
    '''
    current_value = 0
    copy_vector = vector.copy()
    copy_vector[np.isnan(vector)] = 0
    for i,val in enumerate(copy_vector):
        if val > current_value:
            current_value = val
            copy_vector[i] = current_value
        else:
            copy_vector[i] = current_value

        if i == longevity:
            copy_vector[i] = -1
            copy_vector[i+1::] = np.nan
            return copy_vector

    return copy_vector

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

def get_recurrence(data):
    '''
    Function calculates the recurrence rate of extremes experienced by a pteropod
    '''
    idx_matrix = data*0 + np.arange(data.shape[1])
    available = data*idx_matrix
    available[data==False] = -1

    idxs = [np.diff(entry[entry>=0]) for entry in available]

    return [np.nanmean(idx[idx>0]) for idx in idxs]

def calculate_longevity_change(turning_points, longevity, num_obs):
    '''
    Function calculates the average decrease in longevity caused by the exposure to extremes
    througout the trajectory of a pteropod
    '''

    longevity_change = longevity.copy()*np.nan
    indeces_death = np.argwhere(turning_points > 0)
    longevities = longevity[indeces_death[:,0]]

    #for each unique index, calculate the mean change
    for value in np.unique(indeces_death[:,0]):

        mask = indeces_death[:,0] == value
        longevity_change[value] = np.sum((longevities[mask] - indeces_death[mask,1]))/num_obs[value]

    return longevity_change

def determine_array_size(year,directory_trend,folder,file_attrs, outdir):

    final_day = 366 if calendar.isleap(year) else 365
    max_ID = 0
    ref_ID = 0
    my_dir = directory_trend + folder.format(year)

    day_trange = trange(1,final_day)
    for day in day_trange:

        file_ptero = my_dir + file_attrs.format(day,'reduced')
        if os.path.exists(file_ptero):

            ds_ptero = xr.open_dataset(file_ptero)
            IDs = ds_ptero.MyID.values

            if day == 1:

                ref_ID = IDs.min()

            max_ID = max(max_ID,IDs.max())

    np.save(outdir+f'obs_per_year_{year}.npy', np.array([year, max_ID, ref_ID]))
    return

def determine_number_of_individuals(year,life_stage,gen,directory_trend,folder,file_attrs, outdir, outfile, start_day=None, end_day=None):


    my_dir = directory_trend + folder.format(year)
    first_entry = True
    if start_day is None:
        start_day=1
    if end_day is None:
        end_day = 366 if calendar.isleap(year) else 365

    day_trange = trange(start_day,end_day)
    for day in day_trange:

        file_ptero = my_dir + file_attrs.format(day,'reduced')
        if os.path.exists(file_ptero):

            ds_ptero = xr.open_dataset(file_ptero)
            stages = ds_ptero.stage.values
            generation = ds_ptero.generation.values%2

            if gen is None:
                IDs = ds_ptero.MyID.values[(stages>=life_stage)]
            else:
                IDs = ds_ptero.MyID.values[(stages>=life_stage)*(generation==gen)]


            if IDs.shape[0]>0:
                if first_entry:
                    possible_IDs = IDs
                    first_entry = False
                else:
                    possible_IDs = np.unique(np.hstack((possible_IDs, IDs)))

    np.save(outdir+ outfile.format(year), possible_IDs.shape)
    return


def compile_observations(year, directory_N, directory_NE, directory_NT,
                         directory_NTE ,folder,file_attrs,
                         extremes_only=True, num_traj_par=None):
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

    #if not already done, find the largest and smallest ID to allocate memory
    final_day = 366 if calendar.isleap(year) else 365
    max_ID = 0

    my_dir = directory_N + folder.format(year)

    if num_traj_par is None:
        day_trange = trange(1,final_day, leave=True)
        for day in day_trange:

            file_ptero = my_dir + file_attrs.format(day,'reduced')

            if os.path.exists(file_ptero):
                ds_ptero = xr.open_dataset(file_ptero)

                ds_ptero = xr.open_dataset(file_ptero)
                IDs = ds_ptero.MyID.values

                if day == 1:
                    ref_ID = IDs.min()

                max_ID = max(max_ID,IDs.max())

    else:
        max_ID = num_traj_par[0]
        ref_ID = num_traj_par[1]

    num_traj = int(max_ID-ref_ID)+1

    # ================================================
    # Allocate memory for compiled data
    # ================================================
    trajectories_extreme = np.ones((num_traj,300))*np.nan
    trajectories_duration = np.ones((num_traj,300))*np.nan
    trajectories_duration_extremes = np.ones((num_traj,300))*np.nan
    trajectories_distance = np.ones((num_traj,300))*np.nan

    trajectories_intensity_base = np.ones((num_traj,300))*np.nan
    trajectories_intensity_N = np.ones((num_traj,300))*np.nan
    trajectories_intensity_NE = np.ones((num_traj,300))*np.nan
    trajectories_intensity_NT = np.ones((num_traj,300))*np.nan
    trajectories_intensity_NTE = np.ones((num_traj,300))*np.nan

    trajectories_mort_N = np.ones((num_traj,300))*np.nan
    trajectories_mort_NE = np.ones((num_traj,300))*np.nan
    trajectories_mort_NT = np.ones((num_traj,300))*np.nan
    trajectories_mort_NTE = np.ones((num_traj,300))*np.nan

    trajectory_shell_base = np.ones((num_traj,300))*np.nan
    trajectory_shell_N = np.ones((num_traj,300))*np.nan
    trajectory_shell_NE = np.ones((num_traj,300))*np.nan
    trajectory_shell_NT = np.ones((num_traj,300))*np.nan
    trajectory_shell_NTE = np.ones((num_traj,300))*np.nan

    damage_acc_base = np.ones((num_traj,300))*np.nan
    damage_acc_N = np.ones((num_traj,300))*np.nan
    damage_acc_NE = np.ones((num_traj,300))*np.nan
    damage_acc_NT = np.ones((num_traj,300))*np.nan
    damage_acc_NTE = np.ones((num_traj,300))*np.nan

    trajectory_weight_factor = np.ones((num_traj,300))*np.nan

    # ================================================
    # Calculate the default growth function
    # ================================================
    growth_fct = spIBM.calculate_growth_fct()
    growth_rate = [(growth_fct[i]-growth_fct[i-1])/growth_fct[i-1] for i in range(1,len(growth_fct))]
    growth_rate.insert(0,growth_rate[0])
    growth_rate.append(growth_rate[-1])
    growth_rate = np.array(growth_rate)

    # ===============================================
    # Loop over each day
    # ================================================
    day_trange = trange(1,final_day, leave=True)
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
            # ================================================
            # Load reduced and env data for each simulation set
            # ================================================
            ds_ptero_N = xr.open_dataset(file_ptero_N)
            ds_ptero_NE = xr.open_dataset(file_ptero_NE)
            ds_ptero_NT = xr.open_dataset(file_ptero_NT)
            ds_ptero_NTE = xr.open_dataset(file_ptero_NTE)
            ds_env_NTE = xr.open_dataset(file_env_NTE)

            # ================================================
            #get IDs and produce indeces for trajectory tables
            # ================================================
            myids = ds_ptero_NTE.MyID
            index_ID = myids.values.astype(int) - int(ref_ID)

            # ================================================
            #extract flag of extremes and aragonite saturation states
            # ================================================
            arag_X_flag = ds_env_NTE.extreme_arag_flag
            arag_24h = ds_env_NTE.extreme_arag

            step_counter_arag = ds_ptero_NTE.step_counter_arag

            arag_sum_base = xr.where(step_counter_arag>0, ds_ptero_NTE.arag_sum/step_counter_arag, 1.5)
            arag_sum_N = xr.where(step_counter_arag>0, ds_ptero_N.extreme_arag_sum/step_counter_arag, 1.5)
            arag_sum_NE = xr.where(step_counter_arag>0, ds_ptero_NE.extreme_arag_sum/step_counter_arag, 1.5)
            arag_sum_NT = xr.where(step_counter_arag>0, ds_ptero_NT.extreme_arag_sum/step_counter_arag, 1.5)
            arag_sum_NTE = xr.where(step_counter_arag>0, ds_ptero_NTE.extreme_arag_sum/step_counter_arag, 1.5)

            arag_sum_base = xr.where(step_counter_arag==0, 1.5, arag_sum_base)
            arag_sum_N = xr.where(step_counter_arag==0, 1.5, arag_sum_N)
            arag_sum_NE = xr.where(step_counter_arag==0, 1.5, arag_sum_NE)
            arag_sum_NT = xr.where(step_counter_arag==0, 1.5, arag_sum_NT)
            arag_sum_NTE = xr.where(step_counter_arag==0, 1.5, arag_sum_NTE)

            # ================================================
            # Produce mask for events, i.e., either an extreme or everything,
            # but the aragonite saturation state has to be lower than 1.5
            # ================================================
            if extremes_only:
                mask_extremes_only = xr.where(arag_X_flag == 1, True, False).squeeze()
            else:
                mask_extremes_only = xr.where(arag_X_flag >= 0, True, False).squeeze()

            #only consider trajectories where there was an event of at least 1 hour
            duration_extremes_only = mask_extremes_only.sum(dim='obs')
            mask_extremes_only_sum = xr.where(duration_extremes_only > 0, True, False)

            #take only trajectories that can have an effect, i.e. are below the threshold 1.5
            mask_extremes_only_15 = xr.where(arag_sum_NTE < 1.5, True, False).squeeze()

            mask_extremes_only_ptero = mask_extremes_only_sum*mask_extremes_only_15

            hours_corrosive = xr.where(arag_24h < 1.5, True, False).sum(dim='obs').squeeze()

            if mask_extremes_only_ptero.sum() > 0:
                #extract indeces of trajectories that experienced an event
                index_ID_X = index_ID[mask_extremes_only_ptero]

                #get the day of growth of the pteropod
                days_of_growth = ds_ptero_NTE.days_of_growth[mask_extremes_only_ptero].values.astype(int)

                mask_extremes_isolated = xr.where(arag_X_flag == 1, True, False).sum(dim='obs')

                is_extreme = xr.where(mask_extremes_isolated > 0, 1, 0)
                hours_corrosive_extreme = xr.where(arag_X_flag == 1, True, False).sum(dim='obs')

                size_base = ds_ptero_NTE.shell_size[mask_extremes_only_ptero]
                size_N = ds_ptero_N.extreme_shell_size[mask_extremes_only_ptero]
                size_NE = ds_ptero_NE.extreme_shell_size[mask_extremes_only_ptero]
                size_NT = ds_ptero_NT.extreme_shell_size[mask_extremes_only_ptero]
                size_NTE = ds_ptero_NTE.extreme_shell_size[mask_extremes_only_ptero]

                L_base = size_base.values.squeeze()
                L_N = size_N.values.squeeze()
                L_NE = size_NE.values.squeeze()
                L_NT = size_NT.values.squeeze()
                L_NTE = size_NTE.values.squeeze()

                damage_base = ds_ptero_NTE.damage[mask_extremes_only_ptero].values.squeeze()
                damage_N = ds_ptero_N.extreme_damage[mask_extremes_only_ptero].values.squeeze()
                damage_NE = ds_ptero_NE.extreme_damage[mask_extremes_only_ptero].values.squeeze()
                damage_NT = ds_ptero_NT.extreme_damage[mask_extremes_only_ptero].values.squeeze()
                damage_NTE = ds_ptero_NTE.extreme_damage[mask_extremes_only_ptero].values.squeeze()

                # Get environmental variables
                distance = ds_ptero_NTE.distance[mask_extremes_only_ptero].values.squeeze()
                T = (ds_ptero_NTE.temp_sum[mask_extremes_only_ptero].values/24).squeeze()
                Food = (ds_ptero_NTE.food_sum[mask_extremes_only_ptero].values/24).squeeze()

                Arag_base = arag_sum_base[mask_extremes_only_ptero].values.squeeze()
                Arag_N = arag_sum_N[mask_extremes_only_ptero].values.squeeze()
                Arag_NE = arag_sum_NE[mask_extremes_only_ptero].values.squeeze()
                Arag_NT = arag_sum_NT[mask_extremes_only_ptero].values.squeeze()
                Arag_NTE = arag_sum_NTE[mask_extremes_only_ptero].values.squeeze()

                #Growth without dissolution --> L, L_X, damage, damage_X
                #Growth loss due to N
                weight_factor = (ds_ptero_NTE.step_counter_arag[mask_extremes_only_ptero]/ds_ptero_NTE.step_counter[mask_extremes_only_ptero]).values.squeeze()

                res = shell_growth(L_base,L_N,damage_base,damage_N,T,Food,Arag_base,Arag_N,growth_fct,growth_rate, weight_factor)
                size_base = res[0]
                damage_base = res[2]
                size_N = res[1]
                damage_N = res[3]

#                #Growth loss due to NE
                res = shell_growth(L_base,L_NE,damage_base,damage_NE,T,Food,Arag_base,Arag_NE,growth_fct,growth_rate, weight_factor)
                size_NE = res[1]
                damage_NE = res[3]

#                #Growth loss due to NT
                res = shell_growth(L_base,L_NT,damage_base,damage_NT,T,Food,Arag_base,Arag_NT,growth_fct,growth_rate, weight_factor)
                size_NT = res[1]
                damage_NT = res[3]

                #Growth loss due to NTE
                res = shell_growth(L_base,L_NTE,damage_base,damage_NTE,T,Food,Arag_base,Arag_NTE,growth_fct,growth_rate, weight_factor)
                size_NTE = res[1]
                damage_NTE = res[3]

                arag_exp_base = Arag_base
                arag_exp_N = Arag_N
                arag_exp_NE = Arag_NE
                arag_exp_NT = Arag_NT
                arag_exp_NTE = Arag_NTE

                wt = hours_corrosive_extreme[mask_extremes_only_ptero].values.squeeze()/24
                # calculate changes in mortality
                mort_rate_change_N = np.squeeze(spIBM.mortality_delta(T, Arag_N, 0.1, wt=weight_factor, temp_optimal=14, arag_optimal=1.5))
                mort_rate_change_NE = np.squeeze(spIBM.mortality_delta(T, Arag_NE, 0.1, wt=weight_factor, temp_optimal=14, arag_optimal=1.5))
                mort_rate_change_NT = np.squeeze(spIBM.mortality_delta(T, Arag_NT, 0.1, wt=weight_factor, temp_optimal=14, arag_optimal=1.5))
                mort_rate_change_NTE = np.squeeze(spIBM.mortality_delta(T, Arag_NTE, 0.1, wt=weight_factor, temp_optimal=14, arag_optimal=1.5))

                # ================================================
                # Write impacts in the right position
                # ================================================
#                trajectories_duration[index_ID_X,days_of_growth] = \
#                    duration_extremes_only[mask_extremes_only_ptero].values.reshape((-1,1))

                trajectories_duration[index_ID_X,days_of_growth] = \
                    hours_corrosive[mask_extremes_only_ptero].values.reshape((-1,1))

                trajectories_duration_extremes[index_ID_X,days_of_growth] = \
                    hours_corrosive_extreme[mask_extremes_only_ptero].values.reshape((-1,1))

                trajectories_extreme[index_ID_X,days_of_growth] = \
                    is_extreme[mask_extremes_only_ptero].values.reshape((-1,1))

                trajectories_distance[index_ID_X,days_of_growth] = \
                    distance.reshape((-1,1))

                trajectories_intensity_base[index_ID_X,days_of_growth] = arag_exp_base.reshape((-1,1))
                trajectories_intensity_N[index_ID_X,days_of_growth] = arag_exp_N.reshape((-1,1))
                trajectories_intensity_NE[index_ID_X,days_of_growth] = arag_exp_NE.reshape((-1,1))
                trajectories_intensity_NT[index_ID_X,days_of_growth] = arag_exp_NT.reshape((-1,1))
                trajectories_intensity_NTE[index_ID_X,days_of_growth] = arag_exp_NTE.reshape((-1,1))

                trajectories_mort_N[index_ID_X,days_of_growth] = mort_rate_change_N.reshape((-1,1))
                trajectories_mort_NE[index_ID_X,days_of_growth] = mort_rate_change_NE.reshape((-1,1))
                trajectories_mort_NT[index_ID_X,days_of_growth] = mort_rate_change_NT.reshape((-1,1))
                trajectories_mort_NTE[index_ID_X,days_of_growth] = mort_rate_change_NTE.reshape((-1,1))

                trajectory_shell_base[index_ID_X,days_of_growth] = size_base.reshape((-1,1))
                trajectory_shell_N[index_ID_X,days_of_growth] = size_N.reshape((-1,1))
                trajectory_shell_NE[index_ID_X,days_of_growth] = size_NE.reshape((-1,1))
                trajectory_shell_NT[index_ID_X,days_of_growth] = size_NT.reshape((-1,1))
                trajectory_shell_NTE[index_ID_X,days_of_growth] = size_NTE.reshape((-1,1))

                damage_acc_base[index_ID_X,days_of_growth] = damage_base.reshape((-1,1))
                damage_acc_N[index_ID_X,days_of_growth] = damage_N.reshape((-1,1))
                damage_acc_NE[index_ID_X,days_of_growth] = damage_NE.reshape((-1,1))
                damage_acc_NT[index_ID_X,days_of_growth] = damage_NT.reshape((-1,1))
                damage_acc_NTE[index_ID_X,days_of_growth] = damage_NTE.reshape((-1,1))

                trajectory_weight_factor[index_ID_X,days_of_growth] = weight_factor.reshape((-1,1))

    return trajectories_duration, trajectories_duration_extremes, trajectories_extreme, trajectories_distance, \
        trajectories_intensity_base, trajectories_intensity_N, trajectories_intensity_NE, \
        trajectories_intensity_NT, trajectories_intensity_NTE, \
        trajectories_mort_N, trajectories_mort_NE, trajectories_mort_NT, \
        trajectories_mort_NTE, \
        trajectory_shell_base, trajectory_shell_N, trajectory_shell_NE, trajectory_shell_NT, \
        trajectory_shell_NTE, damage_acc_base, damage_acc_N, damage_acc_NE, \
        damage_acc_NT, damage_acc_NTE, trajectory_weight_factor


def compile_observations_24h(year, directory_N, directory_NE, directory_NT, directory_NTE,
                             folder,file_attrs, indeces_extremes,
                             num_traj_par=None):


    final_day = 366 if calendar.isleap(year) else 365
    max_ID = 0

    my_dir = directory_NTE + folder.format(year)

    if num_traj_par is None:
        day_trange = trange(1,final_day, leave=True)
        for day in day_trange:

            file_ptero = my_dir + file_attrs.format(day,'reduced')

            if os.path.exists(file_ptero):

                ds_ptero = xr.open_dataset(file_ptero)
                IDs = ds_ptero.MyID.values

                if day == 1:
                    ref_ID = IDs.min()

                max_ID = max(max_ID,IDs.max())

    else:
        max_ID = num_traj_par[0]
        ref_ID = num_traj_par[1]

    num_traj = int(max_ID-ref_ID)+1

    trajectories_doy = np.ones((num_traj,300))*np.nan
    trajectories_lat = np.ones((num_traj,300))*np.nan
    trajectories_lon = np.ones((num_traj,300))*np.nan
    trajectories_depth = np.ones((num_traj,300))*np.nan
    trajectories_stage = np.ones((num_traj,300))*np.nan
    trajectories_generation = np.ones((num_traj,300))*np.nan
    trajectories_spawned = np.ones((num_traj,300))*np.nan

    trajectories_temp = np.ones((num_traj,300))*np.nan
    trajectories_food = np.ones((num_traj,300))*np.nan
    trajectories_arag_ref = np.ones((num_traj,300))*np.nan
    trajectories_arag_N = np.ones((num_traj,300))*np.nan
    trajectories_arag_NE = np.ones((num_traj,300))*np.nan
    trajectories_arag_NT = np.ones((num_traj,300))*np.nan
    trajectories_arag_NTE = np.ones((num_traj,300))*np.nan
    trajectories_size_ref = np.ones((num_traj,300))*np.nan
    trajectories_size_N = np.ones((num_traj,300))*np.nan
    trajectories_size_NE = np.ones((num_traj,300))*np.nan
    trajectories_size_NT = np.ones((num_traj,300))*np.nan
    trajectories_size_NTE = np.ones((num_traj,300))*np.nan

    trajectories_would_die_N = np.ones((num_traj,300))*np.nan
    trajectories_would_die_NE = np.ones((num_traj,300))*np.nan
    trajectories_would_die_NT = np.ones((num_traj,300))*np.nan
    trajectories_would_die_NTE = np.ones((num_traj,300))*np.nan

    longevity = np.ones((num_traj,))*np.nan

    growth_fct = spIBM.calculate_growth_fct()
    growth_rate = [(growth_fct[i]-growth_fct[i-1])/growth_fct[i-1] for i in range(1,len(growth_fct))]
    growth_rate.insert(0,growth_rate[0])
    growth_rate.append(growth_rate[-1])
    growth_rate = np.array(growth_rate)

    day_trange = trange(1,final_day, leave=True)

    for day in day_trange:
        # define paths to extreme end trend simulations
        file_ptero_N = directory_N + folder.format(year) + file_attrs.format(day,'reduced')
        file_ptero_NE = directory_NE + folder.format(year) + file_attrs.format(day,'reduced')
        file_ptero_NT = directory_NT + folder.format(year) + file_attrs.format(day,'reduced')
        file_ptero_NTE = directory_NTE + folder.format(year) + file_attrs.format(day,'reduced')

        flag_files_exist = os.path.exists(file_ptero_N) * os.path.exists(file_ptero_NE) * os.path.exists(file_ptero_NT) * os.path.exists(file_ptero_NTE)
        if flag_files_exist:
            # load reduced and environmental data for each simulation
            ds_ptero_N = xr.open_dataset(file_ptero_N)
            ds_ptero_NE = xr.open_dataset(file_ptero_NE)
            ds_ptero_NT = xr.open_dataset(file_ptero_NT)
            ds_ptero_NTE = xr.open_dataset(file_ptero_NTE)
            #get IDs and produce indeces in trajectory tables
            myids = ds_ptero_NTE.MyID

            index_ID = myids.values.astype(int) - int(ref_ID)

            #both simulation have the same basis, thus NoX values are the same across simulations
            #take indeces of pteropods of thecurrent day that will experience an extreme at some
            #point in their life
            indeces,comm1,comm2 = np.intersect1d(index_ID,indeces_extremes,return_indices=True)
            #comm1 denotes the indeces in index_ID where this is the case --> can be used directly
            #to index observations that in the future or at some point experience an extreme

            days_of_growth_all = ds_ptero_NTE.days_of_growth.values.astype(int)[comm1]
            lat = ds_ptero_NTE.lat.values[comm1]
            lon = ds_ptero_NTE.lon.values[comm1]
            depth = ds_ptero_NTE.z.values[comm1]
            stages = ds_ptero_NTE.stage.values[comm1]
            generation = ds_ptero_NTE.generation.values[comm1]
            spawned = ds_ptero_NTE.num_spawning_event.values[comm1]

            zombie_N = ds_ptero_N.flag_would_die.values[comm1]
            zombie_NE = ds_ptero_NE.flag_would_die.values[comm1]
            zombie_NT = ds_ptero_NT.flag_would_die.values[comm1]
            zombie_NTE = ds_ptero_NTE.flag_would_die.values[comm1]

            temp = (ds_ptero_NTE.temp_sum.values[comm1]/24).squeeze()
            food = (ds_ptero_NTE.food_sum.values[comm1]/24).squeeze()

            step_counter_arag = ds_ptero_NTE.step_counter_arag.values[comm1].squeeze()

            arag_ref = ds_ptero_NTE.arag_sum.values[comm1].squeeze()
            arag_N = ds_ptero_N.extreme_arag_sum.values[comm1].squeeze()
            arag_NE = ds_ptero_NE.extreme_arag_sum.values[comm1].squeeze()
            arag_NT = ds_ptero_NT.extreme_arag_sum.values[comm1].squeeze()
            arag_NTE = ds_ptero_NTE.extreme_arag_sum.values[comm1].squeeze()

            mask_available = step_counter_arag>0

            arag_ref[mask_available] = arag_ref[mask_available]/step_counter_arag[mask_available]
            arag_ref[mask_available==False] = 1.5

            arag_N[mask_available] = arag_N[mask_available]/step_counter_arag[mask_available]
            arag_N[mask_available==False] = 1.5

            arag_NE[mask_available] = arag_NE[mask_available]/step_counter_arag[mask_available]
            arag_NE[mask_available==False] = 1.5

            arag_NT[mask_available] = arag_NT[mask_available]/step_counter_arag[mask_available]
            arag_NT[mask_available==False] = 1.5

            arag_NTE[mask_available] = arag_NTE[mask_available]/step_counter_arag[mask_available]
            arag_NTE[mask_available==False] = 1.5

            L_base = ds_ptero_NTE.shell_size.values[comm1].squeeze()
            L_N = ds_ptero_N.extreme_shell_size.values[comm1].squeeze()
            L_NE = ds_ptero_NE.extreme_shell_size.values[comm1].squeeze()
            L_NT = ds_ptero_NT.extreme_shell_size.values[comm1].squeeze()
            L_NTE = ds_ptero_NTE.extreme_shell_size.values[comm1].squeeze()

            damage_base = ds_ptero_NTE.damage.values[comm1].squeeze()
            damage_N = ds_ptero_N.extreme_damage.values[comm1].squeeze()
            damage_NE = ds_ptero_NE.extreme_damage.values[comm1].squeeze()
            damage_NT = ds_ptero_NT.extreme_damage.values[comm1].squeeze()
            damage_NTE = ds_ptero_NTE.extreme_damage.values[comm1].squeeze()

            weight_factor = (ds_ptero_NTE.step_counter_arag/ds_ptero_NTE.step_counter).values[comm1].squeeze()

            res = shell_growth(L_base,L_N,damage_base,damage_N,temp,food,arag_ref,arag_N,growth_fct,growth_rate, weight_factor)
            size_ref = res[0]
            size_N = res[1]

            res = shell_growth(L_base,L_NE,damage_base,damage_NE,temp,food,arag_ref,arag_NE,growth_fct,growth_rate, weight_factor)
            size_NE = res[1]

            res = shell_growth(L_base,L_NT,damage_base,damage_NT,temp,food,arag_ref,arag_NT,growth_fct,growth_rate, weight_factor)
            size_NT = res[1]

            res = shell_growth(L_base,L_NTE,damage_base,damage_NTE,temp,food,arag_ref,arag_NTE,growth_fct,growth_rate, weight_factor)
            size_ref = res[0]
            size_NTE = res[1]

            longevity[index_ID[comm1],] = days_of_growth_all.reshape((-1,1))
            trajectories_doy[index_ID[comm1],days_of_growth_all] = day
            trajectories_lat[index_ID[comm1],days_of_growth_all] = lat.reshape((-1,1))
            trajectories_lon[index_ID[comm1],days_of_growth_all] = lon.reshape((-1,1))
            trajectories_depth[index_ID[comm1],days_of_growth_all] = depth.reshape((-1,1))
            trajectories_stage[index_ID[comm1],days_of_growth_all] = stages.reshape((-1,1))
            trajectories_generation[index_ID[comm1],days_of_growth_all] = generation.reshape((-1,1))
            trajectories_spawned[index_ID[comm1],days_of_growth_all] = spawned.reshape((-1,1))

            trajectories_temp[index_ID[comm1],days_of_growth_all] = temp.reshape((-1,1))
            trajectories_food[index_ID[comm1],days_of_growth_all] = food.reshape((-1,1))

            trajectories_arag_ref[index_ID[comm1],days_of_growth_all] = arag_ref.reshape((-1,1))
            trajectories_arag_N[index_ID[comm1],days_of_growth_all] = arag_N.reshape((-1,1))
            trajectories_arag_NE[index_ID[comm1],days_of_growth_all] = arag_NE.reshape((-1,1))
            trajectories_arag_NT[index_ID[comm1],days_of_growth_all] = arag_NT.reshape((-1,1))
            trajectories_arag_NTE[index_ID[comm1],days_of_growth_all] = arag_NTE.reshape((-1,1))

            trajectories_size_ref[index_ID[comm1],days_of_growth_all] = size_ref.reshape((-1,1))
            trajectories_size_N[index_ID[comm1],days_of_growth_all] = size_N.reshape((-1,1))
            trajectories_size_NE[index_ID[comm1],days_of_growth_all] = size_NE.reshape((-1,1))
            trajectories_size_NT[index_ID[comm1],days_of_growth_all] = size_NT.reshape((-1,1))
            trajectories_size_NTE[index_ID[comm1],days_of_growth_all] = size_NTE.reshape((-1,1))

            trajectories_would_die_N[index_ID[comm1],days_of_growth_all] = zombie_N.reshape((-1,1))
            trajectories_would_die_NE[index_ID[comm1],days_of_growth_all] = zombie_NE.reshape((-1,1))
            trajectories_would_die_NT[index_ID[comm1],days_of_growth_all] = zombie_NT.reshape((-1,1))
            trajectories_would_die_NTE[index_ID[comm1],days_of_growth_all] = zombie_NTE.reshape((-1,1))

    return longevity, trajectories_doy, trajectories_lat, trajectories_lon, trajectories_depth, trajectories_stage, \
           trajectories_generation, trajectories_spawned, trajectories_temp, trajectories_food, trajectories_arag_ref, \
           trajectories_arag_N, trajectories_arag_NE, trajectories_arag_NT, trajectories_arag_NTE, \
           trajectories_size_ref, trajectories_size_N, trajectories_size_NE, trajectories_size_NT, \
           trajectories_size_NTE, trajectories_would_die_N, trajectories_would_die_NE, trajectories_would_die_NT, \
           trajectories_would_die_NTE

def main(year, max_ID, ref_ID, directory_N, directory_NE, directory_NT, directory_NTE, folder, file_attrs, outdir,  extremes_only):

    print(year, max_ID, ref_ID)
    year = int(year)

    duration, duration_extremes, is_extreme, distance, instensity_exp_base, intensity_exp_N, \
    intensity_exp_NE, intensity_exp_NT, intensity_exp_NTE, \
    mort_N, mort_NE, mort_NT, mort_NTE, shell_base, shell_N, shell_NE, \
    shell_NT, shell_NTE, damage_base, damage_N, damage_NE,\
    damage_NT, damage_NTE, weight_factor = compile_observations(year, directory_N, directory_NE, directory_NT,
                         directory_NTE, folder, file_attrs, extremes_only=extremes_only ,num_traj_par=[max_ID,ref_ID])

    idx_events = np.unique(np.argwhere(duration > 0)[:,0])

    longevity, day_of_year, lat, lon, depth, stage, generation, spawned,\
    temp_all, food_all, arag_ref_all, arag_N_all, arag_NE_all, arag_NT_all, \
    arag_NTE_all, size_ref_all, size_N_all, size_NE_all, size_NT_all, \
    size_NTE_all, would_die_N_all, would_die_NE_all, would_die_NT_all, \
    would_die_NTE_all = compile_observations_24h(year, directory_N, directory_NE, directory_NT,
                         directory_NTE, folder, file_attrs, idx_events, num_traj_par=[max_ID,ref_ID])

    turning_points_N = would_die_N_all.copy()*np.nan
    turning_points_NE = would_die_NE_all.copy()*np.nan
    turning_points_NT = would_die_NT_all.copy()*np.nan
    turning_points_NTE = would_die_NTE_all.copy()*np.nan

    zombie_states_N = would_die_N_all.copy()*np.nan
    zombie_states_NE = would_die_NE_all.copy()*np.nan
    zombie_states_NT = would_die_NT_all.copy()*np.nan
    zombie_states_NTE = would_die_NTE_all.copy()*np.nan

    for i, idx in enumerate(idx_events):

        turning_points_N[idx,:] = zombie_turning_points(would_die_N_all[idx,:],longevity[idx])
        zombie_states_N[idx,:] = zombie_state(would_die_N_all[idx,:],longevity[idx])

        turning_points_NE[idx,:] = zombie_turning_points(would_die_NE_all[idx,:],longevity[idx])
        zombie_states_NE[idx,:] = zombie_state(would_die_NE_all[idx,:],longevity[idx])

        turning_points_NT[idx,:] = zombie_turning_points(would_die_NT_all[idx,:],longevity[idx])
        zombie_states_NT[idx,:] = zombie_state(would_die_NT_all[idx,:],longevity[idx])

        turning_points_NTE[idx,:] = zombie_turning_points(would_die_NTE_all[idx,:],longevity[idx])
        zombie_states_NTE[idx,:] = zombie_state(would_die_NTE_all[idx,:],longevity[idx])

    # ================================================
    # Save the compiled data
    # ================================================
    mask_events = np.isnan(longevity)==False

    np.save(f'{outdir}Day_of_year_{year}.npy', day_of_year[mask_events,:])

    np.save(f'{outdir}Duration_corrosive_year_{year}.npy',duration[mask_events,:])
    np.save(f'{outdir}Duration_extremes_year_{year}.npy',duration_extremes[mask_events,:])
    np.save(f'{outdir}IsExtreme_year_{year}.npy',is_extreme[mask_events,:])
    np.save(f'{outdir}Distance_year_{year}.npy',distance[mask_events,:])

    np.save(f'{outdir}Intensity_base_exp_year_{year}.npy',instensity_exp_base[mask_events,:])
    np.save(f'{outdir}Intensity_N_exp_year_{year}.npy',intensity_exp_N[mask_events,:])
    np.save(f'{outdir}Intensity_NE_exp_year_{year}.npy',intensity_exp_NE[mask_events,:])
    np.save(f'{outdir}Intensity_NT_exp_year_{year}.npy',intensity_exp_NT[mask_events,:])
    np.save(f'{outdir}Intensity_NTE_exp_year_{year}.npy',intensity_exp_NTE[mask_events,:])

    np.save(f'{outdir}Mort_N_year_{year}.npy',mort_N[mask_events,:])
    np.save(f'{outdir}Mort_NE_year_{year}.npy',mort_NE[mask_events,:])
    np.save(f'{outdir}Mort_NT_year_{year}.npy',mort_NT[mask_events,:])
    np.save(f'{outdir}Mort_NTE_year_{year}.npy',mort_NTE[mask_events,:])

    np.save(f'{outdir}Shell_Nodiss_year_{year}.npy',shell_base[mask_events,:])
    np.save(f'{outdir}Shell_N_year_{year}.npy',shell_N[mask_events,:])
    np.save(f'{outdir}Shell_NE_year_{year}.npy',shell_NE[mask_events,:])
    np.save(f'{outdir}Shell_NT_year_{year}.npy',shell_NT[mask_events,:])
    np.save(f'{outdir}Shell_NTE_year_{year}.npy',shell_NTE[mask_events,:])

    np.save(f'{outdir}Damage_Nodiss_year_{year}.npy',damage_base[mask_events,:])
    np.save(f'{outdir}Damage_N_year_{year}.npy',damage_N[mask_events,:])
    np.save(f'{outdir}Damage_NE_year_{year}.npy',damage_NE[mask_events,:])
    np.save(f'{outdir}Damage_NT_year_{year}.npy',damage_NT[mask_events,:])
    np.save(f'{outdir}Damage_NTE_year_{year}.npy',damage_NTE[mask_events,:])

    np.save(f'{outdir}Weight_factor_year_{year}.npy',weight_factor[mask_events,:])

    np.save(f'{outdir}Turning_N_year_{year}.npy',turning_points_N[mask_events,:])
    np.save(f'{outdir}Turning_NE_year_{year}.npy',turning_points_NE[mask_events,:])
    np.save(f'{outdir}Turning_NT_year_{year}.npy',turning_points_NT[mask_events,:])
    np.save(f'{outdir}Turning_NTE_year_{year}.npy',turning_points_NTE[mask_events,:])

    np.save(f'{outdir}ZombieState_N_year_{year}.npy',zombie_states_N[mask_events,:])
    np.save(f'{outdir}ZombieState_NE_year_{year}.npy',zombie_states_NE[mask_events,:])
    np.save(f'{outdir}ZombieState_NT_year_{year}.npy',zombie_states_NT[mask_events,:])
    np.save(f'{outdir}ZombieState_NTE_year_{year}.npy',zombie_states_NTE[mask_events,:])

    np.save(f'{outdir}Lat_year_{year}.npy',lat[mask_events,:])
    np.save(f'{outdir}Lon_year_{year}.npy',lon[mask_events,:])
    np.save(f'{outdir}Depth_year_{year}.npy',depth[mask_events,:])
    np.save(f'{outdir}Stage_year_{year}.npy',stage[mask_events,:])
    np.save(f'{outdir}Generation_year_{year}.npy',generation[mask_events,:])
    np.save(f'{outdir}Spawned_year_{year}.npy',spawned[mask_events,:])

    np.save(f'{outdir}temp_all_year_{year}.npy',temp_all[mask_events,:])
    np.save(f'{outdir}food_all_year_{year}.npy',food_all[mask_events,:])

    np.save(f'{outdir}arag_ref_all_year_{year}.npy',arag_ref_all[mask_events,:])
    np.save(f'{outdir}arag_N_all_year_{year}.npy',arag_N_all[mask_events,:])
    np.save(f'{outdir}arag_NE_all_year_{year}.npy',arag_NE_all[mask_events,:])
    np.save(f'{outdir}arag_NT_all_year_{year}.npy',arag_NT_all[mask_events,:])
    np.save(f'{outdir}arag_NTE_all_year_{year}.npy',arag_NTE_all[mask_events,:])

    np.save(f'{outdir}size_ref_all_year_{year}.npy',size_ref_all[mask_events,:])
    np.save(f'{outdir}size_N_all_year_{year}.npy',size_N_all[mask_events,:])
    np.save(f'{outdir}size_NE_all_year_{year}.npy',size_NE_all[mask_events,:])
    np.save(f'{outdir}size_NT_all_year_{year}.npy',size_NT_all[mask_events,:])
    np.save(f'{outdir}size_NTE_all_year_{year}.npy',size_NTE_all[mask_events,:])

    np.save(f'{outdir}Would_die_N_year_{year}.npy',would_die_N_all[mask_events,:])
    np.save(f'{outdir}Would_die_NE_year_{year}.npy',would_die_NE_all[mask_events,:])
    np.save(f'{outdir}Would_die_NT_year_{year}.npy',would_die_NT_all[mask_events,:])
    np.save(f'{outdir}Would_die_NTE_year_{year}.npy',would_die_NTE_all[mask_events,:])

    print(f'Done with {year}')
    return

'''
Main Function
'''
if __name__ in "__main__":

    dir_root = '/nfs/meso/work/ursho/PhD/Projects/Pteropod_Extremes/Analysis/'

    directory_N = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_noExtremes_14_V4/output/'
    directory_NE = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_14_V4/output/'
    directory_NT = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_Hindcast_noExtremes_14_V4/output/'
    directory_NTE = f'{dir_root}NEW_protoconch_Pteropod_Acidification_NoDiss_Hindcast_14_V4/output/'

    directory_N = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_noExtremes_14_V4/output/'
    directory_NE = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_ConstCO2_14_V4/output/'
    directory_NT = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_Hindcast_noExtremes_14_V4/output/'
    directory_NTE = f'{dir_root}NoDissplacement_protoconch_Pteropod_Acidification_NoDiss_Hindcast_14_V4/output/'

    outdir = f'{directory_NTE}characteristics_acidification_V4/'

    folder = 'netcdf/year_{}_V_5_control_1/'
    file_attrs = 'JitPtero_Day_{}_{}.nc'

    extremes_only = False
    flag_calculate_num_individuals = False
    flag_calculate_obs_per_year = not os.path.exists(outdir+'obs_per_year.npy')

    years = np.arange(1984,2020)

    if flag_calculate_obs_per_year:
        array_sizes = np.ones((len(years),3))*np.nan
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

    if flag_calculate_num_individuals:
        outfile = 'august_pop_per_year_{}.npy'
        array_sizes_ind = np.ones((len(years),1))*np.nan
        years = np.arange(1984,2020)
        life_stage = 2
        generation = 0
        jobs = []
        for year in years:
            p = Process(target=determine_number_of_individuals, args=(year, life_stage, None, directory_NTE, folder, file_attrs, outdir, outfile, 222, 246))
            jobs.append(p)

        for process in jobs:
            process.start()
        for process in jobs:
            process.join()

        for i, year in enumerate(years):
            array_sizes_ind[i,0] = np.load(outdir + f'august_pop_per_year_{year}.npy')

        np.save(outdir+'august_pop_per_year.npy', array_sizes_ind)



    for i_rep in range(0,12):
        jobs = []
        for part in array_sizes[int(3*i_rep):int(3*(i_rep+1)),:]:

            year, max_ID, ref_ID = part

            p = Process(target=main, args=(year, max_ID, ref_ID, directory_N, directory_NE, directory_NT, directory_NTE, folder, file_attrs, outdir,  extremes_only))
            jobs.append(p)

        for process in jobs:
            process.start()
        for process in jobs:
            process.join()
