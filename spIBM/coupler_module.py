#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:42:34 2022

@author: ursho
"""

import datetime
import logging
import numpy as np

from parcels import ParticleSet

def calculate_sunrise_sunset(lats,lons,year,current_time,rising_time=True,setting_time=False,
                             local_offset=-7,backwards=False):
    """This function calculates the sunrise and sunset timings for a pteropod at a lon/lat/date.
    The sunset is calculated for the same simulation day.
    The sunrise is calculated for the next day. The shift in one day is done since the simulation
    time step is one day, and the origin is set at noon. Description of algorithm is found
    in: http://www.edwilliams.org/sunrise_sunset_algorithm.htm
    Original document in: https://babel.hathitrust.org/cgi/pt?id=uiug.30112105115718&view=1up&seq=3

    Parameters:
    lats (array): array of latitudes
    lons (array): array of longitudes
    year (int): year in the simulation
    current_time (int): current simulation time in seconds
    rising_time (bool): flag to signal the function to calculate the sunrise timing (default=True)
    setting_time (bool): flag to signal the function to calculate the sunset timing (defualt=False)
    localOffset (int): Number of hours offset from UTC (default=-7)
    day (int): current day (no longer used)

    Returns:
    localT (float): time at which the sun rises on the following day or sun sets on the same day
    seconds_since_origin (int): Number of seconds since the origin that the sun sets or rises

    UHE 14/01/2022
    """

    #define origin time 12:05 is given by ROMS output!
    origin = datetime.datetime(year, 1, 1) + datetime.timedelta(hours=12) + datetime.timedelta(minutes=5)
    #ensure something is calculated
    if setting_time == rising_time:
        rising_time = True
        setting_time = False
    elif setting_time:
        day_offset = 1
        datetime_offset = 0
    elif rising_time:
        day_offset = 0
        datetime_offset = 1

    #define as datetime and date object
    tmp_datetime = datetime.datetime(year,1,1) + datetime.timedelta(seconds=current_time) + datetime.timedelta(days=datetime_offset) - backwards*datetime.timedelta(days=1)
    #get day of year
    tmp = datetime.date(year,1,1) + datetime.timedelta(seconds=current_time) - datetime.timedelta(days=day_offset) - backwards*datetime.timedelta(days=1)
    #one is added since we want the time elapsed in days since 0 January of the current year
    time_elapsed = (tmp - datetime.date(tmp.year,1,1)).days + 1

    longitude = lons - 360 #since they are given in degrees east without negative values #-125
    latitude = lats #no change needed since given as positive for NH and negative for SH

    long_hour = longitude / 15
    if rising_time:
        my_time = time_elapsed + ((6 - long_hour) / 24)
    if setting_time:
        my_time = time_elapsed + ((18 - long_hour) / 24)

    M = (0.9856 * my_time) - 3.289

    L = M + (1.916 * np.sin(np.pi/180 * M)) + (0.020 * np.sin(np.pi/180 * 2 * M)) + 282.634

    f_L = np.vectorize(lambda x: x - 360 if x >= 360 else x + 360 if x < 0 else x)
    L = f_L(L)

    ra = 180/np.pi * np.arctan(0.91764 * np.tan(np.pi/180 * L))
    f_ra = np.vectorize(lambda x: x - 360 if x >= 360 else x + 360 if x < 0 else x)
    ra = f_ra(ra)

    l_quadrant  = (np.floor(L/90)) * 90
    ra_quadrant = (np.floor(ra/90)) * 90
    ra = ra + (l_quadrant - ra_quadrant)

    ra = ra / 15

    sin_decimal = 0.39782 * np.sin(np.pi/180 * L)
    cos_decimal = np.cos(np.arcsin(sin_decimal))

    zenith = 90.83333 #official
    cos_h = (np.cos(np.pi/180 * zenith) - (sin_decimal * np.sin(np.pi/180 * latitude))) / (cos_decimal * np.cos(np.pi/180 * latitude))
    cos_h_above = cos_h > 1
    cos_h_below = cos_h < -1
    cos_h_outside_range = (cos_h > 1) + (cos_h < -1)
    if cos_h_above.sum() >  0:
        logging.warning('The sun never rises on some location (on the specified date)')
    if cos_h_below.sum() > 0:
        logging.warning('The sun never sets on some location (on the specified date)')

    while cos_h_outside_range.sum() > 0:
        negative_lat = latitude < 0
        latitude[cos_h_outside_range*(negative_lat==True)] += 0.1
        latitude[cos_h_outside_range*(negative_lat==False)] -= 0.1
        cos_h = (np.cos(np.pi/180 * zenith) - (sin_decimal * np.sin(np.pi/180 * latitude))) / (cos_decimal * np.cos(np.pi/180 * latitude))
        cos_h_outside_range = (cos_h > 1) + (cos_h < -1)

    if rising_time:
        H = 360 - 180/np.pi * np.arccos(cos_h)
    if setting_time:
        H = 180/np.pi * np.arccos(cos_h)

    H = H / 15
    T = H + ra - (0.06571 * my_time) - 6.622

    UT = T - long_hour
    f_UT = np.vectorize(lambda x: x - 24 if x >= 24 else x + 24 if x < 0 else x)
    UT = f_UT(UT)

    local_time = UT + local_offset
    local_time = np.where(local_time < 0, 24 + local_time, local_time)
    seconds_since_origin = (tmp_datetime - origin).total_seconds() + local_time*60*60

    return seconds_since_origin

def prepare_particles(pset,year,w_down=45,w_up=45,w_down_wings=14,max_size=4.53,
                      max_depth=250.0,backwards=False):
    """This function calculates the variable values used for the DVM movement, i.e. the downward
    and upward swimming speeds, as well as the timings of the onset of the vertical migration.
    This function also resets binary values used to keep track of their position more easily.
    Variables used to calculate mean exposure values are reset to 0 or negative values


    Parameters:
    pset (parcels obj): particle to be deleted
    fieldset (dask array): fieldset describing the environment
    year (float): current simulation year

    Returns:
    pset (parcels obj): parcels object containing all pteropods with updated values
    UHE 5/10/2020

    """

    arrival_time_dist = np.random.normal(20, 20, pset.size)
    dep_time_dist = np.random.normal(20, 20, pset.size)

    w_down = w_down/1000
    w_down_wings = w_down_wings/1000
    w_up = w_up/1000

    lats = pset.particle_data['lat'][:]
    lons = pset.particle_data['lon'][:]
    current_time = pset[0].time

    sunset_seconds = calculate_sunrise_sunset(lats,lons,year,current_time,rising_time=0,
                                              setting_time=1,local_offset=-7,backwards=backwards)
    sunrise_seconds = calculate_sunrise_sunset(lats,lons,year,current_time,rising_time=1,
                                               setting_time=0,local_offset=-7,backwards=backwards)

    shell_sizes = pset.particle_data['shell_size'][:]

    arrival_time = sunset_seconds + arrival_time_dist*60

    departure_time = sunrise_seconds + dep_time_dist*60

    pset.particle_data['departure'][:] = departure_time

    pset.particle_data['up'][:] = w_up*shell_sizes/max_size

    pset.particle_data['down'][:] = w_down*shell_sizes/max_size
    pset.particle_data['down_wings'][:] = w_down_wings*shell_sizes/max_size

    pset.particle_data['max_depth'][:] = max_depth*shell_sizes/max_size

    pset.particle_data['next_max_depth'][:] = max_depth*shell_sizes/max_size

    pset.particle_data['departure_from_depth'][:] = arrival_time  - (pset.particle_data['max_depth'][:]/pset.particle_data['up'][:])

    pset.particle_data['flag_up'][:] = 0
    pset.particle_data['flag_down'][:] = 0
    pset.particle_data['reseed_flag'][:] = 0
    pset.particle_data['chl_max'][:] = -99999.9
    pset.particle_data['chl_ascent'][:] = -99999.9
    pset.particle_data['depth_chl_max'][:] = 0.0

    pset.particle_data['step_counter_arag'][:] = 0
    pset.particle_data['step_counter'][:] = 0
    pset.particle_data['temp_sum'][:] = 0.0
    pset.particle_data['food_sum'][:] = 0.0
    pset.particle_data['arag_sum'][:] = 0.0
    pset.particle_data['oxygen_sum'][:] = 0.0
    pset.particle_data['extreme_arag_sum'][:] = 0.0

    pset.particle_data['rng_surf'][:] = np.random.uniform(0.0, 50, pset.size)
    pset.particle_data['rng_bot'][:] = np.random.uniform(0.0, 50, pset.size)

    return pset

def convert_to_mat(pset):
    """This function converts a parcels object to a numpy array


    Parameters:
    pst (parcels obj): particle to be deleted

    Returns:
    matrix (numpy array): array containing all values stored in pset
    exposure_weight_factor (numpy array): array containing all weight factors for the exposure to
    damaging conditions
    UHE 21/09/2022

    """
    matrix = np.empty((pset.size,22), dtype=np.float32)

    step_counter = pset.particle_data['step_counter'][:]
    step_counter_arag = pset.particle_data['step_counter_arag'][:]
    step_counter_arag[step_counter_arag<1] = -1

    exposure_weight_factor = step_counter_arag/step_counter
    exposure_weight_factor[step_counter_arag<1] = 0

    matrix[:,0] = pset.particle_data['MyID'][:]
    matrix[:,1] = pset.particle_data['generation'][:]
    matrix[:,2] = pset.particle_data['stage'][:]
    matrix[:,3] = pset.particle_data['shell_size'][:]
    matrix[:,4] = pset.particle_data['days_of_growth'][:]
    matrix[:,5] = pset.particle_data['survive'][:]
    matrix[:,6] = pset.particle_data['num_spawning_event'][:]
    matrix[:,7] = pset.particle_data['ERR'][:]
    matrix[:,8] = pset.particle_data['spawned'][:]
    matrix[:,9] = pset.particle_data['Parent_ID'][:]
    matrix[:,10] = pset.particle_data['Parent_shell_size'][:]
    matrix[:,11] = -1
    matrix[:,12] = pset.particle_data['time'][:]

    matrix[:,13] = (pset.particle_data['arag_sum'][:]/step_counter_arag).astype(np.float32)
    matrix[step_counter_arag<1,13] = 1.5

    matrix[:,14] = pset.particle_data['damage'][:]
    matrix[:,15] = (pset.particle_data['temp_sum'][:]/step_counter).astype(np.float32)

    #take the overall sum not the average
    matrix[:,16] = (pset.particle_data['food_sum'][:]/step_counter).astype(np.float32)

    matrix[:,18] = (pset.particle_data['extreme_arag_sum'][:]/step_counter_arag).astype(np.float32)
    matrix[step_counter_arag<1,18] = 1.5

    matrix[:,19] = pset.particle_data['flag_would_die'][:]
    matrix[:,20] = pset.particle_data['extreme_shell_size'][:]
    matrix[:,21] = pset.particle_data['extreme_damage'][:]
    #get the mean aragonite from past five days

    return matrix, exposure_weight_factor


def update_particleset(matrix,pset,fieldset,pclass,year,max_size=4.53,backwards=False):
    """This function translates the values from a numpy array to a parcels object
    In addition, it calculates the swimming velocities, onset of vertical migration and DVM depths
    based on the size of the pteropods. Function also adds eggs to the custom parcels object.
    Binary and cumulative values used to keep track of their exposure, depth, location of maximum
    NPP are reset.

    Parameters:
    matrix (numpy array): array containing updated state variables of pteropods
    pset (parcels obj): custom parcels object containing pteropods before updating the values
    fieldset (dask array): fieldset describing the environment
    year (int): current simulation year
    day (int): current simulation day
    die_list (list): list cotaining indeces of particles that died in this time step

    Returns:
    pset (parcels obj): custom parcels object containing pteropods with updated values
    (state variables and variables used to calculate movement)
    UHE 5/10/2020

    """

    #calculate swimming velocity, depths and timing of DVM for all particles using matrix
    update_values = np.full((matrix.shape[0],8),0.0)
    # ==============================================================================================
    # up/down swim velocity
    # ==============================================================================================
    update_values[:,0] = 45.0/1000
    update_values[:,1] = 14.0/1000
    update_values[:,2] = 45.0/1000

    update_values[:,3] = 250.0
    update_values[:,5] = 250.0
    update_values[:,4] = 250.0
    update_values[:,6] = 250.0
    update_values[:,7] = 250.0

    arrival_time_dist = np.random.normal(20, 20, matrix.shape[0])
    dep_time_dist = np.random.normal(20, 20, matrix.shape[0])

    lats = pset.particle_data['lat'][:]
    lons = pset.particle_data['lon'][:]
    current_time = pset[0].time
    sunset_seconds = calculate_sunrise_sunset(lats,lons,year,current_time,rising_time=0,
                                              setting_time=1,local_offset=-7,backwards=backwards)
    sunrise_seconds = calculate_sunrise_sunset(lats,lons,year,current_time,rising_time=1,
                                               setting_time=0,local_offset=-7,backwards=backwards)

    ids_pset = pset.particle_data['MyID'][:]
    ids_mat = matrix[:,0]
    #find individuals that survived
    idx_survivors = np.squeeze(np.isin(ids_pset,ids_mat))
    #find newcomers, i.e. eggs produced in this time-step
    idx_newcomers = np.squeeze(np.isin(ids_mat,ids_pset)==False)
    ind_mat = idx_newcomers == False

    # update the values of the survivors
    arrival_time = sunset_seconds + arrival_time_dist[ind_mat]*60
    departure_time = sunrise_seconds + dep_time_dist[ind_mat]*60
    size_scale_factor = pset.particle_data['shell_size'][idx_survivors]/max_size

    pset.particle_data['generation'][idx_survivors] = matrix[ind_mat,1]
    pset.particle_data['stage'][idx_survivors] = matrix[ind_mat,2]
    pset.particle_data['shell_size'][idx_survivors] = matrix[ind_mat,3]
    pset.particle_data['days_of_growth'][idx_survivors] = matrix[ind_mat,4]
    pset.particle_data['survive'][idx_survivors] = matrix[ind_mat,5]
    pset.particle_data['num_spawning_event'][idx_survivors] = matrix[ind_mat,6]
    pset.particle_data['ERR'][idx_survivors] = matrix[ind_mat,7]
    pset.particle_data['spawned'][idx_survivors] = matrix[ind_mat,8]
    pset.particle_data['arag_sum'][idx_survivors] = 0.0
    pset.particle_data['temp_sum'][idx_survivors] = 0.0
    pset.particle_data['food_sum'][idx_survivors] = 0.0
    pset.particle_data['oxygen_sum'][idx_survivors] = 0.0
    pset.particle_data['step_counter'][idx_survivors] = 0
    pset.particle_data['step_counter_arag'][idx_survivors] = 0
    pset.particle_data['damage'][idx_survivors] = matrix[ind_mat,14]

    pset.particle_data['shell_thickness'][idx_survivors] = -1

    pset.particle_data['flag_would_die'][idx_survivors] = matrix[ind_mat,19]
    pset.particle_data['extreme_arag_sum'][idx_survivors] = 0.0
    pset.particle_data['extreme_shell_size'][idx_survivors] = matrix[ind_mat,20]
    pset.particle_data['extreme_damage'][idx_survivors] = matrix[ind_mat,21]

    pset.particle_data['down'][idx_survivors] = update_values[ind_mat,0]*size_scale_factor
    pset.particle_data['down_wings'][idx_survivors] = update_values[ind_mat,1]*size_scale_factor
    pset.particle_data['up'][idx_survivors] = update_values[ind_mat,2]*size_scale_factor
    pset.particle_data['max_depth'][idx_survivors] =  update_values[ind_mat,4]*size_scale_factor
    pset.particle_data['next_max_depth'][idx_survivors] = update_values[ind_mat,5]*size_scale_factor
    pset.particle_data['departure_from_depth'][idx_survivors] = arrival_time - pset.particle_data['max_depth'][idx_survivors]/pset.particle_data['up'][idx_survivors]
    pset.particle_data['departure'][idx_survivors] = departure_time
    pset.particle_data['flag_up'][idx_survivors] = 0
    pset.particle_data['flag_down'][idx_survivors] = 0

    pset.particle_data['rng_surf'][:] = np.random.uniform(0.0, 50, pset.size)
    pset.particle_data['rng_bot'][:] = np.random.uniform(0.0, 50, pset.size)
    #Get indeces of parents for all newcomers
    idx_parents = np.squeeze(matrix[idx_newcomers,13]).astype(int)
    flag_would_die_new = (pset.particle_data['flag_would_die'][idx_parents] > 0).astype(int)

    if sum(idx_newcomers) > 0:
        pset.add(ParticleSet(fieldset=fieldset, pclass=pclass,\
                                        lon=pset.particle_data['lon'][idx_parents],\
                                        lat=pset.particle_data['lat'][idx_parents],\
                                        depth=pset.particle_data['depth_chl_max'][idx_parents],\
                                        time=pset.particle_data['time'][idx_parents],\
                                        stage=matrix[idx_newcomers,2],\
                                        generation=pset.particle_data['generation'][idx_parents]+1,\
                                        MyID=matrix[idx_newcomers,0],\
                                        Parent_ID=matrix[idx_newcomers,9],\
                                        flag_would_die=flag_would_die_new,\
                                        Parent_shell_size=matrix[idx_newcomers,10],\
                                        lonlatdepth_dtype=np.float32))


    #reset DVM min depth seeking
    pset.particle_data['chl_max'][:] = -99999.9
    pset.particle_data['chl_ascent'][:] = -99999.9
    pset.particle_data['depth_chl_max'][:] = 99999.9

    return pset


def get_dead_particles(pset,die_list):
    """This function removes a given particle from the cutom parcels object based on their indeces
    given in a list

    Parameters:
    pset (parcels obj): custom parcels object containing pteorpods
    die_list (list): list of indeces of pteropods that died in the current time step

    Returns:
    None
    UHE 5/10/2020
    """
    if die_list.size == 1:
        die_list = np.reshape(die_list,(1,))
    pset.particle_data['survive'][die_list] = 0

    pset.remove_indices(die_list)

    return pset
