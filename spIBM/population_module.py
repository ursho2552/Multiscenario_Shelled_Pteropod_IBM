#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:48:46 2022
Functions used to calculate the mortality, growth, development, and spawning of
the modeled pteropods
@author: ursho
"""
import os

import numpy as np

def calculate_growth_fct():
    """Calculate the shell size of pteropods as a function of their age.
    The formula was taken from Wang et al., 2017. Returns the growth rate in
    (mm) as a function of time after birth

    Keyword arguments:
    None
    """

    winter_point = 0.1
    rate_length_inf = 5.07
    length_inf = 4.53
    growth_amplitude = 0.4
    sinusoidal_growth_start = winter_point - 0.5
    reference_age = 120/365

    age = np.arange(121,121+300)/365

    s_t = (growth_amplitude*rate_length_inf)/(2*np.pi) * np.sin(2*np.pi*(age - sinusoidal_growth_start))
    s_t0 = (growth_amplitude*rate_length_inf)/(2*np.pi) * np.sin(2*np.pi*(reference_age - sinusoidal_growth_start))
    length_function = length_inf*(1-np.exp(-(rate_length_inf*(age - reference_age) + s_t - s_t0)))
    length_function += 0.15 - length_function[0]

    return length_function

def get_number_individuals_simplified(indeces, rate_generation_stage):
    """This function uses beta survival (b) rates to calculate the fraction of
    individual that survive after one time step. Returns the number of
    individuals that would die on position 0, and the mortality rate (1-rate)
    on position 1.

    The function used is taken from Bednarsek et al., 2016:
    N_{t+1} = N_{t}*exp(-b*t),
    rate = exp(-b*t) with t = 1 day.

    Keyword arguments:
    indeces -- list of indeces of a specific stage in the population
    stage -- integer identifier for the life stage
    generation -- integer identifier for the generation, 0 for spring, 1 for
        winter
    rate_gX_Y -- beta survival rates for each stage and generation
    """

    rate = np.exp(-rate_generation_stage)

    return [int(np.around(indeces.size*(1-rate))),1-rate]

def get_number_individuals(indeces, stage, generation, rate_g0_0, rate_g0_1,
                           rate_g0_2, rate_g0_3, rate_g1_0, rate_g1_1,
                           rate_g1_2, rate_g1_3):
    """This function uses beta survival (b) rates to calculate the fraction of
    individual that survive after one time step. Returns the number of
    individuals that would die on position 0, and the mortality rate (1-rate)
    on position 1.

    The function used is taken from Bednarsek et al., 2016:
        N_{t+1} = N_{t}*exp(-b*t),
        rate = exp(-b*t) with t = 1 day.

    Keyword arguments:
    indeces -- list of indeces of a specific stage in the population
    stage -- integer identifier for the life stage
    generation -- integer identifier for the generation, 0 for spring, 1 for
                  winter
    rate_gX_Y -- beta survival rates for each stage and generation
    """
    if generation == 0:
        if stage == 0:
            rate = np.exp(-rate_g0_0)
        elif stage == 1:
            rate = np.exp(-rate_g0_1)
        elif stage == 2:
            rate = np.exp(-rate_g0_2)
        elif stage == 3:
            rate = np.exp(-rate_g0_3)

    elif generation == 1:
        if stage == 0:
            rate = np.exp(-rate_g1_0)
        if stage == 1:
            rate = np.exp(-rate_g1_1)
        elif stage == 2:
            rate = np.exp(-rate_g1_2)
        elif stage == 3:
            rate = np.exp(-rate_g1_3)

    return [int(np.around(indeces.size*(1 - rate))),1 - rate]

def mortality_delta(temperature, arag, rate, weight_factor=1, temperature_optimal=14.0, arag_optimal=1.5):
    """This function calculates the increase in mortality rate due to temperature
    and aragonite saturation state. Returns an increase in mortality rate due to
    environmental conditons. Formula is based on Bednarsek et al., 2022

    Keyword arguments:
    temp -- temperature experienced by the pteropod
    arag -- aragonite experienced by the pteropods
    rate -- mortality rate
    weight_factor -- fraction of time under damaging conditions (arag < 1.5). Default value is 1
    temperature_optimal -- boptimal temperature used for reference. Default value is 14°C
    arag_optimal -- optimal aragonite used as reference. Default value is 1.5
    """

    temperature_array = np.asarray([temperature]) if np.isscalar(temperature) or temperature.ndim == 0 else np.asarray(temperature)
    aragoniten = np.asarray([arag]) if np.isscalar(arag) or arag.ndim == 0 else np.asarray(arag)

    optimal_mort = (-19.4 + 11.5 * temperature_optimal - 32.7 * arag_optimal)/100
    arag_only = (-19.4 + 11.5 * temperature_optimal - 32.7 * (weight_factor*aragoniten + (1 - weight_factor)*arag_optimal))/100
    temp_only = (-19.4 + 11.5 * temperature_array - 32.7 * arag_optimal)/100
    both = (-19.4 + 11.5 * temperature_array - 32.7 * (weight_factor*aragoniten + (1 - weight_factor)*arag_optimal))/100

    arag_only = (arag_only - optimal_mort)/optimal_mort
    temp_only = (temp_only - optimal_mort)/optimal_mort
    both = (both - optimal_mort)/optimal_mort

    all_possibilities = np.stack((arag_only, temp_only, both), axis=0)

    scale_factor = np.nanmax(all_possibilities, axis=0)
    delta_mortality = scale_factor*rate

    delta_mortality[delta_mortality<0] = 0

    return delta_mortality


def mortality(
        pteropod_list, rate_g0_0=0.211, rate_g0_1=0.09, rate_g0_2=0.09,
        rate_g0_3=0.09, rate_g1_0=0.142, rate_g1_1=0.01, rate_g1_2=0.01,
        rate_g1_3=0.01, arag_optimal=1.5, temp_optimal=14, weight_factor=None, day=None,
        outfile='/cluster/scratch/ursho/output_simulation_extremes/mortalities/',
        ensure_survival=False, dissolution=None):
    """Select subsets of the individuals found in pteropod list according to their stage and
    generation. These are then used in 'get_number_individuals' function to calculate the mortality
    rates. Returns list of indeces of individuals that die, and the pteropod array with updated
    values for survival. UHE 25/09/2020

    Keyword arguments:
    pteorpod_list -- array containing all state variables characterizing the pteropods
    rate_gX_Y -- beta mortality rates for each stage and generation
    day -- list containing year, version, day and control for saving causes for mortalitites
          (background, old age, spawning,...; default None)
    outfile -- output directory

    """

    num_dead_nat = 0
    num_dead_dis = 0
    num_dead_old = 0
    num_dead_dis_arag = 0

    all_rands = np.array([])
    all_delta_rates = np.array([])
    all_rates = np.array([])
    all_tmps = np.array([])

    if weight_factor is None:
        weight_factor = pteropod_list[:,1].copy()*0 + 1

    for gen in range(2):
        for i in range(4):

            tmp = np.squeeze(np.argwhere((pteropod_list[:,2] == i) & (pteropod_list[:,1]%2 == gen)))
            if tmp.size == 1:
                tmp = np.array([tmp])

            if tmp.size > 0:
                num_ind, mort_rate = get_number_individuals(tmp,i,gen,rate_g0_0,rate_g0_1,
                                                            rate_g0_2,rate_g0_3, rate_g1_0,
                                                            rate_g1_1,rate_g1_2,rate_g1_3)

                num_ind = (num_ind > 0) + (not ensure_survival)

                if num_ind:
                    rands = np.random.random(tmp.size)

                    aragt_nox = pteropod_list[tmp,13] if dissolution is None else pteropod_list[tmp,13]*0 + 1.5
                    aragt_x = pteropod_list[tmp,18]

                    temperature = pteropod_list[tmp,15]
                    wt_temp = weight_factor[tmp]

                    #calculate change in mortality in different scenarion
                    delta_rate_control = mortality_delta(temperature, aragt_nox, mort_rate,
                                                         weight_factor=wt_temp,
                                                        temperature_optimal=temp_optimal,
                                                        arag_optimal=arag_optimal)

                    delta_rate_xarag = mortality_delta(temperature, aragt_x, mort_rate,
                                                       weight_factor=wt_temp,
                                                        temperature_optimal=temp_optimal,
                                                        arag_optimal=arag_optimal)

                    #check mortality due to natural mortality
                    pteropod_list[tmp[np.squeeze(rands < mort_rate)],5] = 0
                    #count number of dead pteorpods (natural death)
                    num_dead_nat += np.sum(rands < mort_rate)

                    #flag who would die under the different scenarios
                    death_control = np.squeeze(rands < mort_rate + delta_rate_control)
                    death_xarag = np.squeeze(rands < mort_rate + delta_rate_xarag)

                    #death due to arag extreme only
                    death_flag_xarag = np.squeeze((death_control == False)*(death_xarag == True))

                    pteropod_list[tmp[np.squeeze(rands < mort_rate + delta_rate_control)],5] = 0

                    #mark individuals that would be dead in different scenarios
                    pteropod_list[tmp[death_flag_xarag],19] += 1

                    #count increase in mortality (dissolution death)
                    num_dead_dis += np.sum((rands >= mort_rate)&(rands < mort_rate + delta_rate_control))
                    num_dead_dis_arag += np.sum((rands >= mort_rate)&(rands < mort_rate + delta_rate_xarag))

                    all_rands = np.hstack((all_rands,np.squeeze(rands)))
                    all_delta_rates = np.hstack((all_delta_rates,np.squeeze(delta_rate_control)))
                    all_rates = np.hstack((all_rates,np.squeeze(delta_rate_control*0+mort_rate)))
                    all_tmps = np.hstack((all_tmps,np.squeeze(tmp)))

    #remove pteropods that are too old, that already spawned, or that are beached (given as 0 food)
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,4] > 300)),5] = 0
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,6] >= 1)),5] = 0

    #count mortality due to age
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,16] <= 0)),5] = 0
    num_dead_old = np.sum(pteropod_list[:,5]==0) - num_dead_dis - num_dead_nat

    #get overall list of dead particles, needed for parcels objects
    dead_particles = np.squeeze(np.argwhere(pteropod_list[:,5] == 0)).astype(int)

    num_ptero = np.squeeze(np.argwhere(pteropod_list[:,5] == 1)).astype(int)

    if num_ptero.size == 1:
        num_ptero = np.array(num_ptero)
        new_pteropod_list = np.reshape(pteropod_list[num_ptero,:].copy(),(1,22))
    else:
        new_pteropod_list = pteropod_list[num_ptero,:].copy()

    #==========================================================
    # Save rands, rate, dealta_rate, index, UHE 02/06/2021
    #==========================================================
    if not day is None:
        assert len(day) == 4,"Argument 'day' should have the lenght 4 (year,version,day,control)"

        outfile_mort = outfile
        file_name = f'Mortality_year_{day[0]}_V_{day[1]}_control_{day[3]}.csv'
        if not os.path.exists(outfile_mort):
            os.makedirs(outfile_mort)

            with open(outfile_mort+file_name,'w') as file:
                file.write('day, background_mort, dissolution_mort, extreme_mort, age_mort')
                file.write('\n')

        array_save = np.array([day[2],num_dead_nat,num_dead_dis,num_dead_dis_arag,num_dead_old])
        save_str = ''
        for i,num in enumerate(array_save):
            save_str += str(int(num))
            if i < 4:
                save_str += ', '
        with open(outfile_mort + file_name,'a') as file:
            file.write(save_str)
            file.write('\n')

    return dead_particles,new_pteropod_list

def calculate_shell_carbonate(length):
    """This function calculates the calcium carbonate content in the shell of a pteropod
    The function is taken from Bednarsek et al., Deep Sea Research Part 2 59: 105-116 (2012)
    Returns the calcium carbonate content in mg CaCO3 on position 0 and the dry weight in mg DW on
    position 1. UHE 25/09/2020

    Keyword arguments:
    length -- Shell size in mm

    """
    dry_weight = (0.137*length**1.5005)
    shell_calc = dry_weight*0.25*0.27*8.33

    return shell_calc, dry_weight

def calculate_dissolution_calcification(length, damage, delta_length, arag, weight_factor=1,
                                        size_threshold=0.5):
    """This function calculates the loss and gain of CaCO3 given the current size, growth function,
    and exposure to aragonite saturation states. The function first determines the dissolution, and
    compares it to the calcium carbonate that could be produced under non-corrosive conditions
    to calculate the net gain/loss of CaCO3. The pteropod can then either repair the damage as much
    as possible or grow. Returns the new shell size in mm after dissolution/lack of accretion on
    position 0, and the accumulated damage in mg CaCO3 on position 1. UHE 13/01/2022


    Keyword arguments:
    length -- Shell size in mm
    damage -- Current accumulated damage in mg CaCO3
    delta_length -- Current potential increase under idealized conditions in mm
    Arag -- Aragonite saturation state experienced by pteropod
    weight_factor -- weight factor of exposure. Given as percentage of exposure in the day.
                     Default value is 1
    size_threshold -- size threshold for shell. Given in mm. Default value is 0.5 for protoconch.
    """

    #ensure the input is an array even if scalars are given as input
    length_array = np.asarray([length]) if np.isscalar(length) or length.ndim == 0 else np.asarray(length)
    damagen = np.asarray([damage]).astype(np.float64) if np.isscalar(damage) or damage.ndim == 0 else np.asarray(damage).astype(np.float64)
    delta_length_array = np.asarray([delta_length]) if np.isscalar(delta_length) or delta_length.ndim == 0  else np.asarray(delta_length)
    arag_array = np.asarray([arag]) if np.isscalar(arag) or arag.ndim == 0  else np.asarray(arag)
    weight_factor = np.asarray([weight_factor]) if np.isscalar(weight_factor) or weight_factor.ndim == 0  else np.asarray(weight_factor)

    weight_factor[weight_factor>1] = 1
    weight_factor[weight_factor<0] = 0
    #if they don't have a shell then no dissolution occurs.
    weight_factor[length_array<size_threshold] = 0

    shell_calc, _ = calculate_shell_carbonate(length_array)

    #calculate the shell loss, and scale it by the weight_factor,
    # i.e., the duration of the exposure. UHE 21.09.2022 following discussion with Meike
    loss = (65.76 * np.exp(-4.7606*arag_array)*shell_calc/100)*weight_factor

    zero = np.zeros(length.size)
    length_new = length_array.copy()

    length_pot = length_array + delta_length_array
    shell_calc_new, _ = calculate_shell_carbonate(length_pot)

    net = shell_calc_new - shell_calc - loss

    flag_smaller_damage = net > damagen
    if np.sum(flag_smaller_damage) > 0:
        shell_new = shell_calc[flag_smaller_damage] + net[flag_smaller_damage] - damagen[flag_smaller_damage]
        length_new[flag_smaller_damage] = (shell_new/(0.25*0.27*8.33*0.137))**(1/float(1.5005))
    damagen = np.max([zero,damagen - net],axis=0)

    return length_new, damagen


def cardinal_temperature_model_inflection(temperature, temperature_opt, temperature_max,
                                          temperature_min):
    """This function uses the cardinal temperature model with inflection to calculate the
    reduction in growth rate. Returns coefficient alpha to scale growth rate.
    UHE 05/09/2022

    Keyword arguments:
    temperature -- Temperature experienced by the pteropods
    temperature_opt -- Optimal temperature for growth
    temperature_max -- Maximum temperature for growth
    temperature_min -- Minimum temperature for growth
    """
    temperature = np.asarray([temperature]) if np.isscalar(temperature) or temperature.ndim == 0 else np.asarray(temperature)

    lambda_f = (temperature - temperature_max)*(temperature - temperature_min)**2
    beta = (temperature_opt - temperature_min)*((temperature_opt - temperature_min)*(temperature - temperature_opt) - (temperature_opt - temperature_max)*(temperature_opt + temperature_min - 2*temperature))

    alpha_ctmi = lambda_f/beta
    alpha_ctmi[temperature<=temperature_opt] = 1
    alpha_ctmi[temperature<temperature_min] = 0
    alpha_ctmi[temperature>temperature_max] = 0

    return alpha_ctmi

def shell_growth(pteropod_list, growth_fct_gen0, arag=4, arag_x=None, temperature=18, q10=1.3,
                 food=7, temperature_0=14.5, half_saturation=4.8,temperature_opt=18,
                 temperature_max=31,temperature_min=0.6,weight_factor=1,size_threshold=0.5,
                 day=None,outfile='/cluster/scratch/ursho/output_simulation_extremes/Growth/'):
    """This function determines the net shell growth given the aragonite saturation state, current
    size, and generation. Returns array containing updated attributes characterizing the pteropods.
    UHE 25/09/2020

    Keyword arguments:
    pteorpod_list -- Array containing all state variables characterizing the pteropods
    growth_fct_gen0 -- Shell size as function of time for spring (X=0) and winter (X=1) generation
    arag -- Aragonite saturation state experiences by each pteropod on one day
    temperature -- Temperature. Default value was set to 16 to simulate optimal conditions
    food -- Food/Phytoplankton carbon available. Default value 7 was chosen to simulate optimal
            conditions
    temperature_0 -- Refernce temperature for the growth rate. Default value set to 14.5 according
                        to Wang et al. 2017
    half_saturation -- Food/Phytoplankton carbon half-saturation constant. The default value is set
                         to 2.6
    temperature_opt -- Optimal temperature for growth. Default value is set to 18
    temperature_max -- Maximum temperature for growth. Default value is set to 31
    temperature_min -- Minimum temperature for growth. Default value is set to 0.6
    weight_factor -- Weight factor of exposure to undersaturation
    size_threshold -- Size threshold for shell dissolution. Default value is 0.5mm
    day -- list containing year, version, day and control for saving all growth rates for each day
            (default None)
    outfile -- output directory

    """
    #If dissolution is turned off, then create array with experienced aragonite saturation states
    #that are too high to not have an effect on shell growth
    if np.ndim(arag) == 0:
        #only happens if there is no input or only a single pteorpod in the list
        arag = pteropod_list[:,5].copy()*arag
        arag_x = pteropod_list[:,5].copy()*arag_x

    list_days_of_growth = np.arange(pteropod_list.shape[0])
    #increase shell size according to temp and food, UHE 17/03/2021
    if list_days_of_growth.size > 0:

        #get the growth rate as fraction of size increase in each time step
        growth_rate = [(growth_fct_gen0[i]-growth_fct_gen0[i-1])/growth_fct_gen0[i-1] for i in range(1,len(growth_fct_gen0))]
        #repeat the first at the beginning and the last one at the end
        growth_rate.insert(0,growth_rate[0])
        growth_rate.append(growth_rate[-1])
        #convert to numpy array
        growth_rate = np.array(growth_rate)

        #current length
        length = pteropod_list[list_days_of_growth,3]
        length_x = pteropod_list[list_days_of_growth,20]
        #ensure the structure of L is correct if there is only one pteropod or multiple pteropods
        if length.shape[0] != 1:
            length = np.squeeze(length)
        if length_x.shape[0] != 1:
            length_x = np.squeeze(length_x)
        #calculate distance to reference and find index with minimum distance
        #the sizes should be rounded to 4 decimal places at a later stage
        pos_idx = np.array([np.squeeze(np.argwhere(abs(growth_fct_gen0-i) == abs(growth_fct_gen0-i).min())) for i in np.around(length,4)])
        pos_idx_x = np.array([np.squeeze(np.argwhere(abs(growth_fct_gen0-i) == abs(growth_fct_gen0-i).min())) for i in np.around(length_x,4)])
        food_effect = food/(half_saturation + food)

        alpha_ctmi_temperature_nox = cardinal_temperature_model_inflection(temperature,
                                                                           temperature_opt,
                                                                           temperature_max,
                                                                           temperature_min)

        rate = growth_rate[pos_idx]*q10**((temperature - temperature_0)/10) * food_effect * alpha_ctmi_temperature_nox
        rate_x = growth_rate[pos_idx_x]*q10**((temperature - temperature_0)/10) * food_effect * alpha_ctmi_temperature_nox

        delta_length = rate*length
        delta_length_x = rate_x*length_x

        damage = np.squeeze(pteropod_list[list_days_of_growth,14])
        experienced_arag = arag[list_days_of_growth]
        pteropod_list[list_days_of_growth,3],pteropod_list[list_days_of_growth,14]  = calculate_dissolution_calcification(length,damage,delta_length,experienced_arag, weight_factor, size_threshold)

        damage_x = np.squeeze(pteropod_list[list_days_of_growth,21])
        experienced_arag_x = arag[list_days_of_growth] if arag_x is None else arag_x[list_days_of_growth]
        pteropod_list[list_days_of_growth,20],pteropod_list[list_days_of_growth,21]  = calculate_dissolution_calcification(length_x,damage_x,delta_length_x,experienced_arag_x, weight_factor, size_threshold)

        #==========================================================
        # Save rate, delta_L, T, F, damage, UHE 02/06/2021
        #==========================================================
        if not day is None:
            assert len(day) == 4, "Argument 'day' should contain year, version, day, and control"
            #save variables as csv file
            array_save = np.array([rate,delta_length,damage,temperature,food])
            outfile_growth = f'{outfile}year_{day[0]}_V_{day[1]}_control_{day[3]}/'

            if not os.path.exists(outfile_growth):
                os.makedirs(outfile_growth)
            np.savetxt(f'{outfile_growth}Growth_Day_{int(day[2])}.csv', array_save, delimiter=',')

    return pteropod_list

def development(pteropod_list,growth_function):
    """This function determines the life stage depending on the size of the pteropods.
    And increases the growth time by one day. Returns array containing updated attributes
    characterizing the pteropods. UHE 25/09/2020

    Keyword arguments:
    pteropod_list -- Array containing all state variables characterizing the pteropods
    growth_function -- Shell size as function of time

    """
    #adapt stages using thresholds
    pteropod_list[np.squeeze(np.where(pteropod_list[:,3] >= growth_function[6])).astype(int),2] = 1
    pteropod_list[np.squeeze(np.where(pteropod_list[:,3] >= growth_function[30])).astype(int),2] = 2
    pteropod_list[np.squeeze(np.where(pteropod_list[:,3] >= growth_function[90])).astype(int),2] = 3

    #increase days of growth if they survive
    pteropod_list[:,4] = pteropod_list[:,4] + 1

    return pteropod_list

def spawning(pteropod_list, current_generation, next_id, num_eggs=500, delta_err=20):
    """This function subsets the adult pteropods of a given generation, and determines which
    pteropods are ready to spawn eggs. Returns the array containing the updated attributes
    characterizing the pteropods, the next largest ID, and the current generation spawning.
    UHE 25/09/2020

    Keyword arguments:
    pteropod_list -- Array containing all state variables characterizing the pteropods
    current_generation -- Identifier of the current generation that will spawn the next generation
    next_id -- The largest ID + 1 out of the entire population
    num_eggs -- Number of eggs spawned per adult of a single spawning event
    delta_err -- increase in the Egg Release Readiness (ERR) index per day as 1/delta_err

    """

    #add to the ERR if adults have repaired the damage
    pteropod_list[np.squeeze(np.argwhere(pteropod_list[:,2] == 3)).astype(int),7] += 1/(delta_err/2)

    available_generations = pteropod_list[np.argwhere(pteropod_list[:,2] == 3),1]
    if len(np.unique(available_generations)) >= 1  and  max(np.unique(available_generations)) !=  current_generation:
        current_generation = max(np.unique(available_generations))

    #get number of adults in the current generation that can produce eggs

    idx = 0
    #get number of adults in the current generation that can produce eggs
    adults_ind = np.squeeze(np.argwhere((pteropod_list[:,2] == 3) &
                                (pteropod_list[:,1]%2 == current_generation%2) &
                                (pteropod_list[:,7] >= 1.0) &
                                (pteropod_list[:,6] == idx))).astype(int)


    #get the total number of new eggs(particles)
    if adults_ind.size > 0:
        #for each entry in adult, create egg more entries
        #get the generation, Parent_ID, Shell_size, time_of_birth
        generation = np.squeeze(pteropod_list[adults_ind,1])
        parent_id = np.squeeze(pteropod_list[adults_ind,0])
        parent_shell_size = np.squeeze(pteropod_list[adults_ind,3])
        time_birth = np.squeeze(pteropod_list[adults_ind,12])
        flag_would_be_dead = (np.squeeze(pteropod_list[adults_ind,19]) > 0).astype(int)

        eggs = np.random.rand(adults_ind.size, 22) #21
        #ID
        eggs[:,0] = -1
        #generation
        eggs[:,1] = generation+1
        #stage
        eggs[:,2] = 0
        #shell_size
        eggs[:,3] = 0.15
        #days_of_growth
        eggs[:,4] = 0
        #survive
        eggs[:,5] = 1
        #num. spawning events
        eggs[:,6] = 0
        #ERR distribution around -1 and std 0.1
        eggs[:,7] = np.random.normal(-1,0.1,adults_ind.size)
        #spawned
        eggs[:,8] = 0
        #Parent_ID
        eggs[:,9] = parent_id
        #Parent_shell_size
        eggs[:,10] = parent_shell_size
        #time_birth
        eggs[:,11] = -1
        #current_time
        eggs[:,12] = time_birth
        #aragonite, but used as parent index in matrix
        eggs[:,13] = adults_ind
        #damage accumulated
        eggs[:,14] = 0
        #temperature
        eggs[:,15] = 0
        #food
        eggs[:,16] = 0
        #shell thickness
        eggs[:,17] = 0
        #aragonite X scenario
        eggs[:,18] = np.nan
        #would be dead flag
        eggs[:,19] = flag_would_be_dead
        #shell size extremes
        eggs[:,20] = 0.15
        #damage extremes
        eggs[:,21] = 0

        egg_list = np.repeat(eggs, repeats=int(num_eggs/(idx+1)), axis=0)
        egg_list[:,0] = np.arange(next_id,next_id+egg_list.shape[0])

        pteropod_list = np.concatenate((pteropod_list,egg_list))
        next_id = max(pteropod_list[:,0])+1
        pteropod_list[adults_ind,6] = 1

    return pteropod_list, next_id, current_generation
