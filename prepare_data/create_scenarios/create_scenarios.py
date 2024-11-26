#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:47:16 2021

This script is run from the terminal using:

python create_scenarios.py --variable "Aragonite" --experiment "no_effect" --year 1999


@author: ursho
"""
import os
import calendar
import time
import logging
import argparse
import yaml

from typing import Tuple
from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class ConfigParameters():

    #logger name
    logger_name: str

    #variable name in ROMS file
    variable: str
    variable_climatology: str

    #name of the scenario experiment
    calculate_no_effect_scenario: bool

    #year from which a scenario is created
    year: int

    # threshold to define an effect, and how to apply it
    threshold: float
    effect_above_threshold: bool

    #climatology, hindcast, and constant CO2 files with path
    climatology_file: str
    hindcast_file: str
    constant_file: str

    #path where to store output and name of the file
    output_dir: str
    output_file: str

    def __post_init__(self):

        self.variable_climatology = self.variable
        if calendar.isleap(self.year):
            self.variable_climatology = f'{self.variable}_leap'

        self.hindcast_file = self.hindcast_file.format(self.year)
        self.constant_file = self.constant_file.format(self.year)
        self.output_file = self.output_dir + self.output_file.format(self.year)


def read_config_files(config_file: str, year: int = None, config_class: ConfigParameters = ConfigParameters):


    assert '.yaml' in config_file.lower(), "The configuration file should be a '.yaml' file"

    with open(config_file, encoding='utf-8') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    #overwrite the year argument by the terminal input
    if not year is None:
        config_list['year'] = int(year)

    config = config_class(**config_list)

    return config


def parse_inputs() -> str:
    """
    This function parses command line inputs for producing scenarios.

    Returns:
    str: The name of the configuration file.

    Note:
    The function uses the argparse library to parse command line inputs. It requires a
    configuration file (--config_file) as an input.
    """

    parser = argparse.ArgumentParser(description="Produce scenarios")
    parser.add_argument("--config_file", required=True, type=str,
                        help="Name of the configuration file")
    parser.add_argument("--start_year", required=False, type=float, default=None,
                        help="Year to calculate scenario")
    parser.add_argument("--end_year", required=False, type=float, default=None,
                        help="Year to calculate scenario")

    args = parser.parse_args()

    return args.config_file, args.start_year, args.end_year


def create_scenarios(config) -> None:
    """
    This function calculates either the extremes of a given environmental variable, or calculates a
    scenario without effects of the same environmental variable.

    Parameters:
    file_name_one_hind (str): The path to the hindcast file.
    file_name_one_const (str): The path to the baseline file.
    varname (str): The name of the variable.
    varname_climatology (str): The name of the climatology variable.
    climatology_path (str): The path to the climatology file.
    threshold (float): The threshold for the variable.
    out_file (str): The path to the output file.

    Returns:
    None

    Note:
    The function uses the xarray library to open the data files and to save the output. It logs the
    start and end of the process, and the time it took to save the file.
    """

    logging.info('Loading data')
    year_data_hind = xr.open_dataset(config.hindcast_file)
    year_data_const = xr.open_dataset(config.constant_file)
    climatology = xr.open_dataset(config.climatology_file)

    #create scenarios depending on configuration file

    # Create no effect scenario
    if config.calculate_no_effect_scenario:
        logging.info('Creating no effect scenario for {}'.format(config.variable))

        if config.effect_above_threshold:
            mask_condition_threshold = year_data_const[config.variable].values > config.threshold

        else:
            mask_condition_threshold = year_data_const[config.variable].values < config.threshold

        year_data_const[config.variable].values = np.where(mask_condition_threshold,
                                                       config.threshold, year_data_const[config.variable].values)

    # Create scenarios
    else:
        logging.info('Creating scenario for {}'.format(config.variable))
        year_data_const['Extremes'] = year_data_const[config.variable]


        if config.effect_above_threshold:
            mask_extreme_threshold = (climatology[config.variable_climatology].values <= config.threshold) & \
                                                (year_data_hind[config.variable].values > (config.threshold)) & \
                                                (year_data_const[config.variable].values > config.threshold)

            mask_extremes = (climatology[config.variable_climatology].values <= config.threshold) & \
                                        (year_data_hind[config.variable].values > (config.threshold))

        else:
            mask_extreme_threshold = (climatology[config.variable_climatology].values >= config.threshold) & \
                                                (year_data_hind[config.variable].values < (config.threshold)) & \
                                                (year_data_const[config.variable].values < config.threshold)

            mask_extremes = (climatology[config.variable_climatology].values >= config.threshold) & \
                                        (year_data_hind[config.variable].values < (config.threshold))

        year_data_const['Extremes'].values = np.where(mask_extremes, 1, 0)
        year_data_const[config.variable].values = np.where(mask_extreme_threshold, config.threshold,
                                                   year_data_const[config.variable].values)


    logging.info(f'Saving file for year {config.year}')
    start_time = time.time()

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    year_data_const.to_netcdf(config.output_file)

    logging.info(f'Saved file in {time.time()-start_time} seconds')


# ==================================================================================================
#                                   MAIN FUNCTION
# ==================================================================================================
if __name__ in "__main__":



    #parse terminal arguments
    MY_CONFIG_FILE, START_YEAR, END_YEAR = parse_inputs()

    if not os.path.exists('log_files/'):
        os.makedirs('log_files/')


    if not START_YEAR is None:
        for year in range(int(START_YEAR), int(END_YEAR) + 1):

            config = read_config_files(MY_CONFIG_FILE, year)
            if year == START_YEAR:
                #setup logger
                logging.basicConfig(filename=f'log_files/{config.logger_name}_scenarios.log', level=logging.DEBUG)
            create_scenarios(config)


    else:
        config = read_config_files(MY_CONFIG_FILE, START_YEAR)
        #setup logger
        logging.basicConfig(filename=f'log_files/{config.logger_name}_scenarios.log', level=logging.DEBUG)
        create_scenarios(config)

