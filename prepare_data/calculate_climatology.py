#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:32:32 2022

Adapted from koehne.
Script to calculate X-day rolling daily climatology of a roms model output

Script usage from terminal:

1. For Aragonite:
    python calculate_climatology.py --config_file Aragonite_climatology_config.yaml
2. For Temperature
    python calculate_climatology.py --config_file Temperature_climatology_config.yaml

@author: ursho
"""

#########################
## Load needed modules ##
#########################

import os
import logging
import argparse
import yaml

import numpy as np
import xarray as xr

from typing import List, Tuple
import numpy.typing as npt
from dataclasses import dataclass
from tqdm import tqdm
from my_roms_tools import GetGrid, writeNetCDF

@dataclass
class ConfigParameters():
    """
    A data class to define paths and parameters used to calculate the climatology.

    Attributes:
    gridfile (str): The path to the ROMS grid file.
    climdir (str): The path to the directory where the climatology is stored.
    modeldir (str): The path to the directory where individual files are stored.
    individual_file (str): The name of the files used to calculate the climatology.
    attribute_file (str): The path to a simulation file with all attributes.
    varname (str): The name of the variable.
    run_name (str): The name of the run.
    start_year (int): The start year to include in the climatology.
    end_year (int): The end year to include in the climatology.
    size_rolling (int): The size of the rolling window.
    days_per_year (int): The number of days in a year.

    Methods:
    __post_init__: A method that runs after the class is instantiated. It sets the attribute_file
    attribute.
    """
    #path to ROMS grid file
    gridfile: str

    #path to directory where climatology is stored
    climdir: str

    #path to directory where individual files are stored
    modeldir: str

    #name of files used to calculate the climatology
    individual_file: str

    #path to a simulation file with all attributes (former filepath)
    attribute_file: str

    #name of variable
    varname: str

    #name of run
    run_name: str

    #years to include in climatology
    start_year: int
    end_year: int

    #size of rolling window
    size_rolling: int

    days_per_year: int

    def __post_init__(self):

        self.attribute_file = self.modeldir + self.individual_file.format(1984)


def read_config_files(config_file: str, config_class: ConfigParameters = ConfigParameters):
    """
    This function reads a configuration file and returns a configuration object.

    Parameters:
    config_file (str): The path to the configuration file. It should be a '.yaml' file.
    config_class (class, optional): The class to create the configuration object. Defaults to
    ConfigParameters.

    Returns:
    object: An instance of the configuration class with the parameters from the configuration file.

    Raises:
    AssertionError: If the configuration file is not a '.yaml' file or if the size of the rolling
    window is not an odd integer.

    Note:
    The function uses the yaml library to read the configuration file.
    """

    assert '.yaml' in config_file.lower(), "The configuration file should be a '.yaml' file"

    with open(config_file, encoding='utf-8') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    config = config_class(**config_list)

    assert config.size_rolling%2 == 1, "The size of the rolling window should be an odd integer."

    return config

def parse_inputs() -> str:
    """
    This function parses command line inputs for calculating daily climatology.

    Returns:
    str: The name of the configuration file.

    Note:
    The function uses the argparse library to parse command line inputs. It requires a
    configuration file (--config_file) as an input.
    """

    parser = argparse.ArgumentParser(description="Calculate daily climatology")
    parser.add_argument("--config_file", required=True, type=str,
                        help="Name of the configuration file")

    args = parser.parse_args()

    return args.config_file

def calculate_rolling_window(size_rolling: int) -> Tuple[npt.NDArray, int]:
    """
    This function calculates the rolling window for a given size.

    Parameters:
    size_rolling (int): The size of the rolling window. It should be an odd integer.

    Returns:
    tuple: A tuple containing the rolling window and its width.

    Raises:
    AssertionError: If the size of the rolling window is not an odd integer.

    Note:
    The rolling window is a numpy array ranging from negative half size to positive half size.
    """

    assert size_rolling%2 == 1, "The size of the rolling window should be an odd integer."

    width_rolling = np.floor(size_rolling/2).astype(int)
    window = np.arange(-width_rolling,width_rolling+1)

    return window, width_rolling


def calculate_running_mean(modeldir: str, individual_file: str, varname: str,
                           period: npt.NDArray, days_per_year: int,
                           size_window: int, doy: int,
                           dims: List[int]=None) -> Tuple[npt.NDArray, List[int]]:
    """
    This function calculates the running mean of a timeseries.

    Parameters:
    modeldir (str): The directory of the model.
    individual_file (str): The individual file to calculate the running mean.
    varname (str): The variable name in the climatology data.
    period (list): The period to calculate the running mean. It should be a list of years.
    days_per_year (int): The number of days in a year.
    size_window (int): The size of the rolling window.
    doy (int): The day of the year.
    dims (tuple, optional): The dimensions of the climatology data. If not provided, it will be set
    based on the data.

    Returns:
    tuple: A tuple containing the climatology data and its dimensions.

    Note:
    The function assumes that the individual file is a NetCDF file and uses the xarray library to
    read it.
    """

    start_year = period[0]
    end_year = period[-1]
    if dims is None:
        logging.info('Setting dimensions of the climatology')
        ds_dum = xr.open_dataset(modeldir+individual_file.format(start_year))
        dims = ds_dum.variables[varname].shape

    #allocate variable for climatology
    climatology = np.zeros((dims[1],dims[2],dims[3]))

    for year in period:

        file_name = individual_file.format(year)
        file_name_next = individual_file.format(year+1)
        file_name_previous = individual_file.format(year-1)

        window, width_rolling_window = calculate_rolling_window(size_window)
        window += doy

        file_data = xr.open_dataset(modeldir+file_name)

        if doy < width_rolling_window:
            vals = file_data.variables[varname][0:window[-1]+1,:,:,:].sum(axis=0)

            if year == start_year:
                vals_previous = file_data.variables[varname][window[0]:,:,:,:].sum(axis=0)

            else:
                file_data_prev = xr.open_dataset(modeldir+file_name_previous)
                vals_previous = file_data_prev.variables[varname][window[0]:,:,:,:].sum(axis=0)

            climatology += vals + vals_previous

        elif doy >= days_per_year-width_rolling_window:
            window = window%days_per_year
            vals = file_data.variables[varname][window[0]:,:,:,:].sum(axis=0)

            if year == end_year:
                vals_next = file_data.variables[varname][0:window[-1]+1,:,:,:].sum(axis=0)

            else:
                file_data_next = xr.open_dataset(modeldir+file_name_next)
                vals_next = file_data_next.variables[varname][0:window[-1]+1,:,:,:].sum(axis=0)

            climatology += vals + vals_next

        else:
            vals = file_data.variables[varname][doy-3:doy+4,:,:,:].sum(axis=0)
            climatology += vals


    return climatology/len(period)/len(window), dims

def store_running_mean(climatology: npt.NDArray, name_running_mean: str, varname: str,
                       gridfile: str, attribute_file: str, outpath: str) -> None:
    """
    This function stores the running mean climatology in a NetCDF file.

    Parameters:
    climatology (numpy.ndarray): The climatology data. It should be a 2D or 3D numpy array.
    name_running_mean (str): The name of the running mean to be stored.
    varname (str): The variable name in the climatology data.
    gridfile (str): The path to the grid file.
    attribute_file (str): The path to the attribute file.
    outpath (str): The output path where the NetCDF file will be stored.

    Raises:
    AssertionError: If the climatology data is not a 2D or 3D array.

    Note:
    The function will create the output directory if it does not exist.
"""

    # create directory if it does not exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    roms_grid = GetGrid(gridfile)
    roms_grid.getAttrs(attribute_file)
    roms_grid.getLatLon()
    roms_grid.getArea()
    roms_grid.getAngle()

    my_data = {varname:climatology}

    assert climatology.ndim < 4, 'Initial dataset has to be in at least 4D'

    if climatology.ndim == 3:
        logging.info('Data is in 3D')
        my_dimensions = {varname:{'dims':('s_rho','eta_v','xi_rho'),
                                  'dimlens':list(np.shape(climatology))}}
    elif climatology.ndim == 2:
        logging.info('Data is in 3D')
        my_dimensions = {varname:{'dims':('eta_rho','xi_rho'),
                                  'dimlens':list(np.shape(climatology))}}

    writeNetCDF(roms_grid,my_data,outpath+name_running_mean,dDims=my_dimensions)


def main(config_file: str) -> None:
    """
    This function calculates the climatology of a dataset, stores it in daily files, merges the
    daily files, and extends the climatology by leap year.

    Parameters:
    config_file (str): The path to the configuration file.

    The configuration file should contain the following parameters:
    - modeldir: The directory of the model.
    - individual_file: The individual file to calculate the running mean.
    - varname: The variable name in the climatology data.
    - start_year: The start year of the period.
    - end_year: The end year of the period.
    - days_per_year: The number of days in a year.
    - size_rolling: The size of the rolling window.
    - run_name: The name of the run.
    - gridfile: The path to the grid file.
    - attribute_file: The path to the attribute file.
    - climdir: The directory to store the climatology data.

    Note:
    The function will create the output directory if it does not exist.
    """

    config = read_config_files(config_file, config_class=ConfigParameters)

    # =============================================================================
    # Read in the model grid
    # =============================================================================
    logging.info('Reading model grid information')

    period = np.arange(config.start_year, config.end_year+1)

    # =============================================================================
    # Calculate climatology
    # =============================================================================

    dims = None
    for doy in tqdm(np.arange(0,config.days_per_year), desc='Calculating climatology'):

        climatology,dims = calculate_running_mean(config.modeldir, config.individual_file,
                                                  config.varname, period, config.days_per_year,
                                                  config.size_rolling,doy,dims)

        name_running_mean = '{}_daily_clim_{}_{}_{}_doy_{:03d}.nc'.format(config.run_name,
                                                                          config.start_year,
                                                                          config.end_year,
                                                                          config.varname, doy+1)

        store_running_mean(climatology, name_running_mean, config.varname, config.gridfile,
                           config.attribute_file, config.climdir)

    # Merging the daily files together
    doy_files = f'{config.climdir}{config.run_name}_daily_clim_{config.start_year}_{config.end_year}_{config.varname}_doy_*.nc'
    combined_file = f'{config.climdir}{config.run_name}_daily_clim_{config.start_year}_{config.end_year}_{config.varname}_{config.size_rolling}day.nc'
    command = f'cdo cat {doy_files} {combined_file}'

    os.system(command)

    #remove individual doys
    os.system(f'rm {doy_files}')

    # ==================================================================================
    # Extend climatology by leap year
    # ==================================================================================

    climatology = xr.open_dataset(f'{combined_file}')
    no_leap_var = climatology[config.varname].values

    leap_var = np.ones((366,64,518,604))*np.nan

    leap_var[:58,:,:,:] = no_leap_var[:58,:,:,:]
    leap_var[59:,:,:,:] = no_leap_var[58:,:,:,:]

    leap_var[58,:,:,:] = np.mean(no_leap_var[57:59,:,:,:],axis=0)

    climatology[f'{config.varname}_leap'] = (('time_leap', 's_rho', 'eta_rho', 'xi_rho'), leap_var)
    climatology.assign_coords({"time_leap": np.zeros((366,))})

    extended_file = f'{config.climdir}Extended_{config.run_name}_daily_clim_{config.start_year}_{config.end_year}_{config.varname}_{config.size_rolling}day.nc'
    extended_file = f'{config.climdir}{config.run_name}_climatology_{config.size_rolling}_day_moving_average.nc'
    climatology.to_netcdf(f'{extended_file}')

    os.system(f'rm {combined_file}')

# =======================================
#            Main Function
# =======================================
if __name__ in "__main__":

    logging.basicConfig(level=logging.DEBUG)

    MY_CONFIG_FILE = parse_inputs()

    main(MY_CONFIG_FILE)
