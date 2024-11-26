#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:50:10 2022

@author: ursho
copied from Martin Frischkneckt
"""

#!/usr/bin/env python
import netCDF4
import sys
import datetime

import numpy as np
from typing import Dict
import numpy.typing as npt

class GetGrid:
    """
    A class to read the basics of ROMS setup for further use in other functions and classes.

    Attributes:
    grid_file (str): The grid file.
    grid_data (netCDF4.Dataset): The grid data from the grid file.
    mask_rho (np.ndarray): The mask_rho variable from the grid data.
    FillValue (float): The _FillValue attribute of the mask_rho variable.
    rho_y (int): The y-dimension of the mask_rho variable.
    rho_x (int): The x-dimension of the mask_rho variable.
    ncclm (netCDF4.Dataset): The climatology file.
    theta_s (float): The theta_s attribute from the climatology file.
    theta_b (float): The theta_b attribute from the climatology file.
    hc (float): The hc attribute from the climatology file.
    NZ (int): The size of the s_rho dimension from the climatology file.
    ncclm2 (netCDF4.Dataset): The second climatology file.
    h (np.ndarray): The h variable from the grid data.
    hmin (float): The hmin attribute from the grid data.
    hmax (float): The hmax attribute from the grid data.
    lon_rho (np.ndarray): The lon_rho variable from the grid data.
    lat_rho (np.ndarray): The lat_rho variable from the grid data.
    area (np.ndarray): The area calculated from the pm and pn variables from the grid data.
    angle (np.ndarray): The angle variable from the grid data.

    Methods:
    getAttrs(clim_file:str): Reads the attributes from the climatology file.
    setClmFiles(clm_file:str, clm2_file:str): Sets the climatology files.
    getTopo(): Reads the topography from the grid data.
    getLatLon(): Reads the latitudes and longitudes from the grid data.
    getArea(): Calculates the area from the pm and pn variables from the grid data.
    getAngle(): Reads the angle from the grid data.
    """

    # Read grid file
    def __init__(self, grid_file:str):
        # Set grd file
        self.grid_file = grid_file
        self.grid_data = netCDF4.Dataset(grid_file, mode='r')
        # Read mask
        self.mask_rho = self.grid_data.variables['mask_rho'][:]
        self.FillValue = getattr(self.grid_data.variables['mask_rho'],'_FillValue')
	# Read dimensions
        self.rho_y = self.mask_rho.shape[0]
        self.rho_x = self.mask_rho.shape[1]

    def getAttrs(self, clim_file:str):
        # Set clm file
        self.ncclm  = netCDF4.Dataset(clim_file, mode='r')
        # Read attributes
        try:
            self.theta_s = getattr(self.ncclm,'theta_s')
            self.theta_b = getattr(self.ncclm,'theta_b')
            self.hc      = getattr(self.ncclm,'hc')
        except AttributeError:
            self.theta_s = self.ncclm.variables['theta_s'][0]
            self.theta_b = self.ncclm.variables['theta_b'][0]
            self.hc      = self.ncclm.variables['hc'][0]
        # Vertical dimension
        self.NZ       = self.ncclm.dimensions['s_rho'].size

    def setClmFiles(self, clm_file: str, clm2_file: str):
	# Set clm file
        if not hasattr(self, 'ncclm'):
            self.ncclm  = netCDF4.Dataset(clm_file, mode='r')
        # Set clm2 file
        self.ncclm2 = netCDF4.Dataset(clm2_file, mode='r')

    def getTopo(self):
        # Read topography
        self.h     = self.grid_data.variables['h'][:]
        self.hmin  = getattr(self.grid_data,'hmin')
        self.hmax  = getattr(self.grid_data,'hmax')

    def getLatLon(self):
        # Read Lat/Lon
        self.lon_rho  = self.grid_data.variables['lon_rho'][:]
        self.lat_rho  = self.grid_data.variables['lat_rho'][:]

    def getArea(self):
        # Read pm/pn
        self.area  = 1/(self.grid_data.variables['pm'][:]*self.grid_data.variables['pn'][:])

    def getAngle(self):
        # Read angle
        self.angle  = self.grid_data.variables['angle'][:]


def compute_zlev(fpin:str, fpin_grd:str, NZ:int, type: str, zeta:float=None,
                 stype:int=3) -> npt.NDArray[float]:
    """
    Computes the z levels of rho points for zero SSH.

    Parameters:
    fpin (str): File descriptor pointing to a NetCDF file containing theta_b, theta_s and
    Tcline or hc.
    fpin_grd (str): File descriptor pointing to a NetCDF file containing h.
    NZ (int): Number of vertical (rho) levels.
    type (str): Specifies the type of points. 'r' for rho points, 'w' for w points.
    zeta (float, optional): Sea surface height. Defaults to None.
    stype (int, optional): Specifies type of sigma levels used. Defaults to 3.
        1: Similar to Song, Haidvogel 1994.
        2: Shchepetkin 2006.
        3: Shchepetkin 2010 (or so).

    Returns:
    npt.NDArray[float]: The z levels of rho points.

    Note:
    The function computes the z levels of rho points for zero SSH. It first reads the necessary
    variables from the input NetCDF files, then computes the z levels based on the specified sigma
    levels. The function uses numpy for computations and netCDF4 for reading the input files.
    """

    assert type in ['r', 'w'], "Wrong type chosen, it should be 'r' or 'w'."
    assert (stype>0)*(stype<4), "Wrong sigma level specifier."

    h = fpin_grd.variables['h'][:,:]
    try:
        theta_b = fpin.theta_b
        theta_s = fpin.theta_s
    except AttributeError:
        # theta_b/s may be variables:
        theta_b = fpin.variables['theta_b'][0]
        theta_s = fpin.variables['theta_s'][0]

    if stype == 1:
        hmin = min(min(h))
        try:
            Tcline = fpin.Tcline
            hc = min(hmin,Tcline)
        except AttributeError:
            hc = fpin.hc
            hc = min(hmin,hc)
    elif stype == 2 or stype == 3:
        try:
            hc = fpin.hc
        except AttributeError:
            # hc may be a variable:
            hc = fpin.variables['hc'][0]
    else:
        msg = '%s: Unknown type of sigma levels'.format(stype)
        sys.exit(msg)
    ds = 1./NZ  # float, to prevent integer division in sc
    if type == 'w':
        lev = np.arange(NZ+1)
        sc = (lev - NZ) * ds
        nr_zlev = NZ+1 # number of vertical levels
    else:
        lev = np.arange(1,NZ+1)
        sc = -1 + (lev-0.5)*ds
        nr_zlev = NZ # number of vertical levels
    Ptheta = np.sinh(theta_s*sc)/np.sinh(theta_s)
    Rtheta = np.tanh(theta_s*(sc+.5))/(2*np.tanh(.5*theta_s))-.5
    if stype <= 2:
        Cs = (1-theta_b)*Ptheta+theta_b*Rtheta
    elif stype == 3:
        if theta_s > 0:
            csrf=(1.-np.cosh(theta_s*sc))/(np.cosh(theta_s)-1.)
        else:
            csrf=-sc**2
        if theta_b > 0:
            Cs=(np.exp(theta_b*csrf)-1.)/(1.-np.exp(-theta_b))
        else:
            Cs=csrf
    z0 = np.zeros((nr_zlev,h.shape[0],h.shape[1]),np.float)
    if stype == 1:
        cff = (sc-Cs)*hc
        cff1 = Cs
        hinv = 1.0 / h
        for k in range(nr_zlev):
            z0[k,:,:] = cff[k]+cff1[k]*h
            if not (zeta is None):
                z0[k,:,:] = z0[k,:,:]+zeta*(1.+z0[k,:,:]*hinv)
    elif stype == 2 or stype == 3:
        hinv = 1.0/(h+hc)
        cff = hc*sc
        cff1 = Cs
        for k in range(nr_zlev):
            tmp1 = cff[k]+cff1[k]*h
            tmp2 = np.multiply(tmp1,hinv)
            if zeta is None:
                z0[k,:,:] = np.multiply(h,tmp2)
            else:
                z0[k,:,:] = zeta + np.multiply((zeta+h),tmp2)

    return z0

def compute_dz(fpin:str, fpin_grd:str, NZ:int, zeta:float=None,stype:int=3) -> npt.NDArray[float]:

    """
    Computes the dz of sigma level rho points for zero SSH.

    Parameters:
    fpin (str): File descriptor pointing to a NetCDF file containing theta_b, theta_s and
    Tcline or hc.
    fpin_grd (str): File descriptor pointing to a NetCDF file containing h.
    NZ (int): Number of vertical (rho) levels.
    zeta (float, optional): Sea surface height. Defaults to None.
    stype (int, optional): Specifies type of sigma levels used. Defaults to 3.
        1: Similar to Song, Haidvogel 1994.
        2: Shchepetkin 2006.
        3: Shchepetkin 2010 (or so).

    Returns:
    npt.NDArray[float]: The dz between w sigma levels (i.e., dz of sigma layer).

    Note:
    The function first computes the depth of w sigma levels, then computes the dz between these
    levels. The function uses the compute_zlev function to compute the depth of w sigma levels.
    """

    # Compute depth of w sigma levels
    depth_w = -compute_zlev(fpin,fpin_grd,NZ,type='w',zeta=zeta,stype=stype)

    # Compute dz between w sigma levels (= dz of sigma layer)
    dz_sigma = depth_w[:-1]-depth_w[1:]

    return dz_sigma


def writeNetCDF(romsGrd:str,dData:Dict, outfile:str, dDims:Dict=None, dAttrs:Dict=None,
                LatLon:bool=False, author:str='ursho') -> None:
    """
    Writes given data with ROMS grid dimensions to a NetCDF file.

    Parameters:
    romsGrd (str): A grid class with necessary attributes.
    dData (Dict): A dictionary containing data for different variables
    (e.g. dData['temp'],dData['salt']).
    outfile (str): The specified path and filename for output as a string
    (e.g. '/net/kryo/work/martinfr/Data/pacsg/slavg/new_file_name.nc').
    dDims (Dict, optional): A dictionary containing the dimensions and their lengths of the
    variables. Defaults to None.
    dAttrs (Dict, optional): A dictionary containing the attributes of the variables.
    Defaults to None.
    LatLon (bool, optional): A flag indicating whether to write longitude and latitude data.
    Defaults to False.
    author (str, optional): The author of the data. Defaults to 'ursho'.

    Returns:
    None

    Note:
    The function writes the data to a NetCDF file at the given directory and filename. It creates
    dimensions based on the provided dimensions dictionary, writes variable data, sets attributes,
    and optionally writes longitude and latitude data. The function uses the netCDF4 library to
    create and write to the NetCDF file.
    """

    # Create dimensions
    if dDims==None:
        print('Please provide dimensions of data...')
        return
    else:
        # Open new NetCDF file to write to
        ncfile = netCDF4.Dataset(outfile, 'w', format='NETCDF4')

        # Take a variable as reference for dimensions
        for key in dDims.keys():
            for dim,dimlen in zip(dDims[key]['dims'],dDims[key]['dimlens']):
                if not dim in ncfile.dimensions:
                    ncfile.createDimension(dim,dimlen)

    # Set author attributes
    setattr(ncfile, 'author', author)
    setattr(ncfile, 'created', str(datetime.date.today()))

    # Writing variable data
    for var in dData.keys():
        sz = dData[var].shape
        try:
            data = ncfile.createVariable(var,'f4',dDims[var]['dims'],fill_value=dData[var].fill_value)
        except AttributeError:
            data = ncfile.createVariable(var,'f4',dDims[var]['dims'])
        # If 4D variable do sequential writing
        if len(sz)>3:
            for t in range(sz[0]):
                data[t] = dData[var][t]
        else:
            data[:] = dData[var][:]
        # Set atrributes
        if not dAttrs==None:
            for attr in dAttrs[var].keys():
                if attr=='_FillValue':
                    pass
                else:
                    setattr(data, attr, dAttrs[var][attr])

    # Write lon-lat data
    if LatLon==True:
        for var in ['lon_rho','lat_rho']:
            varDims = ('eta_rho','xi_rho')
            data = ncfile.createVariable(var,'f4',varDims,fill_value=romsGrd.grid_data.variables[var]._FillValue)
            data[:] = romsGrd.grid_data.variables[var][:]
            for attr in romsGrd.grid_data.variables[var].ncattrs():
                if attr=='_FillValue':
                    pass
                else:
                    setattr(data, attr, getattr(romsGrd.grid_data.variables[var],attr))
    # Close the file and end the program.
    ncfile.close()
