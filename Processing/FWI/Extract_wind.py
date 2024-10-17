#!/usr/bin/env python
"""
This script is used to extract wind speed from the MERRA2 data of the GFWED calculations.

"""
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset

# ---------------------------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------------------------
date_from = '2002-01-01'
date_to = '2019-01-01'

root_filename = 'Wx.MERRA2.Daily.Default.'
root_precip = 'Prec.MERRA2.Daily.Default.'
path = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Northern/FWI/Reference/wxInput/MERRA2'
output_filename = os.path.join(
    '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/Reference',
    'MERRA2_combined.nc')

# determine length of time vector
startdate = pd.to_datetime(date_from)
enddate = pd.to_datetime(date_to)
ndays = (pd.to_datetime(date_to) - pd.to_datetime(date_from)).days
days = np.linspace(1, ndays, ndays)
day = str(startdate).split(' ')
day = str(day[0]).split('-')
year = day[0]

filename = os.path.join(path, year + "/" + root_filename + day[0] + day[1] + day[2] + '.nc')
ds_st = Dataset(filename, 'r')
lat = ds_st['lat'][:]
lon = ds_st['lon'][:]

ds = Dataset(output_filename, mode='w', format='NETCDF4')
timeunit = 'days since 2000-01-01 00:00'
ds.createDimension('time', None)
ds.createDimension('lat', len(lat))
ds.createDimension('lon', len(lon))
ds.createVariable('time', days.dtype, dimensions='time', zlib=True)
ds.variables['time'][:] = days + (pd.to_datetime(date_from) - pd.to_datetime('2000-01-01')).days - 1
ds.createVariable('lat', lat.dtype, dimensions='lat', zlib=True)
ds.variables['lat'][:] = lat
ds.createVariable('lon', lon.dtype, dimensions='lon', zlib=True)
ds.variables['lon'][:] = lon
ds.createVariable('MERRA2_wdSpd', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_T', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_RH', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_P', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)

ds.variables['time'].setncatts({'long_name': 'time', 'units': timeunit})
ds.variables['lat'].setncatts({'long_name': 'latitude', 'units': 'degrees_north'})
ds.variables['lon'].setncatts({'long_name': 'longitude', 'units': 'degrees_east'})
ds.variables['MERRA2_wdSpd'].setncatts({'long_name': 'MERRA2 noon 10m wind speed', 'units': 'km/h'})
ds.variables['MERRA2_T'].setncatts({'long_name': 'MERRA2 noon 2m temperature', 'units': 'degrees Celsius'})
ds.variables['MERRA2_RH'].setncatts({'long_name': 'MERRA2 noon 2m relative humidity', 'units': '-'})
ds.variables['MERRA2_P'].setncatts({'long_name': 'MERRA2 24h accumulated P', 'units': 'mm'})

d = pd.to_timedelta(startdate - pd.to_datetime('2010-01-01')).days
for d in range(d, d + ndays + 1):
    days_passed = pd.to_timedelta(d, unit='d')
    a = pd.to_datetime('2010-01-01') + days_passed
    i = str(a)
    day = i.split(' ')
    day = day[0].split('-')
    year = day[0]
    filename = os.path.join(path, year + "/" + root_filename + day[0] + day[1] + day[2] + '.nc')
    filename_prec = os.path.join(path, year + "/" + root_precip + day[0] + day[1] + day[2] + '.nc')
    print('processing ' + filename)
    ds_in = Dataset(filename, mode='r')
    ds_in2 = Dataset(filename_prec, 'r')

    ds.variables['MERRA2_wdSpd'][d, :, :] = ds_in.variables['MERRA2_wdSpd'][:]
    ds.variables['MERRA2_T'][d, :, :] = ds_in.variables['MERRA2_t'][:]
    ds.variables['MERRA2_RH'][d, :, :] = ds_in.variables['MERRA2_rh'][:]
    ds.variables['MERRA2_P'][d, :, :] = ds_in2.variables['MERRA2_prec'][:]


ds.close()
