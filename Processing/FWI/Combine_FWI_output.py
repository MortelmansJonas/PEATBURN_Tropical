#!/usr/bin/env python

'''

This file takes all the daily FWI output of the GFWED code and combines it into one netCDF file.

'''

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
start_date = '20020101'
end_date = '20201231'

path = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/FWI/Reference/fwiCalcs.MERRA2/Default/MERRA2/'
filename_root = 'FWI.MERRA2.Daily.Default.'

days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
start_month = str(pd.to_datetime(start_date)).split('-')[1]
end_month = str(pd.to_datetime(end_date)).split('-')[1]
start_year = str(pd.to_datetime(start_date)).split('-')[0]
end_year = str(pd.to_datetime(start_date)).split('-')[0]

# ---------------------------------------------------------------------------------------------
# CREATE NC FILE
# ---------------------------------------------------------------------------------------------
# First load 1 file for dimensions
ds_test = Dataset('/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/FWI/Reference/fwiCalcs.MERRA2/Default/MERRA2/2010'
                  '/FWI.MERRA2.Daily.Default.20100101.nc', mode='r')
lats = ds_test['lat'][:]
lons = ds_test['lon'][:]

lon, lat = np.meshgrid(lons, lats)

total_days = np.linspace(1, days, days)
ds = Dataset('/scratch/leuven/336/vsc33651/PEATBURN/Tropical/FWI_MERRA2_combined.nc', mode='w', format='NETCDF4')
timeunit = 'hours since 2000-01-01 00:00'
ds.createDimension('time', None)
ds.createDimension('lat', 267)
ds.createDimension('lon', 576)
ds.createVariable('time', total_days.dtype, dimensions='time', zlib=True)
ds.variables['time'][:] = total_days + (pd.to_datetime(start_date) - pd.to_datetime('2000-01-01')).days * 24 - 1
ds.createVariable('lat', lat.dtype, dimensions=('lat', 'lon'), zlib=True)
ds.variables['lat'][:] = lat
ds.createVariable('lon', lon.dtype, dimensions=('lat', 'lon'), zlib=True)
ds.variables['lon'][:] = lon
ds.createVariable('MERRA2_BUI', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_DC', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_DMC', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_DSR', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_FFMC', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_FWI', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds.createVariable('MERRA2_ISI', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)

ds.variables['time'].setncatts({'long_name': 'time', 'units': timeunit})
ds.variables['lat'].setncatts({'long_name': 'latitude', 'units': 'degrees_north'})
ds.variables['lon'].setncatts({'long_name': 'longitude', 'units': 'degrees_east'})
ds.variables['MERRA2_BUI'].setncatts({'long_name': 'Build-up Index', 'units': '-'})
ds.variables['MERRA2_DC'].setncatts({'long_name': 'Drought Code', 'units': '-'})
ds.variables['MERRA2_DMC'].setncatts({'long_name': 'Duff Moisture Code', 'units': '-'})
ds.variables['MERRA2_DSR'].setncatts({'long_name': 'Daily Severity Rating', 'units': '-'})
ds.variables['MERRA2_FFMC'].setncatts({'long_name': 'Fine Fuel Moisture Code', 'units': '-'})
ds.variables['MERRA2_FWI'].setncatts({'long_name': 'Fire Weather Index', 'units': '-'})
ds.variables['MERRA2_ISI'].setncatts({'long_name': 'Initial Spread Index', 'units': '-'})

for d in range(0, days + 1):
    date = str(
        pd.to_datetime((pd.to_datetime(start_date, format='%Y%m%d') + pd.to_timedelta(d, unit='D'))).strftime('%Y%m%d'))
    year = date[0:4]
    file = os.path.join(path, year + '/' + filename_root + date + '.nc')
    print('processing ' + date)
    ds_in = Dataset(file, 'r')

    ds.variables['MERRA2_BUI'][d, :, :] = ds_in.variables['MERRA2_BUI'][:]
    ds.variables['MERRA2_DC'][d, :, :] = ds_in.variables['MERRA2_DC'][:]
    ds.variables['MERRA2_DMC'][d, :, :] = ds_in.variables['MERRA2_DMC'][:]
    ds.variables['MERRA2_DSR'][d, :, :] = ds_in.variables['MERRA2_DSR'][:]
    ds.variables['MERRA2_FFMC'][d, :, :] = ds_in.variables['MERRA2_FFMC'][:]
    ds.variables['MERRA2_FWI'][d, :, :] = ds_in.variables['MERRA2_FWI'][:]
    ds.variables['MERRA2_ISI'][d, :, :] = ds_in.variables['MERRA2_ISI'][:]
ds.close()
