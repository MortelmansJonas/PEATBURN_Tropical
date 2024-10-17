#!/usr/bin/env python
"""
This script is used to split the Miettinen data to drained and undrained peatlands
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# LOAD MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
import gdal
import pandas as pd
import os
from python_functions import create_file_from_source
from copy import deepcopy as dcopy

# ---------------------------------------------------------------------------------------------
# SPECIFY TIER AND CORRESPONDING PATHS
# ---------------------------------------------------------------------------------------------
Tier = 'Genius'

if Tier == 'Hortense':
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Reference'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/CDF_matched'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures'
    path_peatlands = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATCLSM'
    path_PEATMAP = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATMAP' \
                   '/Miettinen2016-PeatLCC900715'

elif Tier == 'Genius':
    path_ref = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/Reference'
    path_out = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/output/CDF_matched'
    path_fires = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/Fire_data'
    path_figs = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/output/Figures'
    path_peatlands = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/PEATCLSM'
    path_PEATMAP = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/PEATMAP' \
                   '/Miettinen2016-PeatLCC900715'

else:
    print('Error: Tier can only be Breniac, Hortense, or Genius.')

# endregion
'''
'Drained_fraction': fraction of the fire that was on drained areas (All LUT classes except for water, seasonal_water, 
pristine_PSF, and mangrove are considered as drained)
'Undrained_fraction': fraction of the fire that was on undrained areas (sum of water, seasonal_water, pristine_PSF, 
and mangrove)
'''

ds_Miettinen2007 = gdal.Open(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84.tif'))
Miettinen2007 = ds_Miettinen2007.GetRasterBand(1).ReadAsArray()
Drained_2007 = dcopy(Miettinen2007)
Drained_2007[Drained_2007 <= 3] = np.nan
Drained_2007[(Drained_2007 > 3) & (Drained_2007 < 11)] = 1
Drained_2007[Drained_2007 == 11] = np.nan
Drained_2007[Drained_2007 > 11] = 1

Natural_2007 = dcopy(Miettinen2007)
Natural_2007[Natural_2007 == 0] = np.nan
Natural_2007[Natural_2007 <= 3] = 1
Natural_2007[(Natural_2007 > 3) & (Natural_2007 < 11)] = np.nan
Natural_2007[Natural_2007 == 11] = 1
Natural_2007[Natural_2007 > 11] = np.nan

lats, lons = Miettinen2007.shape

ds_Miettinen2015 = gdal.Open(os.path.join(path_PEATMAP, 'Miettinen_2015_WGS84.tif'))
Miettinen2015 = ds_Miettinen2015.GetRasterBand(1).ReadAsArray()
Drained_2015 = dcopy(Miettinen2015)
Drained_2015[Drained_2015 <= 3] = np.nan
Drained_2015[(Drained_2015 > 3) & (Drained_2015 < 11)] = 1
Drained_2015[Drained_2015 == 11] = np.nan

Natural_2015 = dcopy(Miettinen2015)
Natural_2015[Natural_2015 == 0] = np.nan
Natural_2015[Natural_2015 <= 3] = 1
Natural_2015[(Natural_2015 > 3) & (Natural_2015 < 11)] = np.nan
Natural_2015[Natural_2015 == 11] = 1

# Write to output:
driver = gdal.GetDriverByName("GTiff")
outdata_2007_Dra = driver.Create(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84_Dra.tif'), lons, lats, 1,
                                 gdal.GDT_UInt16)
outdata_2007_Dra.SetGeoTransform(ds_Miettinen2007.GetGeoTransform())  # Sets same geotransform as input
outdata_2007_Dra.SetProjection(ds_Miettinen2007.GetProjection())  # Sets same projection as input
outdata_2007_Dra.GetRasterBand(1).WriteArray(Drained_2007)
outdata_2007_Dra.FlushCache()  # saves to disk
outdata_2007_Dra = None

outdata_2007_Nat = driver.Create(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84_Nat.tif'), lons, lats, 1,
                                 gdal.GDT_UInt16)
outdata_2007_Nat.SetGeoTransform(ds_Miettinen2007.GetGeoTransform())  # Sets same geotransform as input
outdata_2007_Nat.SetProjection(ds_Miettinen2007.GetProjection())  # Sets same projection as input
outdata_2007_Nat.GetRasterBand(1).WriteArray(Natural_2007)
outdata_2007_Nat.FlushCache()  # saves to disk
outdata_2007_Nat = None

outdata_2015_Dra = driver.Create(os.path.join(path_PEATMAP, 'Miettinen_2015_WGS84_Dra.tif'), lons, lats, 1,
                                 gdal.GDT_UInt16)
outdata_2015_Dra.SetGeoTransform(ds_Miettinen2015.GetGeoTransform())  # Sets same geotransform as input
outdata_2015_Dra.SetProjection(ds_Miettinen2015.GetProjection())  # Sets same projection as input
outdata_2015_Dra.GetRasterBand(1).WriteArray(Drained_2015)
outdata_2015_Dra.FlushCache()  # saves to disk
outdata_2015_Dra = None

outdata_2015_Nat = driver.Create(os.path.join(path_PEATMAP, 'Miettinen_2015_WGS84_Nat.tif'), lons, lats, 1,
                                 gdal.GDT_UInt16)
outdata_2015_Nat.SetGeoTransform(ds_Miettinen2015.GetGeoTransform())  # Sets same geotransform as input
outdata_2015_Nat.SetProjection(ds_Miettinen2015.GetProjection())  # Sets same projection as input
outdata_2015_Nat.GetRasterBand(1).WriteArray(Natural_2015)
outdata_2015_Nat.FlushCache()  # saves to disk
outdata_2015_Nat = None

