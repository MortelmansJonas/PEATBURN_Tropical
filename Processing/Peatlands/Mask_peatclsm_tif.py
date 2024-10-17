#!/usr/bin/env python
"""
This script is used to mask the PEATCLSM_TN and PEATCLSM_TD output to only peatlands, based on the Miettinen maps.
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# LOAD MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import os
from python_functions import create_file_from_source
import gdal
import pandas as pd

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

'''--------------------------------------------------Load datasets--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# PEATCLSM OUTPUT
# ---------------------------------------------------------------------------------------------
# the PEATCLSM output of the Tropical Natural run
ds_natural = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TN.nc'), 'r')
lons_PEATCLSM = ds_natural['lon'][:].data
lats_PEATCLSM = ds_natural['lat'][:].data
ds_drained = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TD.nc'), 'r')

times = pd.date_range("2000-01-01", "2020-10-30", freq="D")
inds_2007 = np.where(times.year <= 2010)[0]
inds_2015 = np.where(times.year > 2010)[0]

# ---------------------------------------------------------------------------------------------
# MIETINNEN PEATLAND MAPS
# ---------------------------------------------------------------------------------------------
ds_drained_2007 = gdal.Open(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84_Dra_compressed.tif'))
Drained_2007 = ds_drained_2007.GetRasterBand(1).ReadAsArray()
ds_drained_2015 = gdal.Open(os.path.join(path_PEATMAP, 'Miettinen_2015_WGS84_Dra_compressed.tif'))
Drained_2015 = ds_drained_2015.GetRasterBand(1).ReadAsArray()
ds_natural_2007 = gdal.Open(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84_Nat_compressed.tif'))
Natural_2007 = ds_natural_2007.GetRasterBand(1).ReadAsArray()
ds_natural_2015 = gdal.Open(os.path.join(path_PEATMAP, 'Miettinen_2015_WGS84_Nat_compressed.tif'))
Natural_2015 = ds_natural_2015.GetRasterBand(1).ReadAsArray()

width = ds_drained_2007.RasterXSize
height = ds_drained_2007.RasterYSize
gt = ds_drained_2007.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5]
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3]

lons = np.linspace(minx, maxx, width)
lats = np.linspace(maxy, miny, height)

print(np.nanmean(Drained_2007))
'''-----------------------------------------------Create new .nc files-----------------------------------------------'''
# region
"""
Create new netCDF files for the masked PEATCLSM data
"""

if not os.path.exists(os.path.join(path_peatlands, 'PEATCLSM_TN_masked_Nat.nc')):
    create_file_from_source(os.path.join(path_peatlands, 'PEATCLSM_TN.nc'),
                            os.path.join(path_peatlands, 'PEATCLSM_TN_masked_Nat.nc'))

    # Set all variables to NaN
    ds_TN_masked = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TN_masked_Nat.nc'), 'a')
    ds_TN_masked['zbar'][:] = np.nan
    ds_TN_masked['sfmc'][:] = np.nan
    ds_TN_masked['rzmc'][:] = np.nan
else:
    ds_TN_masked = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TN_masked_Nat.nc'), 'a')

if not os.path.exists(os.path.join(path_peatlands, 'PEATCLSM_TD_masked_Dra.nc')):
    create_file_from_source(os.path.join(path_peatlands, 'PEATCLSM_TD.nc'),
                            os.path.join(path_peatlands, 'PEATCLSM_TD_masked_Dra.nc'))

    # Set all variables to NaN
    ds_TD_masked = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TD_masked_Dra.nc'), 'a')
    ds_TD_masked['zbar'][:] = np.nan
    ds_TD_masked['sfmc'][:] = np.nan
    ds_TD_masked['rzmc'][:] = np.nan
else:
    ds_TD_masked = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TD_masked_Dra.nc'), 'a')

# endregion

'''------------------------------------------------Mask PEATCLSM data------------------------------------------------'''
# region
"""
Mask the PEATCLSM data based on the Miettinen map. If there is peat (any value) in the Miettinen map, 
the corresponding nearest PEATCLSM pixel should be retained in the masked data. If not, the data should be removed
"""

for lat in range(len(lats_PEATCLSM) - 1):
    print("lats: " + str(lat) + " or " + str(np.round(lat / len(lats_PEATCLSM) * 100, 2)) + "% complete")
    lat_inds = np.asarray(np.where((lats <= lats_PEATCLSM[lat]) & (lats >= lats_PEATCLSM[lat + 1]))[0])

    if lat_inds.size == 0:
        print('empty lat, skipping')
    else:

        for lon in range(len(lons_PEATCLSM) - 1):
            lon_inds = np.asarray(np.where((lons >= lons_PEATCLSM[lon]) & (lons <= lons_PEATCLSM[lon + 1]))[0])

            if lon_inds.size == 0:
                print('empty lon, skipping')
            else:
                subset_Drained2007 = Drained_2007[lat_inds, :]
                subset2_Drained2007 = subset_Drained2007[:, lon_inds]
                if np.nanmean(subset2_Drained2007) > 0:
                    ds_TD_masked['zbar'][inds_2007, lat, lon] = ds_drained['zbar'][inds_2007, lat, lon]
                    ds_TD_masked['sfmc'][inds_2007, lat, lon] = ds_drained['sfmc'][inds_2007, lat, lon]
                    ds_TD_masked['rzmc'][inds_2007, lat, lon] = ds_drained['rzmc'][inds_2007, lat, lon]

                subset_Natural2007 = Natural_2007[lat_inds, :]
                subset2_Natural2007 = subset_Natural2007[:, lon_inds]
                if np.nanmean(subset2_Natural2007) > 0:
                    ds_TN_masked['zbar'][inds_2007, lat, lon] = ds_natural['zbar'][inds_2007, lat, lon]
                    ds_TN_masked['sfmc'][inds_2007, lat, lon] = ds_natural['sfmc'][inds_2007, lat, lon]
                    ds_TN_masked['rzmc'][inds_2007, lat, lon] = ds_natural['rzmc'][inds_2007, lat, lon]

                subset_Drained2015 = Drained_2015[lat_inds, :]
                subset2_Drained2015 = subset_Drained2015[:, lon_inds]
                if np.nanmean(subset2_Drained2015) > 0:
                    ds_TD_masked['zbar'][inds_2015, lat, lon] = ds_drained['zbar'][inds_2015, lat, lon]
                    ds_TD_masked['sfmc'][inds_2015, lat, lon] = ds_drained['sfmc'][inds_2015, lat, lon]
                    ds_TD_masked['rzmc'][inds_2015, lat, lon] = ds_drained['rzmc'][inds_2015, lat, lon]

                subset_Natural2015 = Natural_2015[lat_inds, :]
                subset2_Natural2015 = subset_Natural2015[:, lon_inds]
                if np.nanmean(subset2_Natural2015) > 0:
                    ds_TN_masked['zbar'][inds_2015, lat, lon] = ds_natural['zbar'][inds_2015, lat, lon]
                    ds_TN_masked['sfmc'][inds_2015, lat, lon] = ds_natural['sfmc'][inds_2015, lat, lon]
                    ds_TN_masked['rzmc'][inds_2015, lat, lon] = ds_natural['rzmc'][inds_2015, lat, lon]

ds_TD_masked.close()
ds_TN_masked.close()
# endregion
