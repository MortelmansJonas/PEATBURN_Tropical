#!/usr/bin/env python
"""
This script is used to mask the FWI reference output output to only peatlands, based on the Miettinen maps.
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
ds = Dataset(os.path.join(path_ref, 'FWI_MERRA2_Ref_20022020_regridded.nc'), 'r')
lons_PEATCLSM = ds['lon'][:].data
lats_PEATCLSM = ds['lat'][:].data

times = pd.date_range("2000-01-01", "2020-10-30", freq="D")
inds_2007 = np.where(times.year <= 2010)[0]
inds_2015 = np.where(times.year > 2010)[0]

# ---------------------------------------------------------------------------------------------
# PEATCLSM data
# ---------------------------------------------------------------------------------------------
ds_nat = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TN_masked_Nat_20022018.nc'), 'r')
ds_dra = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TD_masked_Dra_20022018.nc'), 'r')

'''-----------------------------------------------Create new .nc files-----------------------------------------------'''
# region
"""
Create new netCDF files for the masked PEATCLSM data
"""

if not os.path.exists(os.path.join(path_ref, 'FWI_masked_Nat.nc')):
    create_file_from_source(os.path.join(path_ref, 'FWI_MERRA2_Ref_20022020_regridded.nc'),
                            os.path.join(path_ref, 'FWI_masked_Nat.nc'))

    # Set all variables to NaN
    ds_TN_masked = Dataset(os.path.join(path_ref, 'FWI_masked_Nat.nc'), 'a')
    ds_TN_masked['MERRA2_DC'][:] = np.nan
    ds_TN_masked['MERRA2_DMC'][:] = np.nan
    ds_TN_masked['MERRA2_FFMC'][:] = np.nan
    ds_TN_masked['MERRA2_BUI'][:] = np.nan
    ds_TN_masked['MERRA2_ISI'][:] = np.nan
    ds_TN_masked['MERRA2_FWI'][:] = np.nan
else:
    ds_TN_masked = Dataset(os.path.join(path_ref, 'FWI_masked_Nat.nc'), 'a')

if not os.path.exists(os.path.join(path_ref, 'FWI_masked_Dra.nc')):
    create_file_from_source(os.path.join(path_ref, 'FWI_MERRA2_Ref_20022020_regridded.nc'),
                            os.path.join(path_ref, 'FWI_masked_Dra.nc'))

    # Set all variables to NaN
    ds_TD_masked = Dataset(os.path.join(path_ref, 'FWI_masked_Dra.nc'), 'a')
    ds_TD_masked['MERRA2_DC'][:] = np.nan
    ds_TD_masked['MERRA2_DMC'][:] = np.nan
    ds_TD_masked['MERRA2_FFMC'][:] = np.nan
    ds_TD_masked['MERRA2_BUI'][:] = np.nan
    ds_TD_masked['MERRA2_ISI'][:] = np.nan
    ds_TD_masked['MERRA2_FWI'][:] = np.nan
else:
    ds_TD_masked = Dataset(os.path.join(path_ref, 'FWI_masked_Dra.nc'), 'a')

# endregion

'''------------------------------------------------Mask PEATCLSM data------------------------------------------------'''
# region

mask_nat = np.where(~np.isnan(ds_nat['zbar'][:].data), 1, np.nan)
mask_dra = np.where(~np.isnan(ds_dra['zbar'][:].data), 1, np.nan)

ds_TN_masked['MERRA2_DC'][0:6209, :, :] = np.multiply(ds['MERRA2_DC'][0:6209, :, :], mask_nat)
ds_TN_masked['MERRA2_DMC'][0:6209, :, :] = np.multiply(ds['MERRA2_DMC'][0:6209, :, :], mask_nat)
ds_TN_masked['MERRA2_FFMC'][0:6209, :, :] = np.multiply(ds['MERRA2_FFMC'][0:6209, :, :], mask_nat)
ds_TN_masked['MERRA2_BUI'][0:6209, :, :] = np.multiply(ds['MERRA2_BUI'][0:6209, :, :], mask_nat)
ds_TN_masked['MERRA2_ISI'][0:6209, :, :] = np.multiply(ds['MERRA2_ISI'][0:6209, :, :], mask_nat)
ds_TN_masked['MERRA2_FWI'][0:6209, :, :] = np.multiply(ds['MERRA2_FWI'][0:6209, :, :], mask_nat)

ds_TD_masked['MERRA2_DC'][0:6209, :, :] = np.multiply(ds['MERRA2_DC'][0:6209, :, :], mask_dra)
ds_TD_masked['MERRA2_DMC'][0:6209, :, :] = np.multiply(ds['MERRA2_DMC'][0:6209, :, :], mask_dra)
ds_TD_masked['MERRA2_FFMC'][0:6209, :, :] = np.multiply(ds['MERRA2_FFMC'][0:6209, :, :], mask_dra)
ds_TD_masked['MERRA2_BUI'][0:6209, :, :] = np.multiply(ds['MERRA2_BUI'][0:6209, :, :], mask_dra)
ds_TD_masked['MERRA2_ISI'][0:6209, :, :] = np.multiply(ds['MERRA2_ISI'][0:6209, :, :], mask_dra)
ds_TD_masked['MERRA2_FWI'][0:6209, :, :] = np.multiply(ds['MERRA2_FWI'][0:6209, :, :], mask_dra)

ds_TN_masked.close()
ds_TD_masked.close()
