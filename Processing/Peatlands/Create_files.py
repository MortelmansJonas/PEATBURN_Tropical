#!/usr/bin/env python
"""
This script creates the files for the weighted average FWI. It is a separate file because it took some time and this
seemed to work better.
"""
# ---------------------------------------------------------------------------------------------
# LOAD MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import os
from python_functions import create_file_from_source
import gdal
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------
# SPECIFY TIER AND CORRESPONDING PATHS
# ---------------------------------------------------------------------------------------------
Tier = 'Hortense'

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

create_file_from_source(os.path.join(path_ref, 'FWI_masked_Nat_tchunk.nc'),
                        os.path.join(path_ref, 'FWI_Ref_weighted.nc'))

# Set all variables to NaN
ds_M2 = Dataset(os.path.join(path_ref, 'FWI_Ref_weighted.nc'), 'a')
ds_M2['MERRA2_DC'][:] = np.nan
ds_M2['MERRA2_DMC'][:] = np.nan
ds_M2['MERRA2_FFMC'][:] = np.nan
ds_M2['MERRA2_ISI'][:] = np.nan
ds_M2['MERRA2_BUI'][:] = np.nan
ds_M2['MERRA2_FWI'][:] = np.nan

# Add new variables to the netCDF file containing the weights of natural and drained peatlands
ds_M2.createVariable('weight_natural', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds_M2.createVariable('weight_drained', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds_M2['weight_natural'][:] = np.nan
ds_M2['weight_drained'][:] = np.nan
ds_M2.variables['weight_natural'].setncatts({'long_name': 'weight of natural peatlands', 'units': '-'})
ds_M2.variables['weight_drained'].setncatts({'long_name': 'weight of drained peatlands', 'units': '-'})

create_file_from_source(os.path.join(path_out, 'FWI_zbar_FFMC_TN_pixel.nc'),
                        os.path.join(path_out, 'FWI_EXP3b_weighted.nc'))

# Set all variables to NaN
ds_EXP3b = Dataset(os.path.join(path_out, 'FWI_EXP3b_weighted.nc'), 'a')
ds_EXP3b['MERRA2_DC'][:] = np.nan
ds_EXP3b['MERRA2_DMC'][:] = np.nan
ds_EXP3b['MERRA2_FFMC'][:] = np.nan
ds_EXP3b['MERRA2_ISI'][:] = np.nan
ds_EXP3b['MERRA2_BUI'][:] = np.nan
ds_EXP3b['MERRA2_FWI'][:] = np.nan

# Add new variables to the netCDF file containing the weights of natural and drained peatlands
ds_EXP3b.createVariable('weight_natural', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds_EXP3b.createVariable('weight_drained', 'f4', dimensions=('time', 'lat', 'lon'), zlib=True)
ds_EXP3b['weight_natural'][:] = np.nan
ds_EXP3b['weight_drained'][:] = np.nan
ds_EXP3b.variables['weight_natural'].setncatts({'long_name': 'weight of natural peatlands', 'units': '-'})
ds_EXP3b.variables['weight_drained'].setncatts({'long_name': 'weight of drained peatlands', 'units': '-'})

ds_EXP3b.close()
ds_M2.close()