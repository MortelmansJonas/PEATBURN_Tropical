#!/usr/bin/env python
"""
This script is used to calculate the weighted average of the reference FWI and EXP3b for drained and natural peatlands
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# LOAD MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import os

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
    path_ref = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_out = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_fires = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_figs = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_peatlands = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_PEATMAP = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'

else:
    print('Error: Tier can only be Breniac, Hortense, or Genius.')

'''--------------------------------------------------Load datasets--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# PEATCLSM OUTPUT
# ---------------------------------------------------------------------------------------------
ds = Dataset(os.path.join(path_ref, 'FWI_Ref_weighted.nc'), mode='r')
lons = ds['lon'][:]
lats = ds['lat'][:]
ds.close()

'''-------------------------------------------Calculate Weighted Average--------------------------------------------'''
"""
Mask the FWI data based on the Miettinen map. Determine the weighted average for natural and drained peatlands.
"""

# experiments = ["Ref", "EXP3b"]
experiments = ["EXP4"]
# FWI_variables = ['MERRA2_DC', 'MERRA2_DMC', 'MERRA2_FFMC', 'MERRA2_BUI', 'MERRA2_ISI', 'MERRA2_FWI']
# FWI_variables = ['MERRA2_FWI']
FWI_variables = ['zbar']

for experiment in experiments:
    print(experiment + '\n')
    if experiment == "Ref":
        ds = Dataset(os.path.join(path_ref, 'FWI_Ref_weighted.nc'), mode='a')
        ds["MERRA2_DC"][:] = np.nan
        ds["MERRA2_DMC"][:] = np.nan
        ds["MERRA2_FFMC"][:] = np.nan
        ds["MERRA2_ISI"][:] = np.nan
        ds["MERRA2_BUI"][:] = np.nan
        ds["MERRA2_FWI"][:] = np.nan
        ds_in_nat = Dataset(os.path.join(path_ref, 'FWI_masked_Nat.nc'), 'r')
        ds_in_dra = Dataset(os.path.join(path_ref, 'FWI_masked_Dra.nc'), 'r')

    elif experiment == "EXP1":
        ds = Dataset(os.path.join(path_out, 'FWI_EXP1_weighted.nc'), mode='a')
        ds["MERRA2_DC"][:] = np.nan
        ds["MERRA2_DMC"][:] = np.nan
        ds["MERRA2_FFMC"][:] = np.nan
        ds["MERRA2_ISI"][:] = np.nan
        ds["MERRA2_BUI"][:] = np.nan
        ds["MERRA2_FWI"][:] = np.nan
        ds_in_nat = Dataset(os.path.join(path_out, 'FWI_zbar_DC_TN_pixel.nc'), 'r')
        ds_in_dra = Dataset(os.path.join(path_out, 'FWI_zbar_DC_TD_pixel.nc'), 'r')

    elif experiment == "EXP2":
        ds = Dataset(os.path.join(path_out, 'FWI_EXP2_weighted.nc'), mode='a')
        ds["MERRA2_DC"][:] = np.nan
        ds["MERRA2_DMC"][:] = np.nan
        ds["MERRA2_FFMC"][:] = np.nan
        ds["MERRA2_ISI"][:] = np.nan
        ds["MERRA2_BUI"][:] = np.nan
        ds["MERRA2_FWI"][:] = np.nan
        ds_in_nat = Dataset(os.path.join(path_out, 'FWI_sfmc_DMC_TN_pixel.nc'), 'r')
        ds_in_dra = Dataset(os.path.join(path_out, 'FWI_sfmc_DMC_TD_pixel.nc'), 'r')

    elif experiment == "EXP2b":
        ds = Dataset(os.path.join(path_out, 'FWI_EXP2b_weighted.nc'), mode='a')
        ds["MERRA2_DC"][:] = np.nan
        ds["MERRA2_DMC"][:] = np.nan
        ds["MERRA2_FFMC"][:] = np.nan
        ds["MERRA2_ISI"][:] = np.nan
        ds["MERRA2_BUI"][:] = np.nan
        ds["MERRA2_FWI"][:] = np.nan
        ds_in_nat = Dataset(os.path.join(path_out, 'FWI_zbar_DMC_TN_pixel.nc'), 'r')
        ds_in_dra = Dataset(os.path.join(path_out, 'FWI_zbar_DMC_TD_pixel.nc'), 'r')

    elif experiment == "EXP3":
        ds = Dataset(os.path.join(path_out, 'FWI_EXP3_weighted.nc'), mode='a')
        ds["MERRA2_DC"][:] = np.nan
        ds["MERRA2_DMC"][:] = np.nan
        ds["MERRA2_FFMC"][:] = np.nan
        ds["MERRA2_ISI"][:] = np.nan
        ds["MERRA2_BUI"][:] = np.nan
        ds["MERRA2_FWI"][:] = np.nan
        ds_in_nat = Dataset(os.path.join(path_out, 'FWI_sfmc_FFMC_TN_pixel.nc'), 'r')
        ds_in_dra = Dataset(os.path.join(path_out, 'FWI_sfmc_FFMC_TD_pixel.nc'), 'r')

    elif experiment == "EXP3b":
        ds = Dataset(os.path.join(path_out, 'FWI_EXP3b_weighted.nc'), mode='a')
        ds["MERRA2_DC"][:] = np.nan
        ds["MERRA2_DMC"][:] = np.nan
        ds["MERRA2_FFMC"][:] = np.nan
        ds["MERRA2_ISI"][:] = np.nan
        ds["MERRA2_BUI"][:] = np.nan
        ds["MERRA2_FWI"][:] = np.nan
        ds_in_nat = Dataset(os.path.join(path_out, 'FWI_zbar_FFMC_TN_pixel.nc'), 'r')
        ds_in_dra = Dataset(os.path.join(path_out, 'FWI_zbar_FFMC_TD_pixel.nc'), 'r')

    elif experiment == "EXP4":
        ds = Dataset(os.path.join(path_out, 'FWI_EXP4_weighted.nc'), mode='a')
        ds["MERRA2_DC"][:] = np.nan
        ds["MERRA2_DMC"][:] = np.nan
        ds["MERRA2_FFMC"][:] = np.nan
        ds["MERRA2_ISI"][:] = np.nan
        ds["MERRA2_BUI"][:] = np.nan
        ds["MERRA2_FWI"][:] = np.nan
        ds_in_nat = Dataset(os.path.join(path_out, 'FWI_zbar_FWI_TN_pixel.nc'), 'r')
        ds_in_dra = Dataset(os.path.join(path_out, 'FWI_zbar_FWI_TD_pixel.nc'), 'r')
        var = 'zbar'

    for var in FWI_variables:
        var_nat = ds_in_nat[var][0:6209, :, :].data
        var_dra = ds_in_dra[var][0:6209, :, :].data
        var_nat[np.isnan(var_nat)] = 0.0
        var_dra[np.isnan(var_dra)] = 0.0
        weight_nat = ds['weight_natural'][0:6209, :, :].data
        weight_dra = ds['weight_drained'][0:6209, :, :].data
        ds['MERRA2_FWI'][0:6209, :, :] = var_nat * weight_nat + var_dra * weight_dra
    ds.close()


