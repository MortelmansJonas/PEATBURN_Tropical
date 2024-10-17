#!/usr/bin/env python
"""
This script is used to rescale the PEATCLSM data to the FWI data based on the mean and stdev
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# Load Packages
from python_functions import create_file_from_source
from netCDF4 import Dataset
import os
from PEATBURN import Rescaling_mean_stdev
import numpy as np
import pandas as pd
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
    print('Error: Tier can only be Hortense, or Genius.')

# endregion

# PEATCLSM_variables = ['zbar', 'zbar', 'sfmc', 'sfmc']
# FWI_variables = ['DC', 'FWI', 'DMC', 'FFMC']
PEATCLSM_variables = ['zbar']
FWI_variables = ['FWI']
peatland_types = ['TN', 'TD']
types = ['pixel']

for index, PEATCLSM_variable in enumerate(PEATCLSM_variables):
    FWI_variable = FWI_variables[index]
    for peatland_type in peatland_types:
        for type in types:
            if peatland_type == 'TN':
                drainage_abb = 'Nat'
            elif peatland_type == 'TD':
                drainage_abb = 'Dra'

            print(PEATCLSM_variable + '_' + FWI_variable + '_' + peatland_type + '_' + type)


            '''--------------------------------------------Load datasets--------------------------------------------'''
            # region
            # ---------------------------------------------------------------------------------------------
            # LOAD DATASETS
            # ---------------------------------------------------------------------------------------------
            # The original FWI calculations
            ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
            # PEATCLSM Tropical Natural output
            ds_src = Dataset(os.path.join(path_peatlands, 'PEATCLSM_' + peatland_type + '_masked_' + drainage_abb + '_20022018.nc'),
                             'r')
            # endregion

            # ---------------------------------------------------------------------------------------------
            # MASK THE DATA TO THE PEATLANDS
            # ---------------------------------------------------------------------------------------------
            ref = ds_ref['MERRA2_' + FWI_variable][0:6209, :, :].data  # The target to which we want to match

            dim_time = ref.shape[0]
            dim_lat = ref.shape[1]
            dim_lon = ref.shape[2]

            # To verify, also crossmask with the original FWI calculations.
            mask_ref = np.where(~np.isnan(ref), 1, np.nan)
            if PEATCLSM_variable == 'sfmc':
                src = 1 / np.multiply(ds_src[PEATCLSM_variable][:].data, mask_ref)
            elif PEATCLSM_variable == 'zbar':
                src = np.multiply(ds_src[PEATCLSM_variable][:].data, mask_ref) * -1

            ref = np.log10(ref)
            print(pd.DataFrame(ref.flatten()).describe())
            print(np.nanmean(ref))

            src = np.log10(src)
            print(pd.DataFrame(src.flatten()).describe())
            print(np.nanmean(src))

            if not os.path.exists(os.path.join(path_out, 'Rescaled_' + PEATCLSM_variable + '_MERRA2_' +
                                                         FWI_variable + '_' + peatland_type + '_' + type + '.nc')):
                print("Creation of a new file")
                # Create a new file, with the same dimensions etc as the original file:
                create_file_from_source(os.path.join(path_peatlands, 'PEATCLSM_' + peatland_type + '_masked_' +
                                                     drainage_abb + '_20022018.nc'),
                                        os.path.join(path_out, 'Rescaled_' + PEATCLSM_variable + '_MERRA2_' +
                                                     FWI_variable + '_' + peatland_type + '_' + type + '.nc'))

                # And then set everything to nan, so that the file is empty.
                ds_out = Dataset(os.path.join(path_out, 'Rescaled_' + PEATCLSM_variable + '_MERRA2_' +
                                              FWI_variable + '_' + peatland_type + '_' + type + '.nc'), 'a')
                ds_out['zbar'][:] = np.nan
                ds_out['sfmc'][:] = np.nan
                ds_out['rzmc'][:] = np.nan
            else:
                print("Add data to an already existing file")
                ds_out = Dataset(os.path.join(path_out, 'Rescaled_' + PEATCLSM_variable + '_MERRA2_' +
                                              FWI_variable + '_' + peatland_type + '_' + type + '.nc'), 'a')

            # Rescaling
            print("Rescaling")
            if type == 'pixel':
                ds_out[PEATCLSM_variable][:] = 10 ** Rescaling_mean_stdev(src, ref, pixel=True)
                print(np.nanmin(ds_out[PEATCLSM_variable][:]))
                print(np.log10(pd.DataFrame(ds_out[PEATCLSM_variable][:].flatten())).describe())
            elif type == 'domain':
                ds_out[PEATCLSM_variable][:] = Rescaling_mean_stdev(src, ref, pixel=False)

            ds_out.close()
