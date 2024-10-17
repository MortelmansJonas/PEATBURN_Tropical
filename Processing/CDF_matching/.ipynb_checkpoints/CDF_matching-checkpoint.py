#!/usr/bin/env python
"""
This script is used for the CDF matching
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# Load Packages
from python_functions import create_file_from_source
from netCDF4 import Dataset
import os
from PEATBURN import CDF_matching_pixel, CDF_matching_domain
import numpy as np
from copy import deepcopy as dcopy

# ---------------------------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------------------------
# Define the variables
# PEATCLSM_variable = input('sfmc, rzmc, or zbar: ')
# FWI_variable = input('FFMC, DMC, DC, ISI, BUI, or FWI: ')
# peatland_type = input('TN (natural) or TD (drained): ')
#
# if peatland_type == 'TN':
#     drainage_abb = 'Nat'
# elif peatland_type == 'TD':
#     drainage_abb = 'Dra'
#
# CDF_type = input('domain if domain-wise CDF-matching, pixel if pixel per pixel CDF-matching: ')

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
    print('Error: Tier can only be Hortense, or Genius.')

# endregion

# Define the different experiments:
# order: EXP1, EXP2, EXP2b, EXP3, EXP3b, EXP4
PEATCLSM_variables = ['zbar', 'sfmc', 'zbar',  'sfmc', 'zbar', 'zbar']
FWI_variables = ['DC', 'DMC', 'DMC', 'FFMC', 'FFMC', 'FWI']
# Define the two types of peatlands/PEATCLSM models (TN = tropical natural; TD = tropical drained)
peatland_types = ['TN', 'TD']
# Remnant of original setup: pixel-wise CDF matching (originally also domain-wise)
CDF_types = ['pixel']

# Then loop over all options
## Start with looping over the experiments (use 'index' to get the corresponding FWI_variable)
for index, PEATCLSM_variable in enumerate(PEATCLSM_variables):
    FWI_variable = FWI_variables[index]
    ## Then loop over the peatland types
    for peatland_type in peatland_types:
        ## And lastly over the CDF types
        for CDF_type in CDF_types:
            # The files of the FWI_masking are named slightly different -> specify when which name should be used.
            if peatland_type == 'TN':
                drainage_abb = 'Nat'
            elif peatland_type == 'TD':
                drainage_abb = 'Dra'

            print(PEATCLSM_variable + '_' + FWI_variable + '_' + peatland_type + '_' + CDF_type)

            '''--------------------------------------------Load datasets--------------------------------------------'''
            # ---------------------------------------------------------------------------------------------
            # LOAD DATASETS
            # ---------------------------------------------------------------------------------------------
            # The original FWI calculations
            ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
            # PEATCLSM Tropical Natural output
            ds_src = Dataset(os.path.join(path_peatlands, 'PEATCLSM_' + peatland_type + '_masked_' + drainage_abb +
                                          '_20022018.nc'), 'r')

            # ---------------------------------------------------------------------------------------------
            # MASK THE DATA TO THE PEATLANDS
            # ---------------------------------------------------------------------------------------------
            ref = ds_ref['MERRA2_' + FWI_variable][0:6209, :, :].data  # The target to which we want to CDF match

            dim_time = ref.shape[0]
            dim_lat = ref.shape[1]
            dim_lon = ref.shape[2]

            # To verify, also crossmask with the original FWI calculations.
            mask_ref = np.where(~np.isnan(ref), 1, np.nan)
            src = np.multiply(ds_src[PEATCLSM_variable][:].data, mask_ref) * -1

            # Create files if they do not yet exist. Otherwise open files to add data to them
            if not os.path.exists(os.path.join(path_out, 'CDF_matched_' + PEATCLSM_variable + '_MERRA2_' +
                                                         FWI_variable + '_' + peatland_type + '_' + CDF_type + '.nc')):
                print("Creation of a new file")
                # Create a new file, with the same dimensions etc as the original file:
                create_file_from_source(os.path.join(path_peatlands, 'PEATCLSM_' + peatland_type + '_masked_' +
                                                     drainage_abb + '_20022018.nc'),
                                        os.path.join(path_out, 'CDF_matched_' + PEATCLSM_variable + '_MERRA2_' +
                                                     FWI_variable + '_' + peatland_type + '_' + CDF_type + '.nc'))

                # And then set everything to nan, so that the file is empty.
                ds_out = Dataset(os.path.join(path_out, 'CDF_matched_' + PEATCLSM_variable + '_MERRA2_' +
                                              FWI_variable + '_' + peatland_type + '_' + CDF_type + '.nc'), 'a')
                ds_out['zbar'][:] = np.nan
                ds_out['sfmc'][:] = np.nan
                ds_out['rzmc'][:] = np.nan
            else:
                print("Add data to an already existing file")
                ds_out = Dataset(os.path.join(path_out, 'CDF_matched_' + PEATCLSM_variable + '_MERRA2_' +
                                              FWI_variable + '_' + peatland_type + '_' + CDF_type + '.nc'), 'a')

            # CDF-matching
            print("CDF-matching")
            if CDF_type == 'pixel':
                ds_out[PEATCLSM_variable][:] = CDF_matching_pixel(src, ref, dim_time, dim_lat, dim_lon,
                                                                  start_lat=0, start_lon=0)
            elif CDF_type == 'domain':
                ds_out[PEATCLSM_variable][:] = CDF_matching_domain(src, ref, dim_time, dim_lat, dim_lon)

            ds_out.close()
