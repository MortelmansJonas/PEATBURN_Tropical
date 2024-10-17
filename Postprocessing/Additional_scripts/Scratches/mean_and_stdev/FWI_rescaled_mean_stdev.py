#!/usr/bin/env python
"""
This script is used for the FWI calculations based on the new rescaled (based on mean and stdev) PEATCLSM variables
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# Load Packages
from python_functions import create_file_from_source
from netCDF4 import Dataset
import os
from PEATBURN import *
import numpy as np

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

PEATCLSM_variables = ['zbar', 'sfmc', 'sfmc']
FWI_variables = ['DC', 'DMC', 'FFMC']
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
            ds_src = Dataset(os.path.join(path_out, 'Rescaled_' + PEATCLSM_variable + '_MERRA2_' + FWI_variable + '_'
                                          + peatland_type + '_' + type + '.nc'), 'r')

            # endregion

            '''-------------------------------------------Create Datasets-------------------------------------------'''
            # region
            if not os.path.exists(os.path.join(path_out, 'FWI_Rescaled_' + PEATCLSM_variable + '_' + FWI_variable +
                                                         '_' + peatland_type + '_' + type + '.nc')):
                print("Creation of a new file")
                # Create a new file, with the same dimensions etc as the original file:
                create_file_from_source(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'),
                                        os.path.join(path_out, 'FWI_Rescaled_' + PEATCLSM_variable + '_' +
                                                     FWI_variable + '_' + peatland_type + '_' + type + '.nc'))

                # And then set everything to nan, so that the file is empty.
                ds_out = Dataset(os.path.join(path_out, 'FWI_Rescaled_' + PEATCLSM_variable + '_' +
                                              FWI_variable + '_' + peatland_type + '_' + type + '.nc'), 'a')
                ds_out['MERRA2_DC'][:] = np.nan
                ds_out['MERRA2_DMC'][:] = np.nan
                ds_out['MERRA2_FFMC'][:] = np.nan
                ds_out['MERRA2_BUI'][:] = np.nan
                ds_out['MERRA2_ISI'][:] = np.nan
                ds_out['MERRA2_FWI'][:] = np.nan
            else:
                print("Add data to an already existing file")
                ds_out = Dataset(os.path.join(path_out, 'FWI_Rescaled_' + PEATCLSM_variable + '_' +
                                              FWI_variable + '_' + peatland_type + '_' + type + '.nc'), 'a')

            # endregion

            '''-------------------------------------------FWI calculations-------------------------------------------'''
            # region

            print('FWI calculation:')
            if FWI_variable == 'DC':
                DC = ds_src[PEATCLSM_variable][:].data
                DMC = ds_ref['MERRA2_DMC'][0:6209, :, :].data
                ISI = ds_ref['MERRA2_ISI'][0:6209, :, :].data
                FFMC = ds_ref['MERRA2_FFMC'][0:6209, :, :].data

                BUI = BUICalc_time(DMC, DC)

            elif FWI_variable == 'DMC':
                ds_DC = Dataset(os.path.join(path_out, 'Rescaled_zbar_MERRA2_DC_' + peatland_type + '_' + type +
                                             '.nc'), 'r')

                DC = ds_DC['zbar'][0:6209, :, :].data
                DMC = ds_src[PEATCLSM_variable][:].data
                ISI = ds_ref['MERRA2_ISI'][0:6209, :, :].data
                FFMC = ds_ref['MERRA2_FFMC'][0:6209, :, :].data

                BUI = BUICalc_time(DMC, DC)

            elif FWI_variable == 'FFMC':
                ds_wind = Dataset(os.path.join('/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical/Reference',
                                               'MERRA2_combined_regridded.nc'), 'r')
                ds_DC = Dataset(os.path.join(path_out, 'Rescaled_zbar_MERRA2_DC_' + peatland_type + '_' + type +
                                             '.nc'), 'r')
                ds_DMC = Dataset(os.path.join(path_out, 'Rescaled_sfmc_MERRA2_DMC_' + peatland_type + '_' +
                                              type + '.nc'), 'r')

                wind = ds_wind['MERRA2_wdSpd'][:].data
                DC = ds_DC['zbar'][0:6209, :, :].data
                DMC = ds_DMC['sfmc'][0:6209, :, :].data
                FFMC = ds_src[PEATCLSM_variable][:].data

                ISI = ISICalc_time(wind, FFMC)

                BUI = BUICalc_time(DMC, DC)

            ds_out['MERRA2_FWI'][0:6209, :, :] = FWICalc_time(BUI, ISI)

            ds_out['MERRA2_ISI'][0:6209, :, :] = ISI
            ds_out['MERRA2_BUI'][0:6209, :, :] = BUI
            ds_out['MERRA2_FFMC'][0:6209, :, :] = FFMC
            ds_out['MERRA2_DMC'][0:6209, :, :] = DMC
            ds_out['MERRA2_DC'][0:6209, :, :] = DC

            ds_out.close()
            # endregion
