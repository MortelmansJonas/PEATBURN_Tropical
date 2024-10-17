#!/usr/bin/env python
"""
This script is used for the CDF matching
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# Load Packages
from python_functions import *
import os
from pytesmo_anomaly import calc_climatology, calc_anomaly
from tqdm import tqdm
import dask.dataframe as dd
from dask import delayed, compute

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

# Define a delayed function for the inner loop:
@delayed
def process_lat_lon(lat, lon):
    '''--------------------------------Calculate climatology--------------------------------'''
    ## FWI_ref

    ser_ref = pd.Series(FWI_M2[:, lat, lon], index=times)
    clim_ref = calc_climatology(ser_ref, respect_leap_years=True, interpolate_leapday=True,
                                fillna=False)

    ## EXP1
    ser_in = pd.Series(FWI_in[:, lat, lon], index=times)
    clim_in = calc_climatology(ser_in, respect_leap_years=True, interpolate_leapday=True,
                               fillna=False)

    '''----------------------------------Seasonal CDF match----------------------------------'''
    # ------------------------------------------------------------------------------------------
    # CALCULATE THE ANOMALY
    # ------------------------------------------------------------------------------------------
    anom_ref = calc_anomaly(ser_ref, climatology=clim_ref, return_clim=True)
    anom_in = calc_anomaly(ser_in, climatology=clim_in, return_clim=True)

    # ------------------------------------------------------------------------------------------
    # ADD ANOMALY TO CLIM OF REFERENCE
    # ------------------------------------------------------------------------------------------
    ds_out['MERRA2_FWI'][0:6209, lat, lon] = anom_ref['climatology'].values + \
                                             anom_in['anomaly'].values


# Define the different experiments:
# order: EXP1, EXP2, EXP2b, EXP3, EXP3b, EXP4
# PEATCLSM_variables = ['zbar', 'sfmc', 'zbar', 'sfmc', 'zbar', 'zbar']
# FWI_variables = ['DC', 'DMC', 'DMC', 'FFMC', 'FFMC', 'FWI']
PEATCLSM_variables = ['zbar']
FWI_variables = ['DC']
# Define the two types of peatlands/PEATCLSM models (TN = tropical natural; TD = tropical drained)
peatland_types = ['TN', 'TD']
# Remnant of original setup: pixel-wise CDF matching (originally also domain-wise)
CDF_types = ['pixel']

# Then loop over all options
## Start with looping over the experiments (use 'index' to get the corresponding FWI_variable)
times = pd.date_range('2002-01-01', '2018-12-31', freq='D')
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
            ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
            FWI_M2 = ds_ref['MERRA2_FWI'][0:6209, :, :].data

            ds_in = Dataset(os.path.join(path_out, 'FWI_' + PEATCLSM_variable + '_' +
                                         FWI_variable + '_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
            FWI_in = ds_in['MERRA2_FWI'][0:6209, :, :].data

            '''---------------------------------------Open new output dataset---------------------------------------'''
            ds_out = Dataset(os.path.join(path_out, 'Seasonal_' + PEATCLSM_variable + '_' +
                                          FWI_variable + '_' + peatland_type + '_' + CDF_type + '.nc'), 'a')

            ds_out['MERRA2_FWI'][:] = np.nan

            '''----------------------------------------------For loops----------------------------------------------'''
            lats = ds_ref['lat'][:]
            lons = ds_ref['lon'][:]

            delayed_tasks = []

            for lat in tqdm(range(len(lats))):
                print(lat)
                if np.isnan(FWI_M2[:, lat, :]).all():
                    continue
                else:
                    for lon in range(len(lons)):
                        if np.isnan(FWI_M2[:, lat, lon]).all():
                            continue
                        else:
                            delayed_tasks.append(process_lat_lon(lat, lon))

            compute(delayed_tasks)
            print('hold')
            ds_out.close()
