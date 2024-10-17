#!/usr/bin/env python
"""
This script is used to create a time series of the different FWI components.
The season and location are determined in the script Determine_loc_table.py based on where all EXP turned a miss into
a hit. The indices were determined to be [908, 41, 2769] -> so lat_ts = 41, lon_ts = 2769 and the year is 2012.
"""
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib.ticker as mticker
from python_functions import *
import os
import seaborn2 as sns
from copy import deepcopy as dcopy


# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
font_title = 20
font_subtitle = 16
font_axes = 12
font_ticklabels = 10
font_text = 10
font_legend = 10

# colorblind proof:
palette = sns.color_palette('colorblind')

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
    print('Error: Tier can only be Hortense or Genius.')

'''----------------------------------------------Load fire datasets----------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# LOAD FIREDATASETS
# ---------------------------------------------------------------------------------------------
fires_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')
fires = pd.read_csv(fires_file, header=0)
fires['start_date'] = pd.to_datetime(fires['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_data = fires[fires['Drained_I'] >= 0].reset_index(drop=True)

times = pd.date_range('2002-01-01', '2018-12-31', freq='D')

fire_dates2 = pd.DatetimeIndex(fire_data.start_date)
fire_data = fire_data[fire_dates2.year >= 2002].reset_index(drop=True)  # Check if still necessary
fire_dates2 = fire_dates2[fire_dates2.year >= 2002]  # Check if still necessary


peatland_types = ['TN', 'TD']

for peatland_type in peatland_types:
    if peatland_type == 'TN':
        drainage_abb = 'Nat'
        title = 'Undrained'

        fire_file = dcopy(fire_data)
        fire_file = fire_file[fire_file['Drained_I'] == 0].reset_index(drop=True)

        lat_ts = 155
        lon_ts = 115
        year = 2013

    elif peatland_type == 'TD':
        drainage_abb = 'Dra'
        title = 'Drained'

        fire_file = dcopy(fire_data)
        fire_file = fire_file[fire_file['Drained_I'] == 1].reset_index(drop=True)

        lat_ts = 27
        lon_ts = 222
        year = 2013

    # Determine the possible locations and loop over them to make many time series plots:
    locations = np.argwhere((fire_file.M2H_EXP1 == 1) & (fire_file.M2H_EXP2 == 1) &
                            (fire_file.M2H_EXP3 == 1) & (fire_file.M2H_EXP4 == 1) &
                            (fire_file.M2H_EXP2b == 1) & (fire_file.M2H_EXP3b == 1))
    unique_values_col, counts_col = np.unique(locations[:, [1, 2]], axis=0, return_counts=True)
    unique = unique_values_col[counts_col >= 10]
    counts = counts_col[counts_col >= 10]

    '''----------------------------------------------Load datasets----------------------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # LOAD DATASETS
    # ---------------------------------------------------------------------------------------------
    ## FWI_ref
    ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
    FWI_M2 = ds_ref['MERRA2_FWI'][0:6209, lat_ts, lon_ts].data

    # EXP1:
    ds_EXP1 = Dataset(os.path.join(path_out, 'FWI_zbar_DC_' + peatland_type + '_pixel.nc'), 'r')
    FWI_EXP1 = ds_EXP1['MERRA2_FWI'][0:6209, lat_ts, lon_ts].data

    # EXP2:
    ds_EXP2 = Dataset(os.path.join(path_out, 'FWI_sfmc_DMC_' + peatland_type + '_pixel.nc'), 'r')
    FWI_EXP2 = ds_EXP2['MERRA2_FWI'][0:6209, lat_ts, lon_ts].data

    # EXP3:
    ds_EXP3 = Dataset(os.path.join(path_out, 'FWI_sfmc_FFMC_' + peatland_type + '_pixel.nc'), 'r')
    FWI_EXP3 = ds_EXP3['MERRA2_FWI'][0:6209, lat_ts, lon_ts].data

    # EXP4:
    ds_EXP4 = Dataset(os.path.join(path_out, 'FWI_zbar_FWI_' + peatland_type + '_pixel.nc'), 'r')
    FWI_EXP4 = ds_EXP4['zbar'][0:6209, lat_ts, lon_ts].data

    # EXP2b:
    ds_EXP2b = Dataset(os.path.join(path_out, 'FWI_zbar_DMC_' + peatland_type + '_pixel.nc'), 'r')
    FWI_EXP2b = ds_EXP2b['MERRA2_FWI'][0:6209, lat_ts, lon_ts].data

    # EXP3b:
    ds_EXP3b = Dataset(os.path.join(path_out, 'FWI_zbar_FFMC_' + peatland_type + '_pixel.nc'), 'r')
    FWI_EXP3b = ds_EXP3b['MERRA2_FWI'][0:6209, lat_ts, lon_ts].data

    '''---------------------------------------------Prepare time series---------------------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # GET THE LOCATIONS CLOSEST TO THE GIVEN LAT AND LON
    # ---------------------------------------------------------------------------------------------
    lats = ds_ref['lat'][:].data
    lons = ds_ref['lon'][:].data

    # Create raster of drained fires
    lat_indices = np.argmin(np.abs(lats - fire_data['latitude_I'][:, np.newaxis]), axis=1)
    lon_indices = np.argmin(np.abs(lons - fire_data['longitude_I'][:, np.newaxis]), axis=1)

    ignitions_raster = np.zeros((6209, 266, 643))
    duration_raster = np.zeros((6209, 266, 643))

    for i in range(len(fire_data.latitude_I)):
        time_start = (pd.to_datetime(fire_data['start_date'][i], format='%Y-%m-%d') - pd.to_datetime(
            '2010-01-01')).days
        time_end = time_start + fire_data['duration'][i]

        ignitions_raster[time_start, lat_indices[i], lon_indices[i]] += 1
        duration_raster[time_start, lat_indices[i], lon_indices[i]] = fire_data['duration'][i]

    ignitions_raster[ignitions_raster > 0] = 1

    '''------------------------------------------Plotting------------------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # PLOT THE CLIMATOLOGIES
    # ---------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=300)

    # plot the fires:
    fires = ignitions_raster[:, lat_ts, lon_ts]
    fire_ends = duration_raster[:, lat_ts, lon_ts]
    fire_season = np.where(fires == 1)

    for fire in range(0, len(fires)):
        if fires[fire] == 1:
            end = fire + fire_ends[fire].astype('int')
            # Get shaded area
            start_time = times[fire]
            if end < len(times):
                end_time = times[end]
            else:
                end_time = times[-1]

            ax.axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)

            # Get black line to indicate ignition
            ax.axvline(start_time, color='k')

    # plot the FWI:
    ## Ref
    p1, = ax.plot(times, FWI_M2, color=palette[0], label='FWI$_{ref}$', linewidth=1, zorder=2)
    ax.set_ylabel('FWI', fontsize=font_axes)
    # ax.set_xlabel('DOY')

    # The different experiments can be on the same axis
    p2, = ax.plot(times, FWI_EXP1, color=palette[1], label='FWI$_{EXP1}$', linewidth=1, zorder=2)
    p3, = ax.plot(times, FWI_EXP2, color=palette[2], label='FWI$_{EXP2}$', linewidth=1, zorder=2)
    p4, = ax.plot(times, FWI_EXP2b, color=palette[2], label='FWI$_{EXP2b}$', linewidth=1, linestyle='--', zorder=2)
    p5, = ax.plot(times, FWI_EXP3, color='tab:brown', label='FWI$_{EXP3}$', linewidth=1, zorder=2)
    p6, = ax.plot(times, FWI_EXP3b, color='tab:brown', label='FWI$_{EXP3b}$', linewidth=1, linestyle='--', zorder=2)
    p7, = ax.plot(times, FWI_EXP2, color=palette[4], label='FWI$_{EXP4}$', linewidth=1)

    plt.legend(handles=[p1, p2, p3, p4, p5, p6, p7], fontsize=font_legend)
    plt.title(title, fontsize=font_title)
    plt.show()
    # plt.savefig(os.path.join(path_figs, 'TS_test'))

    print('hold')




