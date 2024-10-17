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
from pytesmo_anomaly import calc_climatology
from scipy.signal import argrelextrema
from datetime import datetime
import pickle
import math
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
font_title = 24
font_subtitle = 20
font_axes = 18
font_ticklabels = 16
font_text = 16
font_legend = 16

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

peatland_types = ['TN', 'TD']
CDF_types = ['pixel']
for CDF_type in CDF_types:
    for peatland_type in peatland_types:
        if peatland_type == 'TN':
            drainage_abb = 'Nat'
            lat_ts = 155
            lon_ts = 115
            year = 2013
        elif peatland_type == 'TD':
            drainage_abb = 'Dra'
            lat_ts = 27
            lon_ts = 222
            year = 2008

        '''----------------------------------------------Load datasets----------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # LOAD DATASETS
        # ---------------------------------------------------------------------------------------------
        fires_file = os.path.join(path_fires, 'Table_' + drainage_abb + CDF_type + '.csv')
        fire_data = pd.read_csv(fires_file, header=0)
        fire_data['start_date'] = pd.to_datetime(fire_data['start_date'])

        times = pd.date_range('2002-01-01', '2018-12-31', freq='D')
        fire_dates = pd.DatetimeIndex(fire_data.start_date)

        # years = np.unique(times.year)
        season = np.asarray(np.where((times.year == year))[0])

        ## Reference data
        ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
        DC_M2 = ds_ref['MERRA2_DC'][season, lat_ts, lon_ts].data
        DMC_M2 = ds_ref['MERRA2_DMC'][season, lat_ts, lon_ts].data
        FFMC_M2 = ds_ref['MERRA2_FFMC'][season, lat_ts, lon_ts].data
        ISI_M2 = ds_ref['MERRA2_ISI'][season, lat_ts, lon_ts].data
        BUI_M2 = ds_ref['MERRA2_BUI'][season, lat_ts, lon_ts].data
        FWI_M2 = ds_ref['MERRA2_FWI'][season, lat_ts, lon_ts].data

        # EXP1:
        ds_EXP1 = Dataset(os.path.join(path_out, 'FWI_zbar_DC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
        DC_EXP1 = ds_EXP1['MERRA2_DC'][season, lat_ts, lon_ts].data
        BUI_EXP1 = ds_EXP1['MERRA2_BUI'][season, lat_ts, lon_ts].data
        FWI_EXP1 = ds_EXP1['MERRA2_FWI'][season, lat_ts, lon_ts].data

        # EXP2:
        ds_EXP2 = Dataset(os.path.join(path_out, 'FWI_sfmc_DMC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
        DC_EXP2 = ds_EXP2['MERRA2_DC'][season, lat_ts, lon_ts].data
        DMC_EXP2 = ds_EXP2['MERRA2_DMC'][season, lat_ts, lon_ts].data
        BUI_EXP2 = ds_EXP2['MERRA2_BUI'][season, lat_ts, lon_ts].data
        FWI_EXP2 = ds_EXP2['MERRA2_FWI'][season, lat_ts, lon_ts].data

        # EXP3:
        ds_EXP3 = Dataset(os.path.join(path_out, 'FWI_sfmc_FFMC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
        DC_EXP3 = ds_EXP3['MERRA2_DC'][season, lat_ts, lon_ts].data
        DMC_EXP3 = ds_EXP3['MERRA2_DMC'][season, lat_ts, lon_ts].data
        FFMC_EXP3 = ds_EXP3['MERRA2_FFMC'][season, lat_ts, lon_ts].data
        ISI_EXP3 = ds_EXP3['MERRA2_ISI'][season, lat_ts, lon_ts].data
        BUI_EXP3 = ds_EXP3['MERRA2_BUI'][season, lat_ts, lon_ts].data
        FWI_EXP3 = ds_EXP3['MERRA2_FWI'][season, lat_ts, lon_ts].data

        # EXP4:
        ds_EXP4 = Dataset(os.path.join(path_out, 'FWI_zbar_FWI_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
        FWI_EXP4 = ds_EXP4['zbar'][season, lat_ts, lon_ts].data

        # EXP2b:
        ds_EXP2b = Dataset(os.path.join(path_out, 'FWI_zbar_DMC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
        FWI_EXP2b = ds_EXP2b['MERRA2_FWI'][season, lat_ts, lon_ts].data

        # EXP3b:
        ds_EXP3b = Dataset(os.path.join(path_out, 'FWI_zbar_FFMC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
        FWI_EXP3b = ds_EXP3b['MERRA2_FWI'][season, lat_ts, lon_ts].data

        '''----------------------------------------------Prepare data----------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # GET CORRECT ROWS OF FIRE DATASET
        # ---------------------------------------------------------------------------------------------
        lats = ds_ref["lat"][:]
        lons = ds_ref["lon"][:]

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

        # ---------------------------------------------------------------------------------------------
        # DETERMINE THRESHOLDS
        # ---------------------------------------------------------------------------------------------
        threshold_M2 = np.nanquantile(ds_ref['MERRA2_FWI'][:].data, 0.9)
        threshold_DC = np.nanquantile(ds_EXP1['MERRA2_FWI'][:].data, 0.9)
        threshold_DMC = np.nanquantile(ds_EXP2['MERRA2_FWI'][:].data, 0.9)
        threshold_FFMC = np.nanquantile(ds_EXP3['MERRA2_FWI'][:].data, 0.9)
        threshold_FWI = np.nanquantile(ds_EXP4['zbar'][:].data, 0.9)

        '''-----------------------------------------Plotting-----------------------------------------'''
        fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(15, 15), dpi=300)

        # Get shaded areas for the fires
        ignitions_raster[ignitions_raster > 0] = 1
        fires = ignitions_raster[season, lat_ts, lon_ts]
        fire_ends = duration_raster[season, lat_ts, lon_ts]
        fire_season = np.where(fires == 1)

        for fire in range(0, len(fires)):
            if fires[fire] == 1:
                ts = times[season]
                end = fire + fire_ends[fire].astype('int')
                # Get shaded area
                start_time = ts[fire]
                if end < len(ts):
                    end_time = ts[end]
                else:
                    end_time = ts[-1]

                ax[0].axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)
                ax[1].axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)
                ax[2].axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)
                ax[3].axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)
                ax[4].axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)
                ax[5].axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)

                # Get black line to indicate ignition
                ax[0].axvline(start_time, color='k')
                ax[1].axvline(start_time, color='k')
                ax[2].axvline(start_time, color='k')
                ax[3].axvline(start_time, color='k')
                ax[4].axvline(start_time, color='k')
                ax[5].axvline(start_time, color='k')

        # Get reference data
        ax[0].plot(times[season], FFMC_M2, color=palette[0], label='FWI$_{ref}$')
        ax[1].plot(times[season], DMC_M2, color=palette[0])
        ax[2].plot(times[season], DC_M2, color=palette[0])
        ax[3].plot(times[season], ISI_M2, color=palette[0])
        ax[4].plot(times[season], BUI_M2, color=palette[0])
        ax[5].plot(times[season], FWI_M2, color=palette[0])

        # Plot for each EXP the necessary (changed) FWI codes:
        ## EXP1:
        ax[2].plot(times[season], DC_EXP1, color=palette[1])
        ax[4].plot(times[season], BUI_EXP1, color=palette[1])
        ax[5].plot(times[season], FWI_EXP1, color=palette[1])

        ## EXP2:
        ax[1].plot(times[season], DMC_EXP2, color=palette[2])
        ax[4].plot(times[season], BUI_EXP2, color=palette[2])
        ax[5].plot(times[season], FWI_EXP2, color=palette[2])

        ## EXP3:
        ax[0].plot(times[season], FFMC_EXP3, color="tab:brown")
        ax[3].plot(times[season], ISI_EXP3, color="tab:brown")
        ax[5].plot(times[season], FWI_EXP3, color="tab:brown")

        ## EXP4:
        ax[5].plot(times[season], FWI_EXP4, color=palette[4])

        ## Plot horizontal lines for the thresholds of different experiments:
        ax[5].axhline(threshold_DC, color=palette[1], linestyle='--', linewidth=0.75)
        ax[5].axhline(threshold_DMC, color=palette[2], linestyle='--', linewidth=0.75)
        ax[5].axhline(threshold_FFMC, color='tab:brown', linestyle='--', linewidth=0.75)
        ax[5].axhline(threshold_FWI, color=palette[4], linestyle='--', linewidth=0.75)
        ax[5].axhline(threshold_M2, color=palette[0], linestyle='--', linewidth=0.75)

        # Set x-axis limits correct
        xmin = times[season[0]]
        xmax = times[season[-1]]

        ax[0].set_xlim(xmin, xmax)
        ax[1].set_xlim(xmin, xmax)
        ax[2].set_xlim(xmin, xmax)
        ax[3].set_xlim(xmin, xmax)
        ax[4].set_xlim(xmin, xmax)
        ax[5].set_xlim(xmin, xmax)

        # Set correct x- and y-ticks
        ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        ax[0].tick_params(axis='both', which='major', labelsize=font_ticklabels)
        ax[1].tick_params(axis='both', which='major', labelsize=font_ticklabels)
        ax[2].tick_params(axis='both', which='major', labelsize=font_ticklabels)
        ax[3].tick_params(axis='both', which='major', labelsize=font_ticklabels)
        ax[4].tick_params(axis='both', which='major', labelsize=font_ticklabels)
        ax[5].tick_params(axis='both', which='major', labelsize=font_ticklabels)

        ax[0].set_ylabel('FFMC', fontsize=font_axes)
        ax[1].set_ylabel('DMC', fontsize=font_axes)
        ax[2].set_ylabel('DC', fontsize=font_axes)
        ax[3].set_ylabel('ISI', fontsize=font_axes)
        ax[4].set_ylabel('BUI', fontsize=font_axes)
        ax[5].set_ylabel('FWI', fontsize=font_axes)

        plt.suptitle('Longitude: %.2f\N{DEGREE SIGN}; Latitude: %.2f\N{DEGREE SIGN}' % (lons[lon_ts], lats[lat_ts]),
                     fontsize=font_title)

        # Create legend
        plt.plot([], [], color=palette[0], label='FWI$_{ref}$')
        plt.plot([], [], color=palette[1], label="EXP1")
        plt.plot([], [], color=palette[2], label="EXP2")
        plt.plot([], [], color="tab:brown", label="EXP3")
        plt.plot([], [], color=palette[4], label="EXP4")
        # plt.scatter([], [], color='k', s=50, marker='^', ls='None', label='Miss -> Hit (all EXP)')
        # plt.scatter([], [], color='gray', s=50, marker='^', ls='None', label='Miss -> Hit (EXP3 & EXP4)')
        # plt.scatter([], [], color='tab:red', s=50, marker='d', ls='None', label='Hit for all (PEAT-)FWI')
        # plt.scatter([], [], color='tab:red', s=50, marker='X', ls='None', label='Miss for all (PEAT-)FWI')

        ax[5].legend(bbox_to_anchor=(0.8, -0.2), ncol=5, fontsize=font_legend)
        fig.subplots_adjust(top=0.945, bottom=0.11, left=0.1, right=0.955, hspace=0.2, wspace=0.2)
        plt.show()




