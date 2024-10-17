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

# '''----------------------------------------------Load fire datasets----------------------------------------------'''
# # ---------------------------------------------------------------------------------------------
# # LOAD FIREDATASETS
# # ---------------------------------------------------------------------------------------------
# fires_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')
# fires = pd.read_csv(fires_file, header=0)
# fires['start_date'] = pd.to_datetime(fires['start_date'])
# # only get those fires that are in our domain and in peatclsm pixels:
# fire_data = fires[fires['Drained_I'] >= 0].reset_index(drop=True)
#
# times = pd.date_range('2002-01-01', '2018-12-31', freq='D')
#
# fire_dates2 = pd.DatetimeIndex(fire_data.start_date)
# fire_data = fire_data[fire_dates2.year >= 2002].reset_index(drop=True)  # Check if still necessary
# fire_dates2 = fire_dates2[fire_dates2.year >= 2002]  # Check if still necessary

# peatland_types = ['TN', 'TD']
peatland_types = ['TD']
CDF_types = ['pixel']
for CDF_type in CDF_types:
    for peatland_type in peatland_types:
        if peatland_type == 'TN':
            drainage_abb = 'Nat'
        elif peatland_type == 'TD':
            drainage_abb = 'Dra'

        '''----------------------------------------------Load datasets----------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # LOAD FIRE DATASET
        # ---------------------------------------------------------------------------------------------
        fires_file = os.path.join(path_fires, 'Table_' + drainage_abb + CDF_type + '.csv')
        fire_data = pd.read_csv(fires_file, header=0)
        fire_data['start_date'] = pd.to_datetime(fire_data['start_date'])

        fire_data = fire_data.drop_duplicates(subset=['start_date', 'latitude_I', 'longitude_I', 'end_date'])

        times = pd.date_range('2002-01-01', '2018-12-31', freq='D')
        fire_dates = pd.DatetimeIndex(fire_data.start_date)

        ## Reference data
        ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
        FWI_M2 = ds_ref["MERRA2_FWI"][0:6209, :, :].data

        lats = ds_ref['lat'][:].data
        lons = ds_ref['lon'][:].data

        # ---------------------------------------------------------------------------------------------
        # RASTERIZE FIRE DATASET FOR LATER ON
        # ---------------------------------------------------------------------------------------------
        ignitions_raster = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))
        duration_raster = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))
        M2H_EXP1 = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))
        M2H_EXP2 = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))
        M2H_EXP3 = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))
        M2H_EXP4 = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))
        M2H_EXP2b = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))
        M2H_EXP3b = np.zeros((FWI_M2.shape[0], FWI_M2.shape[1], FWI_M2.shape[2]))

        for i in range(len(fire_data.latitude_I)):
            time_start = (pd.to_datetime(fire_data['start_date'][i], format='%Y-%m-%d') - pd.to_datetime(
                '2010-01-01')).days
            time_end = time_start + fire_data['duration'][i]
            lat_diffs = abs(lats - fire_data['latitude_I'][i])
            lon_diffs = abs(lons - fire_data['longitude_I'][i])

            lat_inds = np.where(lat_diffs == np.nanmin(lat_diffs))[0][0] + 1
            lon_inds = np.where(lon_diffs == np.nanmin(lon_diffs))[0][0] + 1
            ignitions_raster[time_start, lat_inds, lon_inds] += 1
            duration_raster[time_start, lat_inds, lon_inds] = fire_data['duration'][i]
            M2H_EXP1[time_start, lat_inds, lon_inds] = fire_data["M2H_EXP1"][i]
            M2H_EXP2[time_start, lat_inds, lon_inds] = fire_data["M2H_EXP2"][i]
            M2H_EXP3[time_start, lat_inds, lon_inds] = fire_data["M2H_EXP3"][i]
            M2H_EXP4[time_start, lat_inds, lon_inds] = fire_data["M2H_EXP4"][i]
            M2H_EXP2b[time_start, lat_inds, lon_inds] = fire_data["M2H_EXP2b"][i]
            M2H_EXP3b[time_start, lat_inds, lon_inds] = fire_data["M2H_EXP3b"][i]

        locations = np.argwhere((M2H_EXP1 == 1) & (M2H_EXP2 == 1) & (M2H_EXP3 == 1) & (M2H_EXP4 == 1))
        unique_values_col, counts_col = np.unique(locations[:, [1, 2]], axis=0, return_counts=True)
        unique = unique_values_col[counts_col >= 10]
        counts = counts_col[counts_col >= 10]

        for i in range(len(unique)):
            times = locations[(locations[:, 1] == unique[i, 0]) & (locations[:, 2] == unique[i, 1])]
            print(times)
        print("locations shape: " + str(locations.shape[0]))

        # Let's loop over the possible locations and corresponding fire seasons:
        for i in range(locations.shape[0]):
            print("location: " + str(locations[i, 0]))
            # Get the year that belongs to the location
            year = times[locations[i, 0]].year

            time_inds = np.where(times.year == year)[0]

            '''--------------------------------------------Load FWI data--------------------------------------------'''
            # ---------------------------------------------------------------------------------------------
            # LOAD FWI DATASETS
            # ---------------------------------------------------------------------------------------------
            # Reference:
            latitude = locations[i, 1]  # For some reason, it get's shifted here
            longitude = locations[i, 2]
            if ~np.isnan(np.nanmean(ds_ref['MERRA2_DC'][time_inds, latitude, longitude].data)):
                FWI_M2 = ds_ref['MERRA2_FWI'][time_inds, latitude, longitude].data

                # EXP1:
                ds_EXP1 = Dataset(os.path.join(path_out, 'FWI_zbar_DC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
                FWI_EXP1 = ds_EXP1['MERRA2_FWI'][time_inds, latitude, longitude].data

                # EXP2:
                ds_EXP2 = Dataset(os.path.join(path_out, 'FWI_sfmc_DMC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
                FWI_EXP2 = ds_EXP2['MERRA2_FWI'][time_inds, latitude, longitude].data

                # EXP2b:
                ds_EXP2b = Dataset(os.path.join(path_out, 'FWI_zbar_DMC_' + peatland_type + '_' + CDF_type + '.nc'),
                                   'r')
                FWI_EXP2b = ds_EXP2b['MERRA2_FWI'][time_inds, latitude, longitude].data

                # EXP3:
                ds_EXP3 = Dataset(os.path.join(path_out, 'FWI_sfmc_FFMC_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
                FWI_EXP3 = ds_EXP3['MERRA2_FWI'][time_inds, latitude, longitude].data

                ds_EXP3b = Dataset(os.path.join(path_out, 'FWI_zbar_FFMC_' + peatland_type + '_' + CDF_type + '.nc'),
                                   'r')
                FWI_EXP3b = ds_EXP3b['MERRA2_FWI'][time_inds, latitude, longitude].data

                # EXP4:
                ## Only FWI is available
                ds_EXP4 = Dataset(os.path.join(path_out, 'FWI_zbar_FWI_' + peatland_type + '_' + CDF_type + '.nc'), 'r')
                FWI_EXP4 = ds_EXP4['zbar'][time_inds, latitude, longitude].data

                '''-----------------------------------------Plotting-----------------------------------------'''
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), dpi=300)

                # Get shaded areas for the fires
                ignitions_raster[ignitions_raster > 0] = 1
                fires = ignitions_raster[time_inds, locations[i, 1], locations[i, 2]]
                fire_ends = duration_raster[time_inds, locations[i, 1], locations[i, 2]]
                fire_time_inds = np.where(fires == 1)

                for fire in range(0, len(fires)):
                    if fires[fire] == 1:
                        ts = times[time_inds]
                        end = fire + fire_ends[fire].astype('int')
                        # Get shaded area
                        start_time = ts[fire]
                        if end < len(ts):
                            end_time = ts[end]
                        else:
                            end_time = ts[-1]

                        ax.axvspan(start_time, end_time, facecolor='lightgray', alpha=0.75)

                        # Get black line to indicate ignition
                        ax.axvline(start_time, color='k')

                # Get reference data
                ax.plot(times[time_inds], FWI_M2, color=palette[0])

                # Plot for each EXP the necessary (changed) FWI codes:
                ax.plot(times[time_inds], FWI_EXP1, color=palette[1])

                ## EXP2:
                ax.plot(times[time_inds], FWI_EXP2, color=palette[2])

                ## EXP2b:
                ax.plot(times[time_inds], FWI_EXP2b, color=palette[2], linestyle='--')

                ## EXP3:
                ax.plot(times[time_inds], FWI_EXP3, color="tab:brown")

                ## EXP3b:
                ax.plot(times[time_inds], FWI_EXP3b, color="tab:brown", linestyle='--')

                ## EXP4:
                ax.plot(times[time_inds], FWI_EXP4, color=palette[4])

                ## Plot horizontal lines for the thresholds of different experiments:
                threshold_ref = np.nanquantile(ds_ref['MERRA2_FWI'][:].data, 0.9)
                threshold_EXP1 = np.nanquantile(ds_EXP1['MERRA2_FWI'][:].data, 0.9)
                threshold_EXP2 = np.nanquantile(ds_EXP2['MERRA2_FWI'][:].data, 0.9)
                threshold_EXP2b = np.nanquantile(ds_EXP2b['MERRA2_FWI'][:].data, 0.9)
                threshold_EXP3 = np.nanquantile(ds_EXP3['MERRA2_FWI'][:].data, 0.9)
                threshold_EXP3b = np.nanquantile(ds_EXP3b['MERRA2_FWI'][:].data, 0.9)
                threshold_EXP4 = np.nanquantile(ds_EXP4['zbar'][:].data, 0.9)

                ax.axhline(threshold_EXP1, color=palette[1], linestyle='--', linewidth=0.75)
                ax.axhline(threshold_EXP2, color=palette[2], linestyle='--', linewidth=0.75)
                ax.axhline(threshold_EXP2b, color=palette[2], linestyle=':', linewidth=0.75)
                ax.axhline(threshold_EXP3, color='tab:brown', linestyle='--', linewidth=0.75)
                ax.axhline(threshold_EXP3b, color='tab:brown', linestyle=':', linewidth=0.75)
                ax.axhline(threshold_EXP4, color=palette[4], linestyle='--', linewidth=0.75)
                ax.axhline(threshold_ref, color=palette[0], linestyle='--', linewidth=0.75)

                # Set x-axis limits correct:
                xmin = times[time_inds[0]]
                xmax = times[time_inds[-1]]

                ax.set_xlim(xmin, xmax)

                # Set correct x- and y-ticks
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax.tick_params(axis='both', which='major', labelsize=font_ticklabels)
                ax.set_ylabel('FWI', fontsize=font_axes)

                plt.suptitle('Longitude: %.2f\N{DEGREE SIGN}; Latitude: %.2f\N{DEGREE SIGN}' %
                             (lons[longitude], lats[latitude]), fontsize=font_title)

                # Create legend:
                plt.plot([], [], color=palette[0], label='FWI$_{ref}$')
                plt.plot([], [], color=palette[1], label="EXP1")
                plt.plot([], [], color=palette[2], label="EXP2")
                plt.plot([], [], color=palette[2], label="EXP2b", linestyle='--')
                plt.plot([], [], color="tab:brown", label="EXP3")
                plt.plot([], [], color="tab:brown", label="EXP3b", linestyle='--')
                plt.plot([], [], color=palette[4], label="EXP4")

                ax.legend(bbox_to_anchor=(0.9, -0.2), ncol=7, fontsize=font_legend)
                fig.subplots_adjust(top=0.9, bottom=0.11, left=0.10, right=0.970, hspace=0.2, wspace=0.2)
                plt.savefig(os.path.join(path_figs, 'TS/' + peatland_type + '_' +
                                         str(locations[i, 0]) + '_' + str(locations[i, 1]) + '_' + str(locations[i, 2])
                                         + '.png'))
                plt.close()
