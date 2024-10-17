#!/usr/bin/env python
"""
This script creates climatologies of zbar, FWI and fire occurrence for the different regions
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

# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
font_title = 20
font_subtitle = 16
font_axes = 14
font_ticklabels = 12
font_text = 11
font_legend = 12

palette = sns.color_palette()

# ---------------------------------------------------------------------------------------------
# SPECIFY TIER AND CORRESPONDING PATHS
# ---------------------------------------------------------------------------------------------
Tier = "Hortense"

if Tier == "Hortense":
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/Reference'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/CDF_matched'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures' \
                '/Climatologies'
    path_peatlands = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output' \
                     '/Peatland_maps'

elif Tier == "Genius":
    path_ref = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/FWI/Reference'
    path_out = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/FWI/CDF2'
    path_fires = '/data/leuven/336/vsc33651/projects/'
    path_figs = '/data/leuven/336/vsc33651/projects/PEATBURN/Figures'

else:
    print('Error: Tier can only be Hortense or Genius.')

'''----------------------------------------------Load fire datasets----------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# LOAD FIREDATASETS
# ---------------------------------------------------------------------------------------------
# Fires
fires_file = os.path.join(path_fires,
                          'CompleteTable__peatfireatlasPerimeterIgnitions_peatThresh0.4__2010-2018_DOMAINpeatclsm'
                          '.txt')
fires = pd.read_csv(fires_file, header=0)
fires['start_date'] = pd.to_datetime(fires['start_date'])
fires = fires.drop_duplicates(subset=['fire_ID', 'latitude_I', 'longitude_I', 'latitude_P', 'longitude_P', 'n',
                                      'size', 'start_date', 'end_date'], keep='first')

fire_data = fires[fires.inDOMAIN_peatclsmpeat_I == 1].reset_index()
times = pd.date_range('2010-01-01', '2018-12-31', freq='D')

fire_dates = pd.DatetimeIndex(fire_data.start_date)
fire_data = fire_data[fire_dates.year >= 2010].reset_index()
fire_dates = fire_dates[fire_dates.year >= 2010]

# Lightning
# Peatlands
ds_peatmap = Dataset(os.path.join(path_peatlands, 'PEATLANDS_PEATCLSM.nc'), 'r')  # The peatland mask of PEATCLSM
ds_peatmap2 = Dataset(os.path.join(path_peatlands, 'CWIM3_HWSD_Combined_M09_grid.nc'),
                      'r')  # The HWSD mask with CWIM3 over Canada
peatlands = ds_peatmap2['PEATMAP'][:]
peatlands[peatlands > 0.4] = 1  # Only interested in grid cells with a peat fraction > 0.4
peatlands[peatlands <= 0.4] = np.nan
peat_mask = np.multiply(ds_peatmap['Peatlands'][:], peatlands)

ds_lightning = Dataset(os.path.join('/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Northern'
                                    '', 'wglc_timeseries_30m_daily_M09.nc'))
lightning2 = np.multiply(ds_lightning['density'][:], peat_mask)

# FWI
ds_ref = Dataset(os.path.join(path_ref, 'FWI_MERRA2_Ref_Peatlands.nc'), 'r')
FWI = ds_ref['MERRA2_FWI'][0:3287, :, :]
FWI[np.isnan(FWI)] = 0

# ---------------------------------------------------------------------------------------------
# STRATIFY ALL DATASETS IN SPACE
# ---------------------------------------------------------------------------------------------
lons = ds_ref['lon'][:]
lats = ds_ref['lat'][:]

fires_raster = np.zeros((3287, 189, 2945))
for i in range(0, len(fire_data.latitude_I)):
    time_start = (pd.to_datetime(fire_data['start_date'][i], format='%Y-%m-%d') - pd.to_datetime('2010-01-01')).days
    lat_diffs = abs(lats - fire_data['latitude_I'][i])
    lon_diffs = abs(lons - fire_data['longitude_I'][i])

    lat_inds = np.where(lat_diffs == np.nanmin(lat_diffs))
    lon_inds = np.where(lon_diffs == np.nanmin(lon_diffs))

    fires_raster[time_start, lat_inds, lon_inds] += 1

fires_raster[fires_raster > 1] = 1

times = pd.date_range('2010-01-01', '2018-12-31', freq='D')

# Loop over the different experiments
experiments = ['EXP1', 'EXP2', 'EXP3', 'EXP4']
files = ['FWI_zbar_DC', 'FWI_sfmc_DMC', 'FWI_sfmc_FFMC', 'FWI_zbar_FWI']
for file, experiment in enumerate(experiments):

    # ---------------------------------------------------------------------------------------------
    # LOAD DATASETS
    # ---------------------------------------------------------------------------------------------
    # zbar
    ds_src = Dataset(os.path.join(path_out, files[file] + '_gridcells.nc'), 'r')

    if experiment == 'EXP4':
        zbar = ds_src['zbar'][0:3287, :, :]
        zbar[np.isnan(zbar)] = 0
        zbar_avg = np.nanmean(zbar, axis=(1, 2))
    else:
        zbar = ds_src['MERRA2_FWI'][0:3287, :, :]
        zbar[np.isnan(zbar)] = 0
        zbar_avg = np.nanmean(zbar, axis=(1, 2))

    FWI_avg = np.nanmean(FWI, axis=(1, 2))
    fires = np.nanmean(fires_raster[:], axis=(1, 2))
    lightning = np.nanmean(lightning2[0:3287, :], axis=(1, 2))

    # Climatology:
    ## fires
    df_fires = pd.DataFrame({'Date': times, 'fires': fires})
    df_fires.set_index('Date', inplace=True)
    ser_fires = df_fires.squeeze()
    clim_fires = calc_climatology(ser_fires, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## zbar
    df_zbar = pd.DataFrame({'Date': times, 'zbar': zbar_avg})
    df_zbar.set_index('Date', inplace=True)
    ser_zbar = df_zbar.squeeze()
    clim_zbar = calc_climatology(ser_zbar, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## FWI
    df_FWI = pd.DataFrame({'Date': times, 'FWI': FWI_avg})
    df_FWI.set_index('Date', inplace=True)
    ser_FWI = df_FWI.squeeze()
    clim_FWI = calc_climatology(ser_FWI, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## Lightning
    df_lightning = pd.DataFrame({'Date': times, 'lightning': lightning})
    df_lightning.set_index('Date', inplace=True)
    ser_lightning = df_lightning.squeeze()
    clim_lightning = calc_climatology(ser_lightning, respect_leap_years=True, interpolate_leapday=True,
                                      fillna=False)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    # Plot the zbar
    p1, = ax1.plot(clim_zbar, color='blue', linewidth=1, label='FWI$_{peat}$')
    ax1.set_ylabel("FWI")
    # ax1.yaxis.label.set_color(p1.get_color())
    ax1.yaxis.label.set_fontsize(font_axes)
    # ax1.tick_params(axis="y", colors=p1.get_color(), labelsize=font_ticklabels)
    ax1.tick_params(axis="y", labelsize=font_ticklabels)

    ax1.set_xlabel("DOY")

    # Plot FWI
    p2, = ax1.plot(clim_FWI, color='orange', linewidth=1, label='FWI$_{ref}$')
    ax1.grid(False)  # Turn off grid #2
    ax1.set_ylabel("FWI [-]")
    # ax1.yaxis.label.set_color(p2.get_color())
    ax1.yaxis.label.set_fontsize(font_axes)
    # ax1.tick_params(axis="y", colors=p2.get_color(), labelsize=font_ticklabels)

    # Plot fire occurrence
    ax3 = ax1.twinx()
    ## Offset the right spine of ax3
    # ax3.spines["right"].set_position(("axes", 1.12))
    ## plot the line
    p3, = ax3.plot(clim_fires, color='red', linewidth=1, label='Fires')
    ax3.grid(False)  # turn off grid #3
    ax3.set_ylabel("# of fires [-]")
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax3.yaxis.label.set_color(p3.get_color())
    ax3.yaxis.label.set_fontsize(font_axes)
    ax3.tick_params(axis='y', colors=p3.get_color(), labelsize=font_ticklabels)

    # Plot T
    ax4 = ax1.twinx()
    ## Offset the right spine of ax3
    ax4.spines["right"].set_position(("axes", 1.12))
    ## plot the line
    p4, = ax4.plot(clim_lightning, color='green', linewidth=1, label='lightning')
    ax4.grid(False)  # turn off grid #4
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax4.set_ylabel("Lightning density (# per day per km2")
    ax4.yaxis.label.set_color(p4.get_color())
    ax4.yaxis.label.set_fontsize(font_axes)
    ax4.tick_params(axis='y', colors=p4.get_color(), labelsize=font_ticklabels)

    plt.legend(handles=[p1, p2, p3, p4], fontsize=font_legend)
    fig.subplots_adjust(top=0.9, bottom=0.11, left=0.10, right=0.850, hspace=0.2, wspace=0.2)
    plt.title(experiment, fontsize=font_title)
    plt.savefig(os.path.join(path_figs, 'Climatology_boreal_' + experiment))
    plt.close()