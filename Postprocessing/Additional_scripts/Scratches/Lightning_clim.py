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
    print('Error: Tier can only be Hortense or Genius.')

experiments = ['EXP1', 'EXP2', 'EXP3', 'EXP4']
files = ['FWI_zbar_DC', 'FWI_sfmc_DMC', 'FWI_sfmc_FFMC', 'FWI_zbar_FWI']

for file, experiment in enumerate(experiments):

    # ---------------------------------------------------------------------------------------------
    # LOAD DATASETS
    # ---------------------------------------------------------------------------------------------
    # Fires
    fires_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')

    fires = pd.read_csv(fires_file, header=0)
    fires['start_date'] = pd.to_datetime(fires['start_date'])
    # only get those fires that are in our domain and in peatclsm pixels:
    fire_data = fires[fires['Drained_I'] >= 0].reset_index(drop=True)

    times = pd.date_range('2002-01-01', '2018-12-31', freq='D')

    fire_dates = pd.DatetimeIndex(fire_data.start_date)
    fire_data = fire_data[fire_dates.year >= 2010].reset_index(drop=True)
    fire_dates = fire_dates[fire_dates.year >= 2010]

    # Lightning
    # Peatlands
    ds_nat = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TN_masked_Nat_20022018.nc'), 'r')
    ds_dra = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TD_masked_Dra_20022018.nc'), 'r')

    peat_mask = np.where(((~np.isnan(ds_nat['zbar'][2922:, :, :].data)) | (~np.isnan(ds_dra['zbar'][2923, :, :].data))), 1, np.nan)

    ds_lightning = Dataset(os.path.join('/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Tropical'
                                        '', 'wglc_timeseries_30m_daily_IND_20022018.nc'))
    lightning2 = np.multiply(ds_lightning['density'][:], peat_mask)

    # ---------------------------------------------------------------------------------------------
    # STRATIFY ALL DATASETS IN SPACE
    # ---------------------------------------------------------------------------------------------
    lons = ds_nat['lon'][:]
    lats = ds_nat['lat'][:]

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

    fires = np.nanmean(fires_raster[:], axis=(1, 2))
    lightning = np.nanmean(lightning2, axis=(1, 2))

    # Climatology:
    ## fires
    df_fires = pd.DataFrame({'Date': times, 'fires': fires})
    df_fires.set_index('Date', inplace=True)
    ser_fires = df_fires.squeeze()
    clim_fires = calc_climatology(ser_fires, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## Lightning
    df_lightning = pd.DataFrame({'Date': times, 'lightning': lightning})
    df_lightning.set_index('Date', inplace=True)
    ser_lightning = df_lightning.squeeze()
    clim_lightning = calc_climatology(ser_lightning, respect_leap_years=True, interpolate_leapday=True,
                                      fillna=False)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    # Plot the zbar
    p1, = ax1.plot(clim_fires, color='tab:red', linewidth=1)
    ax1.set_ylabel("# of fires [-]")
    ax1.legend(["Fire occurrence"], loc="upper left")
    ax1.yaxis.label.set_color(p1.get_color())
    ax1.yaxis.label.set_fontsize(font_axes)
    ax1.tick_params(axis="y", colors=p1.get_color(), labelsize=font_ticklabels)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax1.set_xlabel("DOY")

    # Plot T
    ax4 = ax1.twinx()
    ## Offset the right spine of ax3
    # ax4.spines["right"].set_position(("axes"))
    ## plot the line
    p4, = ax4.plot(clim_lightning, color='tab:orange', linewidth=1)
    ax4.grid(False)  # turn off grid #4
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax4.set_ylabel("Lightning density (# per day per km2")
    ax4.legend(["Lightning"], loc="upper right")
    ax4.yaxis.label.set_color(p4.get_color())
    ax4.yaxis.label.set_fontsize(font_axes)
    ax4.tick_params(axis='y', colors=p4.get_color(), labelsize=font_ticklabels)

    plt.show()