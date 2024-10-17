#!/usr/bin/env python
"""
This script is used to create the climatologies of the different FWI calculations and the fire occurrences for the
boreal region.
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
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/Reference'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/CDF_matched'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures' \
                '/Climatologies'
    path_peatlands = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output' \
                     '/Peatland_maps'

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

fire_modes = ['ignitions', 'active_fires']
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

# FWI ref
ds_ref = Dataset(os.path.join(path_ref, 'FWI_MERRA2_Ref_Peatlands.nc'), 'r')
FWI_M2 = ds_ref['MERRA2_FWI'][0:3287, :, :]
FWI_M2[np.isnan(FWI_M2)] = 0
FWI_M2 = np.nanmean(FWI_M2, axis=(1, 2))

ds_zbar_dc_gc = Dataset(os.path.join(path_out, 'FWI_zbar_DC_gridcells.nc'), 'r')
FWI_EXP1 = ds_zbar_dc_gc['MERRA2_FWI'][0:3287, :, :]
FWI_EXP1[np.isnan(FWI_EXP1)] = 0
FWI_EXP1 = np.nanmean(FWI_EXP1, axis=(1, 2))
ds_zbar_dc_gc.close()

ds_sfmc_dmc_gc = Dataset(os.path.join(path_out, 'FWI_sfmc_DMC_gridcells.nc'), 'r')
FWI_EXP2 = ds_sfmc_dmc_gc['MERRA2_FWI'][0:3287, :, :]
FWI_EXP2[np.isnan(FWI_EXP2)] = 0
FWI_EXP2 = np.nanmean(FWI_EXP2, axis=(1, 2))
ds_sfmc_dmc_gc.close()

ds_sfmc_EXP3 = Dataset(os.path.join(path_out, 'FWI_sfmc_FFMC_gridcells.nc'), 'r')
FWI_EXP3 = ds_sfmc_EXP3['MERRA2_FWI'][0:3287, :, :]
FWI_EXP3[np.isnan(FWI_EXP3)] = 0
FWI_EXP3 = np.nanmean(FWI_EXP3, axis=(1, 2))
ds_sfmc_EXP3.close()

ds_zbar_EXP4 = Dataset(os.path.join(path_out, 'FWI_zbar_FWI_gridcells.nc'), 'r')
FWI_EXP4 = ds_zbar_EXP4['zbar'][0:3287, :, :]
FWI_EXP4[np.isnan(FWI_EXP4)] = 0
FWI_EXP4 = np.nanmean(FWI_EXP4, axis=(1, 2))
ds_zbar_EXP4.close()


for mode in fire_modes:
    '''----------------------------------------------Fire dataset----------------------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # GET THE RIGHT FIRE DATASET
    # ---------------------------------------------------------------------------------------------
    ## Fires
    if mode == 'ignitions':
        # We only need the fire ignitions for this, not all other data. So count number of fires per day
        fire_counts = fire_data.groupby('start_date').size().reset_index(name='counts')

    elif mode == 'active_fires':
        ## For this, we need all active fires on a certain day, not only the ignitions. We create an expanded
        # dataframe the same way as done in the hits and misses script.

        # Create a new column 'repeat' based on 'duration
        fire_file = dcopy(fire_data)
        fire_file['repeat'] = fire_file['duration']

        # Create a new DataFrame by repeating rows
        expanded_df = fire_file.loc[fire_file.index.repeat(fire_file['repeat'])]

        # Create an index for each group of ducplicated rows
        expanded_df['index'] = expanded_df.groupby(level=0).cumcount()

        # Modify the "start date" column based on the index
        expanded_df['start_date'] = (expanded_df['start_date'] +
                                     expanded_df['index'].apply(lambda x: pd.DateOffset(days=x)))

        # Drop the "repeat" and "index" columns if not needed
        expanded_df = expanded_df.drop(['repeat', 'index'], axis=1).reset_index(drop=True)

        fire_counts = expanded_df.groupby('start_date').size().reset_index(name='counts')

    fire_counts['start_date'] = pd.to_datetime(fire_counts['start_date'])
    fire_counts = fire_counts[fire_counts['counts'] > 0].reset_index(drop=True)

    '''------------------------------------------Calculate climatology------------------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # FOR EACH DATASET, CALCULATE THE CLIMATOLOGY
    # ---------------------------------------------------------------------------------------------
    ## Fires
    fire_counts.set_index('start_date', inplace=True)
    ser_fires = fire_counts.squeeze()
    clim_fires = calc_climatology(ser_fires, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## FWI_ref
    df_ref = pd.DataFrame({'Date': times, 'FWI': FWI_M2})
    df_ref.set_index('Date', inplace=True)
    ser_ref = df_ref.squeeze()
    clim_ref = calc_climatology(ser_ref, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## EXP1
    df_EXP1 = pd.DataFrame({'Date': times, 'FWI': FWI_EXP1})
    df_EXP1.set_index('Date', inplace=True)
    ser_EXP1 = df_EXP1.squeeze()
    clim_EXP1 = calc_climatology(ser_EXP1, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## EXP2
    df_EXP2 = pd.DataFrame({'Date': times, 'FWI': FWI_EXP2})
    df_EXP2.set_index('Date', inplace=True)
    ser_EXP2 = df_EXP2.squeeze()
    clim_EXP2 = calc_climatology(ser_EXP2, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## EXP3
    df_EXP3 = pd.DataFrame({'Date': times, 'FWI': FWI_EXP3})
    df_EXP3.set_index('Date', inplace=True)
    ser_EXP3 = df_EXP3.squeeze()
    clim_EXP3 = calc_climatology(ser_EXP3, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    ## EXP4
    df_EXP4 = pd.DataFrame({'Date': times, 'FWI': FWI_EXP4})
    df_EXP4.set_index('Date', inplace=True)
    ser_EXP4 = df_EXP4.squeeze()
    clim_EXP4 = calc_climatology(ser_EXP4, respect_leap_years=True, interpolate_leapday=True, fillna=False)

    '''------------------------------------------Plotting------------------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # PLOT THE CLIMATOLOGIES
    # ---------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=300)

    ## Fires
    ax2 = ax.twinx()
    p6, = ax2.plot(clim_fires, color='k', linewidth=1, label='Fires')
    if mode == 'ignitions':
        ax2.set_ylabel('# of ignitions', fontsize=font_axes)
    elif mode == 'active_fires':
        ax2.set_ylabel('# of active fires', fontsize=font_axes)

    # plot the FWI on the first axis:
    ## Ref
    p1, = ax.plot(clim_ref, color=palette[0], label='FWI$_{ref}$', linewidth=1)
    ax.set_ylabel('FWI', fontsize=font_axes)
    ax.set_xlabel('DOY')

    # The different experiments can be on the same axis
    p2, = ax.plot(clim_EXP1, color=palette[1], label='FWI$_{EXP1}$', linewidth=1)
    p3, = ax.plot(clim_EXP2, color=palette[2], label='FWI$_{EXP2}$', linewidth=1)
    p4, = ax.plot(clim_EXP3, color='tab:brown', label='FWI$_{EXP3}$', linewidth=1)
    p5, = ax.plot(clim_EXP4, color=palette[4], label='FWI$_{EXP4}$', linewidth=1)

    plt.legend(handles=[p1, p2, p3, p4, p5, p6], loc='upper left', ncol=2, fontsize=font_legend)
    plt.savefig(os.path.join(path_figs, 'Climatologies_boreal_' + mode))
