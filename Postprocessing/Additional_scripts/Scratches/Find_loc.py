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

peatland_types = ['TN', 'TD']
CDF_types = ['domain', 'pixel']
for CDF_type in CDF_types:
    for peatland_type in peatland_types:
        if peatland_type == 'TN':
            drainage_abb = 'Nat'
        elif peatland_type == 'TD':
            drainage_abb = 'Dra'

        print(CDF_type)
        print(peatland_type)

        '''----------------------------------------------Load datasets----------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # LOAD FIRE DATASET
        # ---------------------------------------------------------------------------------------------
        fires_file = os.path.join(path_fires, 'Table_' + drainage_abb + CDF_type + '.csv')
        fire_data = pd.read_csv(fires_file, header=0)
        fire_data['start_date'] = pd.to_datetime(fire_data['start_date'])

        times = pd.date_range('2002-01-01', '2018-12-31', freq='D')
        fire_dates = pd.DatetimeIndex(fire_data.start_date)

        locations = np.where((fire_data["M2H_EXP1"] == 1) & (fire_data["M2H_EXP2"] == 1) &
                             (fire_data["M2H_EXP3"] == 1) & (fire_data["M2H_EXP4"] == 1))[0]

        dates = fire_data.loc[locations, "start_date"]
        latitudes = fire_data.loc[locations, "latitude_I"]
        longitudes = fire_data.loc[locations, "longitude_I"]
        print('hold')

