#!/usr/bin/env python
"""
This script is used for the CDF matching of sfmc to DMC
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# Load Packages
from netCDF4 import Dataset
import os
import matplotlib.pyplot as plt
import seaborn2 as sns
import numpy as np
import pandas as pd
import cartopy.crs as crs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from scipy import ndimage
from mpl_toolkits.basemap import Basemap

# sns.set_theme(style='dark')
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

'''--------------------------------------------------Load datasets--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------------------------------------------
PEATCLSM_variables = ['zbar', 'zbar', 'sfmc', 'sfmc']
FWI_variables = ['DC', 'FWI', 'DMC', 'FFMC']
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

            # The original FWI calculations
            ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')

            # PEATCLSM Tropical Natural output
            ds_src = Dataset(
                os.path.join(path_out, 'CDF_log_' + PEATCLSM_variable + '_MERRA2_' + FWI_variable + '_'
                             + peatland_type + '_' + type + '.nc'), 'r')

            ds_CDF = Dataset(
                os.path.join(path_out, 'CDF_matched_' + PEATCLSM_variable + '_MERRA2_' + FWI_variable + '_'
                             + peatland_type + '_' + type + '.nc'), 'r')

            # check mean and stdev:
            print(PEATCLSM_variable + '_' + FWI_variable + '_' + peatland_type + '_' + type)

            ref = ds_ref['MERRA2_' + FWI_variable][0:6209, :, :].data
            src = ds_src[PEATCLSM_variable][0:6209, :, :].data
            cdf = ds_CDF[PEATCLSM_variable][0:6209, :, :].data

            print(ref[~np.isnan(ref)].shape)
            print(src[~np.isnan(src)].shape)
            print(cdf[~np.isnan(cdf)].shape)

            # Make boxplots:
            data = {
                "FWI": ref[~np.isnan(ref)].flatten(),
                "CDF_log": src[~np.isnan(src)].flatten(),
                "CDF": cdf[~np.isnan(cdf)].flatten()
            }
            df = pd.DataFrame(data)
            df.boxplot()
            plt.title(PEATCLSM_variable + '_' + FWI_variable + '_' + peatland_type + '_' + type)
            plt.show()
