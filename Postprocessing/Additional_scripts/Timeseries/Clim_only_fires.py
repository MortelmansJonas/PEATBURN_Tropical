#!/usr/bin/env python
"""
This script is used to compare the climatologies of the best performing experiments of the tropical and the boreal
region with the climatology of the fire occurrences
"""
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
from python_functions import *
import os
import seaborn2 as sns
from pytesmo_anomaly import calc_climatology
from copy import deepcopy as dcopy

# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
font_title = 20
font_subtitle = 16
font_axes = 14
font_ticklabels = 12
font_text = 12
font_legend = 12

# colorblind proof:
palette = sns.color_palette('colorblind')

# ---------------------------------------------------------------------------------------------
# SPECIFY TIER AND CORRESPONDING PATHS
# ---------------------------------------------------------------------------------------------
Tier = 'Hortense'

if Tier == 'Hortense':
    # Paths to the tropical files
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Reference'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/CDF_matched'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures' \
                '/Climatologies'

    # Paths to the boreal files
    path_ref_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Northern/FWI_ref'
    path_out_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Northern/Output'
    path_fires_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Northern/Fires'
    path_peatlands_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Northern/Peatlands'

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

times_boreal = pd.date_range('2010-01-01', '2018-12-31', freq='D')
times_tropics = pd.date_range('2002-01-01', '2018-12-31', freq='D')

'''----------------------------------------------Load datasets----------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# LOAD FIRE DATASETS
# ---------------------------------------------------------------------------------------------
## Load the tropical fire dataset:
fires_tropics_file = os.path.join(path_fires, 'Table_Drapixel.csv')

fires_tropics = pd.read_csv(fires_tropics_file, header=0)
fires_tropics['start_date'] = pd.to_datetime(fires_tropics['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_data_tropics = fires_tropics[fires_tropics['Drained_I'] >= 0].reset_index(drop=True)

fire_tropics_dates2 = pd.DatetimeIndex(fire_data_tropics.start_date)
fire_data_tropics = fire_data_tropics[fire_tropics_dates2.year >= 2002].reset_index(drop=True)

# We only need the fire occurrences for this, not all other data. So count number of fires per day
fire_counts_tropics = fire_data_tropics.groupby('start_date').size().reset_index(name='counts')

fire_counts_tropics['start_date'] = pd.to_datetime(fire_counts_tropics['start_date'])
fire_counts_tropics = fire_counts_tropics[fire_counts_tropics['counts'] > 0].reset_index(drop=True)

FWI_M2 = fire_data_tropics[["start_date", "FWI_M2"]]
FWI_EXP3b = fire_data_tropics[["start_date", "FWI_EXP3b"]]

fire_counts_tropics.set_index('start_date', inplace=True)
ser_fires_tropics = fire_counts_tropics.squeeze()
clim_fires_tropics = calc_climatology(ser_fires_tropics, respect_leap_years=True, interpolate_leapday=True,
                                      fillna=False)

FWI_M2.set_index('start_date', inplace=True)
ser_FWI_M2 = FWI_M2.squeeze()
clim_FWI_M2 = calc_climatology(ser_FWI_M2, respect_leap_years=True, interpolate_leapday=True,
                                      fillna=False)

FWI_EXP3b.set_index('start_date', inplace=True)
ser_FWI_EXP3b = FWI_EXP3b.squeeze()
clim_FWI_EXP3b = calc_climatology(ser_FWI_EXP3b, respect_leap_years=True, interpolate_leapday=True,
                                      fillna=False)

# ---------------------------------------------------------------------------------------------
# PLOT THE CLIMATOLOGIES
# ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300)

### TROPICS
## Fires
ax2 = ax.twinx()
p3, = ax2.plot(clim_fires_tropics, color='k', linewidth=1, label='Fires')
ax2.set_ylabel('# of active fires', fontsize=font_axes)

# plot the FWI on the first axis:
## Ref
p1, = ax.plot(clim_FWI_M2, color=palette[0], label='FWI$_{ref}$', linewidth=1)
ax.set_ylabel('FWI', fontsize=font_axes)
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.set_xlim(-10, 450)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])

## EXP3b
p2, = ax.plot(clim_FWI_EXP3b, color='tab:brown', label='FWI$_{peat}$', linewidth=1, linestyle='--')

ax.legend(handles=[p1, p2, p3], loc='upper right', ncol=1, fontsize=font_legend)
ax.set_title("Tropics", fontsize=font_subtitle)
plt.show()
