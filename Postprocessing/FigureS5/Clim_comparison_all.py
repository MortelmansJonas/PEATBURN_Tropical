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

# Define colors
c_ref = palette[0]
c_EXP1 = palette[1]
c_EXP2 = palette[2]
c_EXP2b = palette[2]
c_EXP3 = 'tab:brown'
c_EXP3b = 'tab:brown'
c_EXP4 = palette[4]

# Define linestyles
ls_ref = 'solid'
ls_EXP1 = 'dotted'
ls_EXP2 = 'dashdot'
ls_EXP2b = 'dashed'
ls_EXP3 = (0, (3, 1, 1, 1, 1, 1))
ls_EXP3b = 'dashed'
ls_EXP4 = (0, (5, 1))

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
fires_tropics_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')

fires_tropics = pd.read_csv(fires_tropics_file, header=0)
fires_tropics['start_date'] = pd.to_datetime(fires_tropics['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_data_tropics = fires_tropics[fires_tropics['Drained_I'] >= 0].reset_index(drop=True)

fire_tropics_dates2 = pd.DatetimeIndex(fire_data_tropics.start_date)
fire_data_tropics = fire_data_tropics[fire_tropics_dates2.year >= 2002].reset_index(drop=True)

# Create a new column 'repeat' based on 'duration
fire_file_tropics = dcopy(fire_data_tropics)
fire_file_tropics['repeat'] = fire_file_tropics['duration']

# Create a new DataFrame by repeating rows
expanded_df_tropics = fire_file_tropics.loc[fire_file_tropics.index.repeat(fire_file_tropics['repeat'])]

# Create an index for each group of ducplicated rows
expanded_df_tropics['index'] = expanded_df_tropics.groupby(level=0).cumcount()

# Modify the "start date" column based on the index
expanded_df_tropics['start_date'] = (expanded_df_tropics['start_date'] +
                                    expanded_df_tropics['index'].apply(lambda x: pd.DateOffset(days=x)))

# Drop the "repeat" and "index" columns if not needed
expanded_df_tropics = expanded_df_tropics.drop(['repeat', 'index'], axis=1).reset_index(drop=True)

fire_counts_tropics = expanded_df_tropics.groupby('start_date').size().reset_index(name='counts')

fire_counts_tropics['start_date'] = pd.to_datetime(fire_counts_tropics['start_date'])
fire_counts_tropics = fire_counts_tropics[fire_counts_tropics['counts'] > 0].reset_index(drop=True)

## Load the boreal fire dataset:
fires_boreal_file = os.path.join(path_fires_b,
                                 'CompleteTable__peatfireatlasPerimeterIgnitions_peatThresh0.4__2010'
                                 '-2018_DOMAINpeatclsm.txt')
fires_boreal = pd.read_csv(fires_boreal_file, header=0)
fires_boreal['start_date'] = pd.to_datetime(fires_boreal['start_date'])
fires_boreal = fires_boreal.drop_duplicates(subset=['fire_ID', 'latitude_I', 'longitude_I', 'latitude_P',
                                                    'longitude_P', 'n', 'size', 'start_date', 'end_date'],
                                            keep='first')

fire_data_boreal = fires_boreal[fires_boreal.inDOMAIN_peatclsmpeat_I == 1].reset_index()

# Create a new column 'repeat' based on 'duration
fire_file_boreal = dcopy(fire_data_boreal)
fire_file_boreal['repeat'] = fire_file_boreal['duration']

# Create a new DataFrame by repeating rows
expanded_df_boreal = fire_file_boreal.loc[fire_file_boreal.index.repeat(fire_file_boreal['repeat'])]

# Create an index for each group of ducplicated rows
expanded_df_boreal['index'] = expanded_df_boreal.groupby(level=0).cumcount()

# Modify the "start date" column based on the index
expanded_df_boreal['start_date'] = (expanded_df_boreal['start_date'] +
                                    expanded_df_boreal['index'].apply(lambda x: pd.DateOffset(days=x)))

# Drop the "repeat" and "index" columns if not needed
expanded_df_boreal = expanded_df_boreal.drop(['repeat', 'index'], axis=1).reset_index(drop=True)

fire_counts_boreal = expanded_df_boreal.groupby('start_date').size().reset_index(name='counts')

fire_counts_boreal['start_date'] = pd.to_datetime(fire_counts_boreal['start_date'])
fire_counts_boreal = fire_counts_boreal[fire_counts_boreal['counts'] > 0].reset_index(drop=True)

# ---------------------------------------------------------------------------------------------
# LOAD FWI DATASETS
# ---------------------------------------------------------------------------------------------
## TROPICS
# FWI_ref
ds_ref_trop = Dataset(os.path.join(path_ref, 'FWI_Ref_weighted.nc'), 'r')
FWI_M2_trop = ds_ref_trop['MERRA2_FWI'][0:6209, :, :].data
FWI_M2_trop[np.isnan(FWI_M2_trop)] = 0
FWI_M2_trop = np.nanmean(FWI_M2_trop, axis=(1, 2))

# EXP1:
ds_EXP1_trop = Dataset(os.path.join(path_out, 'FWI_EXP1_weighted.nc'), 'r')
FWI_EXP1_trop = ds_EXP1_trop['MERRA2_FWI'][0:6209, :, :].data
FWI_EXP1_trop[np.isnan(FWI_EXP1_trop)] = 0
FWI_EXP1_trop = np.nanmean(FWI_EXP1_trop, axis=(1, 2))

# EXP2:
ds_EXP2_trop = Dataset(os.path.join(path_out, 'FWI_EXP2_weighted.nc'), 'r')
FWI_EXP2_trop = ds_EXP2_trop['MERRA2_FWI'][0:6209, :, :].data
FWI_EXP2_trop[np.isnan(FWI_EXP2_trop)] = 0
FWI_EXP2_trop = np.nanmean(FWI_EXP2_trop, axis=(1, 2))

# EXP2b:
ds_EXP2b_trop = Dataset(os.path.join(path_out, 'FWI_EXP2b_weighted.nc'), 'r')
FWI_EXP2b_trop = ds_EXP2b_trop['MERRA2_FWI'][0:6209, :, :].data
FWI_EXP2b_trop[np.isnan(FWI_EXP2b_trop)] = 0
FWI_EXP2b_trop = np.nanmean(FWI_EXP2b_trop, axis=(1, 2))

# EXP3:
ds_EXP3_trop = Dataset(os.path.join(path_out, 'FWI_EXP3_weighted.nc'), 'r')
FWI_EXP3_trop = ds_EXP3_trop['MERRA2_FWI'][0:6209, :, :].data
FWI_EXP3_trop[np.isnan(FWI_EXP3_trop)] = 0
FWI_EXP3_trop = np.nanmean(FWI_EXP3_trop, axis=(1, 2))

# EXP3b:
ds_EXP3b_trop = Dataset(os.path.join(path_out, 'FWI_EXP3b_weighted.nc'), 'r')
FWI_EXP3b_trop = ds_EXP3b_trop['MERRA2_FWI'][0:6209, :, :].data
FWI_EXP3b_trop[np.isnan(FWI_EXP3b_trop)] = 0
FWI_EXP3b_trop = np.nanmean(FWI_EXP3b_trop, axis=(1, 2))

# EXP4:
ds_EXP4_trop = Dataset(os.path.join(path_out, 'FWI_EXP4_weighted.nc'), 'r')
FWI_EXP4_trop = ds_EXP4_trop['MERRA2_FWI'][0:6209, :, :].data
FWI_EXP4_trop[np.isnan(FWI_EXP4_trop)] = 0
FWI_EXP4_trop = np.nanmean(FWI_EXP4_trop, axis=(1, 2))

## BOREAL
# FWI_ref
ds_ref_bor = Dataset(os.path.join(path_ref_b, 'FWI_MERRA2_Ref_Peatlands.nc'), 'r')
FWI_M2_bor = ds_ref_bor['MERRA2_FWI'][0:3287, :, :]
FWI_M2_bor[np.isnan(FWI_M2_bor)] = 0
FWI_M2_bor = np.nanmean(FWI_M2_bor, axis=(1, 2))

# EXP1:
ds_EXP1_bor = Dataset(os.path.join(path_out_b, 'FWI_zbar_DC_gridcells.nc'), 'r')
FWI_EXP1_bor = ds_EXP1_bor['MERRA2_FWI'][0:3287, :, :]
FWI_EXP1_bor[np.isnan(FWI_EXP1_bor)] = 0
FWI_EXP1_bor = np.nanmean(FWI_EXP1_bor, axis=(1, 2))

# EXP2:
ds_EXP2_bor = Dataset(os.path.join(path_out_b, 'FWI_sfmc_DMC_gridcells.nc'), 'r')
FWI_EXP2_bor = ds_EXP2_bor['MERRA2_FWI'][0:3287, :, :]
FWI_EXP2_bor[np.isnan(FWI_EXP2_bor)] = 0
FWI_EXP2_bor = np.nanmean(FWI_EXP2_bor, axis=(1, 2))

# EXP3:
ds_EXP3_bor = Dataset(os.path.join(path_out_b, 'FWI_sfmc_FFMC_gridcells.nc'), 'r')
FWI_EXP3_bor = ds_EXP3_bor['MERRA2_FWI'][0:3287, :, :]
FWI_EXP3_bor[np.isnan(FWI_EXP3_bor)] = 0
FWI_EXP3_bor = np.nanmean(FWI_EXP3_bor, axis=(1, 2))

# EXP4:
ds_EXP4_bor = Dataset(os.path.join(path_out_b, 'FWI_zbar_FWI_gridcells.nc'), 'r')
FWI_EXP4_bor = ds_EXP4_bor['zbar'][0:3287, :, :]
FWI_EXP4_bor[np.isnan(FWI_EXP4_bor)] = 0
FWI_EXP4_bor = np.nanmean(FWI_EXP4_bor, axis=(1, 2))

'''------------------------------------------Calculate climatology------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# FOR EACH DATASET, CALCULATE THE CLIMATOLOGY
# ---------------------------------------------------------------------------------------------
## TROPICS
## Fires
fire_counts_tropics.set_index('start_date', inplace=True)
ser_fires_tropics = fire_counts_tropics.squeeze()
clim_fires_tropics = calc_climatology(ser_fires_tropics, respect_leap_years=True, interpolate_leapday=True,
                                      fillna=False)

## FWI_ref
df_ref_trop = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_M2_trop})
df_ref_trop.set_index('Date', inplace=True)
ser_ref_trop = df_ref_trop.squeeze()
clim_ref_trop = calc_climatology(ser_ref_trop, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP1
df_EXP1_trop = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_EXP1_trop})
df_EXP1_trop.set_index('Date', inplace=True)
ser_EXP1_trop = df_EXP1_trop.squeeze()
clim_EXP1_trop = calc_climatology(ser_EXP1_trop, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP2
df_EXP2_trop = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_EXP2_trop})
df_EXP2_trop.set_index('Date', inplace=True)
ser_EXP2_trop = df_EXP2_trop.squeeze()
clim_EXP2_trop = calc_climatology(ser_EXP2_trop, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP2b
df_EXP2b_trop = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_EXP2b_trop})
df_EXP2b_trop.set_index('Date', inplace=True)
ser_EXP2b_trop = df_EXP2b_trop.squeeze()
clim_EXP2b_trop = calc_climatology(ser_EXP2b_trop, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP3
df_EXP3_trop = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_EXP3_trop})
df_EXP3_trop.set_index('Date', inplace=True)
ser_EXP3_trop = df_EXP3_trop.squeeze()
clim_EXP3_trop = calc_climatology(ser_EXP3_trop, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP3b
df_EXP3b_trop = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_EXP3b_trop})
df_EXP3b_trop.set_index('Date', inplace=True)
ser_EXP3b_trop = df_EXP3b_trop.squeeze()
clim_EXP3b_trop = calc_climatology(ser_EXP3b_trop, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP4
df_EXP4_trop = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_EXP4_trop})
df_EXP4_trop.set_index('Date', inplace=True)
ser_EXP4_trop = df_EXP4_trop.squeeze()
clim_EXP4_trop = calc_climatology(ser_EXP4_trop, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## BOREAL
## Fires
fire_counts_boreal.set_index('start_date', inplace=True)
ser_fires_bor = fire_counts_boreal.squeeze()
clim_fires_bor = calc_climatology(ser_fires_bor, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## FWI_ref
df_ref_bor = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_M2_bor})
df_ref_bor.set_index('Date', inplace=True)
ser_ref_bor = df_ref_bor.squeeze()
clim_ref_bor = calc_climatology(ser_ref_bor, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP1
df_EXP1_bor = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_EXP1_bor})
df_EXP1_bor.set_index('Date', inplace=True)
ser_EXP1_bor = df_EXP1_bor.squeeze()
clim_EXP1_bor = calc_climatology(ser_EXP1_bor, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP2
df_EXP2_bor = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_EXP2_bor})
df_EXP2_bor.set_index('Date', inplace=True)
ser_EXP2_bor = df_EXP2_bor.squeeze()
clim_EXP2_bor = calc_climatology(ser_EXP2_bor, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP3
df_EXP3_bor = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_EXP3_bor})
df_EXP3_bor.set_index('Date', inplace=True)
ser_EXP3_bor = df_EXP3_bor.squeeze()
clim_EXP3_bor = calc_climatology(ser_EXP3_bor, respect_leap_years=True, interpolate_leapday=True, fillna=False)

## EXP4
df_EXP4_bor = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_EXP4_bor})
df_EXP4_bor.set_index('Date', inplace=True)
ser_EXP4_bor = df_EXP4_bor.squeeze()
clim_EXP4_bor = calc_climatology(ser_EXP4_bor, respect_leap_years=True, interpolate_leapday=True, fillna=False)

'''------------------------------------------Plotting------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# PLOT THE CLIMATOLOGIES
# ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=300)

### TROPICS
## Fires
ax2 = ax[0].twinx()
p8, = ax2.plot(clim_fires_tropics, color='k', linewidth=1, label='Fires')
ax2.set_ylabel('# of fire occurrences\n[day$^{-1}$]', fontsize=font_axes)

# plot the FWI on the first axis:
## Ref
p1, = ax[0].plot(clim_ref_trop, color=c_ref, ls=ls_ref, label='FWI$_{ref}$', linewidth=1)
ax[0].set_ylabel('FWI [-]', fontsize=font_axes)
ax[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax[0].set_xlim(-10, 450)
ax[0].set_xticks([0, 50, 100, 150, 200, 250, 300, 350])

## EXP1
p2, = ax[0].plot(clim_EXP1_trop, color=c_EXP1, ls=ls_EXP1, label='EXP1', linewidth=1)

## EXP2
p3, = ax[0].plot(clim_EXP2_trop, color=c_EXP2, ls=ls_EXP2, label='EXP2', linewidth=1)

## EXP2b
p4, = ax[0].plot(clim_EXP2b_trop, color=c_EXP2b, ls=ls_EXP2b, label='EXP2b', linewidth=1)

## EXP3
p5, = ax[0].plot(clim_EXP3_trop, color=c_EXP3, ls=ls_EXP3, label='EXP3', linewidth=1)

## EXP3b
p6, = ax[0].plot(clim_EXP3b_trop, color=c_EXP3b, ls=ls_EXP3b, label='EXP3b', linewidth=1)

## EXP4
p7, = ax[0].plot(clim_EXP4_trop, color=c_EXP4, ls=ls_EXP4, label='EXP4', linewidth=1)

ax[0].legend(handles=[p1, p2, p3, p4, p5, p6, p7, p8], loc='upper right', ncol=1, fontsize=font_legend)
ax[0].set_title("Tropical SEA", fontsize=font_subtitle)

### BOREAL
## Fires
ax3 = ax[1].twinx()
p6, = ax3.plot(clim_fires_bor, color='k', linewidth=1, label='Fires')
ax3.set_ylabel('# of fire occurrences\n[day$^{-1}$]', fontsize=font_axes)

# plot the FWI on the first axis:
## Ref
p1, = ax[1].plot(clim_ref_bor, color=palette[0], label='FWI$_{ref}$', linewidth=1)
ax[1].set_ylabel('FWI [-]', fontsize=font_axes)
ax[1].set_xlabel('DOY')
ax[1].set_xlim(-10, 450)
ax[1].set_xticks([0, 50, 100, 150, 200, 250, 300, 350])

# The different experiments can be on the same axis
p2, = ax[1].plot(clim_EXP1_bor, color=c_EXP1, ls=ls_EXP1, label='FWI$_{EXP1}$', linewidth=1)
p3, = ax[1].plot(clim_EXP2_bor, color=c_EXP2, ls=ls_EXP2, label='FWI$_{EXP2}$', linewidth=1)
p4, = ax[1].plot(clim_EXP3_bor, color=c_EXP3, ls=ls_EXP3, label='FWI$_{EXP3}$', linewidth=1)
p5, = ax[1].plot(clim_EXP4_bor, color=c_EXP4, ls=ls_EXP4, label='FWI$_{EXP4}$', linewidth=1)

ax[1].legend(handles=[p1, p2, p3, p4, p5, p6], loc='upper right', ncol=1, fontsize=font_legend)
ax[1].set_title("Boreal", fontsize=font_subtitle)

# Add a, b, c, etc to the figures
plt.text(0.05, 0.85, 'A', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', pad=0.3, alpha=1.0),
         horizontalalignment='right', verticalalignment='bottom', transform=ax[0].transAxes, fontsize=font_text, weight='bold')
plt.text(0.05, 0.85, 'B', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', pad=0.3, alpha=1.0),
         horizontalalignment='right', verticalalignment='bottom', transform=ax[1].transAxes, fontsize=font_text, weight='bold')

plt.show()
# plt.savefig(os.path.join(path_figs, 'Climatologies_comparison_all'))
