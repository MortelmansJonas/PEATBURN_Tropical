#!/usr/bin/env python
"""
This script is used to create the climatologies of the different FWI calculations and the fire occurrences
"""
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
from matplotlib.pyplot import clim
from python_functions import *
import os
import seaborn2 as sns
from pytesmo_anomaly import calc_climatology, calc_anomaly
from copy import deepcopy as dcopy

# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
from sympy.physics.units import cl

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
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures' \
                '/Climatologies'
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

fire_modes = ['ignitions', 'active_fires']

for mode in fire_modes:
    for peatland_type in peatland_types:
        if peatland_type == 'TN':
            drainage_abb = 'Nat'
            title = 'Undrained'
        elif peatland_type == 'TD':
            drainage_abb = 'Dra'
            title = 'Drained'

        '''----------------------------------------------Load datasets----------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # LOAD DATASETS
        # ---------------------------------------------------------------------------------------------
        ## Fires
        if mode == 'active_fires':
            fires_file = os.path.join(path_fires, 'Table_' + drainage_abb + 'pixel.csv')
            fire_data = pd.read_csv(fires_file, header=0)
            fire_data['start_date'] = pd.to_datetime(fire_data['start_date'])

            times = pd.date_range('2002-01-01', '2018-12-31', freq='D')
            fire_dates = pd.DatetimeIndex(fire_data.start_date)
            fire_data = fire_data[fire_dates.year >= 2002].reset_index(drop=True)
            fire_dates = fire_dates[fire_dates.year >= 2002]

            # We only need the fire occurrences for this, not all other data. So count number of fires per day
            fire_counts = fire_data.groupby('start_date').size().reset_index(name='counts')

        elif mode == 'ignitions':
            fires_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')
            fires = pd.read_csv(fires_file, header=0)
            fires['start_date'] = pd.to_datetime(fires['start_date'])
            # only get those fires that are in our domain and in peatclsm pixels:
            fire_data = fires[fires['Drained_I'] >= 0].reset_index(drop=True)

            times = pd.date_range('2002-01-01', '2018-12-31', freq='D')

            fire_dates2 = pd.DatetimeIndex(fire_data.start_date)
            fire_data = fire_data[fire_dates2.year >= 2002].reset_index(drop=True)  # Check if still necessary
            fire_dates2 = fire_dates2[fire_dates2.year >= 2002]  # Check if still necessary

            if peatland_type == 'TN':
                fire_file = dcopy(fire_data)
                fire_file = fire_file[fire_file['Drained_I'] == 0].reset_index(drop=True)
            elif peatland_type == 'TD':
                fire_file = dcopy(fire_data)
                fire_file = fire_file[fire_file['Drained_I'] == 1].reset_index(drop=True)

            fire_counts = fire_file.groupby('start_date').size().reset_index(name='counts')


        fire_counts['start_date'] = pd.to_datetime(fire_counts['start_date'])
        fire_counts = fire_counts[fire_counts['counts'] > 0].reset_index(drop=True)

        ## FWI_ref
        ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
        FWI_M2 = np.nanmean(ds_ref['MERRA2_FWI'][0:6209, :, :].data, axis=(1, 2))

        # EXP1:
        ds_EXP1 = Dataset(os.path.join(path_out, 'FWI_zbar_DC_' + peatland_type + '_pixel.nc'), 'r')
        FWI_EXP1 = np.nanmean(ds_EXP1['MERRA2_FWI'][0:6209, :, :].data, axis=(1, 2))

        # EXP2:
        ds_EXP2 = Dataset(os.path.join(path_out, 'FWI_sfmc_DMC_' + peatland_type + '_pixel.nc'), 'r')
        FWI_EXP2 = np.nanmean(ds_EXP2['MERRA2_FWI'][0:6209, :, :].data, axis=(1, 2))

        # EXP3:
        ds_EXP3 = Dataset(os.path.join(path_out, 'FWI_sfmc_FFMC_' + peatland_type + '_pixel.nc'), 'r')
        FWI_EXP3 = np.nanmean(ds_EXP3['MERRA2_FWI'][0:6209, :, :].data, axis=(1, 2))

        # EXP4:
        ds_EXP4 = Dataset(os.path.join(path_out, 'FWI_zbar_FWI_' + peatland_type + '_pixel.nc'), 'r')
        FWI_EXP4 = np.nanmean(ds_EXP4['zbar'][0:6209, :, :].data, axis=(1, 2))

        # EXP2b:
        ds_EXP2b = Dataset(os.path.join(path_out, 'FWI_zbar_DMC_' + peatland_type + '_pixel.nc'), 'r')
        FWI_EXP2b = np.nanmean(ds_EXP2b['MERRA2_FWI'][0:6209, :, :].data, axis=(1, 2))

        # EXP3b:
        ds_EXP3b = Dataset(os.path.join(path_out, 'FWI_zbar_FFMC_' + peatland_type + '_pixel.nc'), 'r')
        FWI_EXP3b = np.nanmean(ds_EXP3b['MERRA2_FWI'][0:6209, :, :].data, axis=(1, 2))

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

        ## EXP2b
        df_EXP2b = pd.DataFrame({'Date': times, 'FWI': FWI_EXP2b})
        df_EXP2b.set_index('Date', inplace=True)
        ser_EXP2b = df_EXP2b.squeeze()
        clim_EXP2b = calc_climatology(ser_EXP2b, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## EXP3
        df_EXP3 = pd.DataFrame({'Date': times, 'FWI': FWI_EXP3})
        df_EXP3.set_index('Date', inplace=True)
        ser_EXP3 = df_EXP3.squeeze()
        clim_EXP3 = calc_climatology(ser_EXP3, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## EXP3b
        df_EXP3b = pd.DataFrame({'Date': times, 'FWI': FWI_EXP3b})
        df_EXP3b.set_index('Date', inplace=True)
        ser_EXP3b = df_EXP3b.squeeze()
        clim_EXP3b = calc_climatology(ser_EXP3b, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## EXP4
        df_EXP4 = pd.DataFrame({'Date': times, 'FWI': FWI_EXP4})
        df_EXP4.set_index('Date', inplace=True)
        ser_EXP4 = df_EXP4.squeeze()
        clim_EXP4 = calc_climatology(ser_EXP4, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        '''------------------------------------------Calculate anomaly------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # FOR EACH DATASET, CALCULATE THE ANOMALY
        # ---------------------------------------------------------------------------------------------
        anom_fires = calc_anomaly(ser_fires, climatology=clim_fires)
        anom_ref = calc_anomaly(ser_ref, climatology=clim_ref)
        anom_EXP1 = calc_anomaly(ser_EXP1, climatology=clim_EXP1)
        anom_EXP2 = calc_anomaly(ser_EXP2, climatology=clim_EXP2)
        anom_EXP2b = calc_anomaly(ser_EXP2b, climatology=clim_EXP2b)
        anom_EXP3 = calc_anomaly(ser_EXP3, climatology=clim_EXP3)
        anom_EXP3b = calc_anomaly(ser_EXP3b, climatology=clim_EXP3b)
        anom_EXP4 = calc_anomaly(ser_EXP4, climatology=clim_EXP4)

        '''------------------------------------------Plotting------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # PLOT THE CLIMATOLOGIES
        # ---------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=300)

        ## Fires
        ax2 = ax.twinx()
        p8, = ax2.plot(clim_fires, color='k', linewidth=1, label='Fires')
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
        p4, = ax.plot(clim_EXP2b, color=palette[2], label='FWI$_{EXP2b}$', linewidth=1, linestyle='--')
        p5, = ax.plot(clim_EXP3, color='tab:brown', label='FWI$_{EXP3}$', linewidth=1)
        p6, = ax.plot(clim_EXP3b, color='tab:brown', label='FWI$_{EXP3b}$', linewidth=1, linestyle='--')
        p7, = ax.plot(clim_EXP4, color=palette[4], label='FWI$_{EXP4}$', linewidth=1)

        plt.legend(handles=[p1, p2, p3, p4, p5, p6, p7, p8], loc='upper left', ncol=2, fontsize=font_legend)
        plt.title(title, fontsize=font_title)
        plt.savefig(os.path.join(path_figs, 'Climatologies_' + mode + '_' + title))

        # ---------------------------------------------------------------------------------------------
        # PLOT THE anomalies
        # ---------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=300)

        ## Fires
        ax2 = ax.twinx()
        p8, = ax2.plot(anom_fires, color='k', linewidth=1, label='Fires')
        if mode == 'ignitions':
            ax2.set_ylabel('# of ignitions', fontsize=font_axes)
        elif mode == 'active_fires':
            ax2.set_ylabel('# of active fires', fontsize=font_axes)

        # plot the FWI on the first axis:
        ## Ref
        p1, = ax.plot(anom_ref, color=palette[0], label='FWI$_{ref}$', linewidth=1)
        ax.set_ylabel('FWI', fontsize=font_axes)
        ax.set_xlabel('DOY')

        # The different experiments can be on the same axis
        p2, = ax.plot(anom_EXP1, color=palette[1], label='FWI$_{EXP1}$', linewidth=1)
        p3, = ax.plot(anom_EXP2, color=palette[2], label='FWI$_{EXP2}$', linewidth=1)
        p4, = ax.plot(anom_EXP2b, color=palette[2], label='FWI$_{EXP2b}$', linewidth=1, linestyle='--')
        p5, = ax.plot(anom_EXP3, color='tab:brown', label='FWI$_{EXP3}$', linewidth=1)
        p6, = ax.plot(anom_EXP3b, color='tab:brown', label='FWI$_{EXP3b}$', linewidth=1, linestyle='--')
        p7, = ax.plot(anom_EXP4, color=palette[4], label='FWI$_{EXP4}$', linewidth=1)

        plt.legend(handles=[p1, p2, p3, p4, p5, p6, p7, p8], loc='upper left', ncol=2, fontsize=font_legend)
        plt.title(title, fontsize=font_title)
        plt.show()