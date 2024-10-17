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
    path_ref_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/Reference'
    path_out_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/CDF_matched'
    path_fires_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output/Fire_data'
    path_peatlands_b = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/Breniac/PEATBURN/output' \
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

times_boreal = pd.date_range('2010-01-01', '2018-12-31', freq='D')
times_tropics = pd.date_range('2002-01-01', '2018-12-31', freq='D')

'''
This script first does the full calculation of the climatologies for the tropical peatlands and then the boreal 
peatlands. The whole script is in a for-loop to loop over the drained and natural tropical peatlands to compare them 
with all boreal peatlands.

There is also the for-loop to compare with fire igntions or active fires
'''

# Define the two peatland types to get the right files in the loops
peatland_types = ['TN', 'TD']
# Create an array to loop over ignitions or active fires.
fire_modes = ['ignitions', 'active_fires']

times_boreal = pd.date_range('2010-01-01', '2018-12-31', freq='D')
times_tropics = pd.date_range('2002-01-01', '2018-12-31', freq='D')

for mode in fire_modes:
    for peatland_type in peatland_types:
        if peatland_type == 'TN':
            drainage_abb = 'Nat'
            title = 'Undrained'
        elif peatland_type == 'TD':
            drainage_abb = 'Dra'
            title = 'Drained'

        '''------------------------------------------TROPICAL PEATLANDS------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # LOAD DATASETS
        # ---------------------------------------------------------------------------------------------
        ## Fires
        if mode == 'active_fires':
            fires_file = os.path.join(path_fires, 'Table_' + drainage_abb + 'pixel.csv')
            fire_data = pd.read_csv(fires_file, header=0)
            fire_data['start_date'] = pd.to_datetime(fire_data['start_date'])

            fire_dates = pd.DatetimeIndex(fire_data.start_date)
            fire_data = fire_data[fire_dates.year >= 2002].reset_index(drop=True)
            fire_dates = fire_dates[fire_dates.year >= 2002]

            # We only need the fire occurrences for this, not all other data. So count number of fires per day
            fire_counts_t = fire_data.groupby('start_date').size().reset_index(name='counts')

        elif mode == 'ignitions':
            fires_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')
            fires = pd.read_csv(fires_file, header=0)
            fires['start_date'] = pd.to_datetime(fires['start_date'])
            # only get those fires that are in our domain and in peatclsm pixels:
            fire_data = fires[fires['Drained_I'] >= 0].reset_index(drop=True)

            fire_dates2 = pd.DatetimeIndex(fire_data.start_date)
            fire_data = fire_data[fire_dates2.year >= 2002].reset_index(drop=True)  # Check if still necessary
            fire_dates2 = fire_dates2[fire_dates2.year >= 2002]  # Check if still necessary

            if peatland_type == 'TN':
                fire_file = dcopy(fire_data)
                fire_file = fire_file[fire_file['Drained_I'] == 0].reset_index(drop=True)
            elif peatland_type == 'TD':
                fire_file = dcopy(fire_data)
                fire_file = fire_file[fire_file['Drained_I'] == 1].reset_index(drop=True)

            fire_counts_t = fire_file.groupby('start_date').size().reset_index(name='counts')

        fire_counts_t['start_date'] = pd.to_datetime(fire_counts_t['start_date'])
        fire_counts_t = fire_counts_t[fire_counts_t['counts'] > 0].reset_index(drop=True)

        ## FWI_ref
        ds_ref = Dataset(os.path.join(path_ref, 'FWI_masked_' + drainage_abb + '.nc'), 'r')
        FWI_M2_t = ds_ref['MERRA2_FWI'][0:6209, :, :].data
        FWI_M2_t[np.isnan(FWI_M2_t)] = 0
        FWI_M2_t = np.nanmean(FWI_M2_t.data, axis=(1, 2))

        # EXP3b:
        ds_EXP3b = Dataset(os.path.join(path_out, 'FWI_zbar_FFMC_' + peatland_type + '_pixel.nc'), 'r')
        FWI_EXP3b = ds_EXP3b['MERRA2_FWI'][0:6209, :, :].data
        FWI_EXP3b[np.isnan(FWI_EXP3b)] = 0
        FWI_EXP3b = np.nanmean(FWI_EXP3b.data, axis=(1, 2))

        # ---------------------------------------------------------------------------------------------
        # FOR EACH DATASET, CALCULATE THE CLIMATOLOGY
        # ---------------------------------------------------------------------------------------------
        ## Fires
        fire_counts_t.set_index('start_date', inplace=True)
        ser_fires_t = fire_counts_t.squeeze()
        clim_fires_t = calc_climatology(ser_fires_t, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## FWI_ref
        df_M2_t = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_M2_t})
        df_M2_t.set_index('Date', inplace=True)
        ser_M2_t = df_M2_t.squeeze()
        clim_M2_t = calc_climatology(ser_M2_t, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## EXP3b
        df_EXP3b = pd.DataFrame({'Date': times_tropics, 'FWI': FWI_EXP3b})
        df_EXP3b.set_index('Date', inplace=True)
        ser_EXP3b = df_EXP3b.squeeze()
        clim_EXP3b = calc_climatology(ser_EXP3b, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        '''------------------------------------------BOREAL PEATLANDS------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # LOAD DATASETS
        # ---------------------------------------------------------------------------------------------
        # Fires
        fires_file = os.path.join(path_fires_b,
                                  'CompleteTable__peatfireatlasPerimeterIgnitions_peatThresh0.4__2010'
                                  '-2018_DOMAINpeatclsm'
                                  '.txt')
        fires = pd.read_csv(fires_file, header=0)
        fires['start_date'] = pd.to_datetime(fires['start_date'])
        fires = fires.drop_duplicates(subset=['fire_ID', 'latitude_I', 'longitude_I', 'latitude_P', 'longitude_P', 'n',
                                              'size', 'start_date', 'end_date'], keep='first')

        fire_data = fires[fires.inDOMAIN_peatclsmpeat_I == 1].reset_index()

        fire_dates = pd.DatetimeIndex(fire_data.start_date)
        fire_data = fire_data[fire_dates.year >= 2010].reset_index()
        fire_dates = fire_dates[fire_dates.year >= 2010]

        if mode == 'ignitions':
            # We only need the fire ignitions for this, not all other data. So count number of fires per day
            fire_counts_b = fire_data.groupby('start_date').size().reset_index(name='counts')

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

            fire_counts_b = expanded_df.groupby('start_date').size().reset_index(name='counts')

        fire_counts_b['start_date'] = pd.to_datetime(fire_counts_b['start_date'])
        fire_counts_b = fire_counts_b[fire_counts_b['counts'] > 0].reset_index(drop=True)

        # FWI ref
        ds_ref = Dataset(os.path.join(path_ref_b, 'FWI_MERRA2_Ref_Peatlands.nc'), 'r')
        FWI_M2_b = ds_ref['MERRA2_FWI'][0:3287, :, :]
        FWI_M2_b[np.isnan(FWI_M2_b)] = 0
        FWI_M2_b = np.nanmean(FWI_M2_b, axis=(1, 2))

        # EXP2
        ds_sfmc_dmc_gc = Dataset(os.path.join(path_out_b, 'FWI_sfmc_DMC_gridcells.nc'), 'r')
        FWI_EXP2 = ds_sfmc_dmc_gc['MERRA2_FWI'][0:3287, :, :]
        FWI_EXP2[np.isnan(FWI_EXP2)] = 0
        FWI_EXP2 = np.nanmean(FWI_EXP2, axis=(1, 2))
        ds_sfmc_dmc_gc.close()

        # EXP4
        ds_zbar_EXP4 = Dataset(os.path.join(path_out_b, 'FWI_zbar_FWI_gridcells.nc'), 'r')
        FWI_EXP4 = ds_zbar_EXP4['zbar'][0:3287, :, :]
        FWI_EXP4[np.isnan(FWI_EXP4)] = 0
        FWI_EXP4 = np.nanmean(FWI_EXP4, axis=(1, 2))
        ds_zbar_EXP4.close()

        # ---------------------------------------------------------------------------------------------
        # FOR EACH DATASET, CALCULATE THE CLIMATOLOGY
        # ---------------------------------------------------------------------------------------------
        ## Fires
        fire_counts_b.set_index('start_date', inplace=True)
        ser_fires_b = fire_counts_b.squeeze()
        clim_fires_b = calc_climatology(ser_fires_b, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## FWI_ref
        df_M2_b = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_M2_b})
        df_M2_b.set_index('Date', inplace=True)
        ser_M2_b = df_M2_b.squeeze()
        clim_M2_b = calc_climatology(ser_M2_b, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## EXP2
        df_EXP2 = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_EXP2})
        df_EXP2.set_index('Date', inplace=True)
        ser_EXP2 = df_EXP2.squeeze()
        clim_EXP2 = calc_climatology(ser_EXP2, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        ## EXP4
        df_EXP4 = pd.DataFrame({'Date': times_boreal, 'FWI': FWI_EXP4})
        df_EXP4.set_index('Date', inplace=True)
        ser_EXP4 = df_EXP4.squeeze()
        clim_EXP4 = calc_climatology(ser_EXP4, respect_leap_years=True, interpolate_leapday=True, fillna=False)

        '''------------------------------------------Plotting------------------------------------------'''
        # ---------------------------------------------------------------------------------------------
        # PLOT THE CLIMATOLOGIES
        # ---------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=300)

        ### TROPICS
        ## Fires
        ax2 = ax[0].twinx()
        p3, = ax2.plot(clim_fires_t, color='k', linewidth=1, label='Fires')
        ax2.set_ylabel('# of active fires', fontsize=font_axes)

        # plot the FWI on the first axis:
        ## Ref
        p1, = ax[0].plot(clim_M2_t, color=palette[0], label='FWI$_{ref}$', linewidth=1)
        ax[0].set_ylabel('FWI', fontsize=font_axes)
        ax[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        ## EXP3b
        p2, = ax[0].plot(clim_EXP3b, color='tab:brown', label='FWI$_{peat}$', linewidth=1, linestyle='--')

        ax[0].legend(handles=[p1, p2, p3], loc='center left', ncol=2, fontsize=font_legend)
        ax[0].set_title("Tropics", fontsize=font_subtitle)

        ### BOREAL
        ## Fires
        ax3 = ax[1].twinx()
        p4, = ax3.plot(clim_fires_b, color='k', linewidth=1, label='Fires')
        ax3.set_ylabel('# of active fires', fontsize=font_axes)

        # plot the FWI on the first axis:
        ## Ref
        p1, = ax[1].plot(clim_M2_b, color=palette[0], label='FWI$_{ref}$', linewidth=1)
        ax[1].set_ylabel('FWI', fontsize=font_axes)
        ax[1].set_xlabel('DOY')

        # The different experiments can be on the same axis
        p2, = ax[1].plot(clim_EXP2, color=palette[2], label='FWI$_{EXP2}$', linewidth=1)
        p3, = ax[1].plot(clim_EXP4, color=palette[4], label='FWI$_{EXP4}$', linewidth=1)

        ax[1].legend(handles=[p1, p2, p3, p4], loc='center left', ncol=2, fontsize=font_legend)
        ax[1].set_title("Boreal", fontsize=font_subtitle)

        # Add a, b, c, etc to the figures
        first = '(a)'
        second = '(b)'

        plt.text(0.05, 0.9, first, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
                 verticalalignment='bottom', transform=ax[0].transAxes, fontsize=font_text)
        plt.text(0.05, 0.9, second, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
                 verticalalignment='bottom', transform=ax[1].transAxes, fontsize=font_text)

        plt.savefig(os.path.join(path_figs, 'Clim_comparison_' + mode + '_' + title))