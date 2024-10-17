#!/usr/bin/env python
"""

This script is used to combine the *.csv files created in "Format_fire_table_Tim.py" to create one dataset that can
be loaded with all the necessary information at once. This will help later on to not always question which table
needs to be loaded etc.

Tim put all fires in both the 2007 and 2015 tables, but (probably) only extracted the
fires until 2010 (included) of the 2007 table and the fires after 2010 (not included) of the 2015 table. Here,
all fires are extracted accordingly and put together in one table with the correct LUT.

Additionally, the ignition and perimeter tables are combined to ease processing later on. This is done by creating a
new table that contains all information, without having duplicate columns.

"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------------------
# SPECIFY TIER AND CORRESPONDING PATHS
# ---------------------------------------------------------------------------------------------
Tier = 'Genius'

if Tier == 'Hortense':
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/...'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/....'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Figures'

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

# endregion

'''--------------------------------------------------Load dataset--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# LOAD ALL FIRE DATASETS
# ---------------------------------------------------------------------------------------------
# region
Perimeters_2007 = pd.read_csv(os.path.join(path_fires, 'Perimeters_for_Miettinen2007.csv'), header=0)
Perimeters_2015 = pd.read_csv(os.path.join(path_fires, 'Perimeters_for_Miettinen2015.csv'), header=0)
Ignitions_2007 = pd.read_csv(os.path.join(path_fires, 'IgnitionPoint_for_Miettinen2007.csv'), header=0)
Ignitions_2015 = pd.read_csv(os.path.join(path_fires, 'IgnitionPoint_for_Miettinen2015.csv'), header=0)

# endregion

# endregion

'''-------------------------------------------------Create new table-------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# PREPARE THE DATA TO BE FITTED IN A NEW DATAFRAME
# ---------------------------------------------------------------------------------------------
# region
# make a datetime out of the start date:
Perimeters_2007['start_date'] = pd.to_datetime(Perimeters_2007['start_date'])
Perimeters_2015['start_date'] = pd.to_datetime(Perimeters_2015['start_date'])
Ignitions_2007['start_date'] = pd.to_datetime(Ignitions_2007['start_date'])
Ignitions_2015['start_date'] = pd.to_datetime(Ignitions_2015['start_date'])

# Sort all dataframes according to start date:
Perimeters_2007 = Perimeters_2007.sort_values('start_date', ascending=True)
Perimeters_2015 = Perimeters_2015.sort_values('start_date', ascending=True)
Ignitions_2007 = Ignitions_2007.sort_values('start_date', ascending=True)
Ignitions_2015 = Ignitions_2015.sort_values('start_date', ascending=True)

# For 2007, select only those fires that occur before 2011 / for 2015, select only those fires that occur from 2011 on:
Perimeters_2007 = Perimeters_2007[Perimeters_2007['start_date'].dt.year <= 2010].reset_index(drop=True)
Ignitions_2007 = Ignitions_2007[Ignitions_2007['start_date'].dt.year <= 2010].reset_index(drop=True)
Perimeters_2015 = Perimeters_2015[Perimeters_2015['start_date'].dt.year > 2010].reset_index(drop=True)
Ignitions_2015 = Ignitions_2015[Ignitions_2015['start_date'].dt.year > 2010].reset_index(drop=True)
# endregion

# ---------------------------------------------------------------------------------------------
# CREATE NEW DATAFRAME
# ---------------------------------------------------------------------------------------------
# region
# Create an empty dataframe:
df = pd.DataFrame(columns=['fire_ID', 'latitude_P', 'longitude_P', 'latitude_I', 'longitude_I', 'size', 'perimeter',
                           'start_date', 'start_DOY', 'end_date', 'end_DOY', 'duration', 'Water', 'Seasonal_water',
                           'Pristine_PSF', 'Degraded_PSF', 'Tall_shrub', 'Low_shrub', 'Small-holder_area',
                           'Plantation', 'Built-up', 'Clearance', 'Mangrove', 'Dominant_Class_P',
                           'LUT_I', 'Drained_I', 'Drained_fraction', 'Undrained_fraction', 'Peatland_fraction', 'DD_I',
                           'DD_mean', 'DD_stdev', 'DD_min', 'DD_max'])

"""
'fire_ID': unique fire identifier number of the GFA + the year of the fire
'latitude_P': latitude according to the Perimeter file 
'longitude_P': longitude according to the Perimeter file 
'latitude_I': latitude according to the Ignition file 
'longitude_I': longitude according to the Ignition file 
'size': total area burned per fire [km2]
'perimeter': final perimeter [km]
'start_date': ignition date [YYYY-MM-DD]
'start_DOY': ignition day of year
'end_date': extinction date [YYYY-MM-DD]
'end_DOY': extinction day of year
'duration': fire duration [days]
'Water': fraction of permanent water bodies. This class includes fish adn crab farming ponds.
'Seasonal_water': fraction of areas that are inundated part of the year. Typically, either extremely degraded areas or
flood zones of rivers. This class also includes small-holder mining sites.
'Pristine_PSF': fraction of pristine swamp forest (PSF) with no clear signs of human intervention.
'Degraded_PSF': fraction of PSF with clear signs of disturbance (e.g., logging), typically in the form of logging tracks
and canals and/or opened canopy.
'Tall_shrub': fraction of shrub land or secondary forest with an average height above 2 m.
'Low_shrub': fraction of ferns and grass or shrub land with an average height of less than 2 m.
'Small-holder_area': fraction of mosaic of housing, agricultural fields, plantations, gardens, fallow shrubland,
etc. Note that the name of the class refers to the patchy land scape patterns, typical in small-holder dominated
areas, but the actual land tenure of the areas is unknown.
'Plantations': fraction of industrial plantations which are assumed to have been already planted with the plantation
species. Mainly oil palm and pulp wood.
'Built-up': fraction of towns, industrial areas, etc.
'Clearance': fraction of open area with no vegetation, including recently burned areas.
'Mangrove': fraction of areas that were mangrove forests in the satellite image interpretation although they were 
located within the peatland maps used in this study.
'Dominant_Class_P': the dominant LUT class (highest fraction) according to the Perimeter file
'LUT_I': LUT class of the ignition point (Derived from the Ignition file)
'Drained_I': drained (1) or undrained (0), based on LUT_I
'Drained_fraction': fraction of the fire that was on drained areas (All LUT classes except for water, seasonal_water, 
pristine_PSF, and mangrove are considered as drained)
'Undrained_fraction': fraction of the fire that was on undrained areas (sum of water, seasonal_water, pristine_PSF, 
and mangrove)
'DD_Ignition': Dadap drainage density on the ignition point (Ignition file)
'DD_mean': mean Dadap drainage density for the whole fire (Perimeter file)
'DD_stdev': standard deviation of the Dadap drainage density for the whole fire (Perimeter file)
'DD_min': minimum Dadap drainage density for the whole fire (Perimeter file)
'DD_max': maximum Dadap drainage density for the whole fire (Perimeter file)
"""

# endregion

# ----------------------------------------------------------------------------------------------
# FILL THE NEW DATAFRAME
# ---------------------------------------------------------------------------------------------
# region
# First, since multiple fires from different years can have the same fire_ID, this needs to be fixed. This can be
# solved by creating a new index that contains the GFA fire_ID and the year of that fire:
Perimeters_2007['fire_ID'] = Perimeters_2007['fire_ID'].astype(str) + '_' + Perimeters_2007[
    'start_date'].dt.year.astype(str)
Perimeters_2015['fire_ID'] = Perimeters_2015['fire_ID'].astype(str) + '_' + Perimeters_2015[
    'start_date'].dt.year.astype(str)
Ignitions_2007['fire_ID'] = Ignitions_2007['fire_ID'].astype(str) + '_' + Ignitions_2007[
    'start_date'].dt.year.astype(str)
Ignitions_2015['fire_ID'] = Ignitions_2015['fire_ID'].astype(str) + '_' + Ignitions_2015[
    'start_date'].dt.year.astype(str)

# Then, we can get the unique fire_IDs over all 4 files and put them in the new dataframe:
IDs = np.concatenate(([Perimeters_2007['fire_ID'], Perimeters_2015['fire_ID'],
                       Ignitions_2007['fire_ID'], Ignitions_2015['fire_ID']]))
unique_IDs = np.unique(IDs)

# And already put them in the dataframe
df['fire_ID'] = unique_IDs
df.reset_index(drop=True, inplace=True)

# region

# Next, we should loop over all fire_IDs and fill in the different columns:
for index, ID in enumerate(df['fire_ID']):
    print(ID)

    year = int(ID.split("_")[1])

    if year <= 2010:
        df_P = Perimeters_2007
        df_I = Ignitions_2007
    elif year > 2010:
        df_P = Perimeters_2015
        df_I = Ignitions_2015

    # Get from both files the data for that fire:
    fire_perimeter = df_P.loc[df_P['fire_ID'] == ID]
    fire_ignition = df_I.loc[df_I['fire_ID'] == ID]

    fire_perimeter = fire_perimeter.reset_index()
    fire_ignition = fire_ignition.reset_index()

    # First all data from the perimeter file:
    if fire_perimeter.shape[0] > 0:
        df['latitude_P'][index] = fire_perimeter.at[0, 'latitude']
        df['longitude_P'][index] = fire_perimeter.at[0, 'longitude']
        df['size'][index] = fire_perimeter.at[0, 'size']
        df['perimeter'][index] = fire_perimeter.at[0, 'perimeter']
        df['start_date'][index] = fire_perimeter.at[0, 'start_date']
        df['start_DOY'][index] = fire_perimeter.at[0, 'start_DOY']
        df['end_date'][index] = fire_perimeter.at[0, 'end_date']
        df['end_DOY'][index] = fire_perimeter.at[0, 'end_DOY']
        df['duration'][index] = fire_perimeter.at[0, 'duration']
        df['Water'][index] = fire_perimeter.at[0, 'Water']
        df['Seasonal_water'][index] = fire_perimeter.at[0, 'Seasonal_water']
        df['Pristine_PSF'][index] = fire_perimeter.at[0, 'Pristine_PSF']
        df['Degraded_PSF'][index] = fire_perimeter.at[0, 'Degraded_PSF']
        df['Tall_shrub'][index] = fire_perimeter.at[0, 'Tall_shrub']
        df['Low_shrub'][index] = fire_perimeter.at[0, 'Low_shrub']
        df['Small-holder_area'][index] = fire_perimeter.at[0, 'Small-holder_area']
        df['Plantation'][index] = fire_perimeter.at[0, 'Plantation']
        df['Built-up'][index] = fire_perimeter.at[0, 'Built-up']
        df['Clearance'][index] = fire_perimeter.at[0, 'Clearance']
        df['Mangrove'][index] = fire_perimeter.at[0, 'Mangrove']
        df['Dominant_Class_P'][index] = fire_perimeter.at[0, 'Dominant Class']
        df['Drained_fraction'][index] = fire_perimeter.at[0, 'Drained_Fraction']
        df['Undrained_fraction'][index] = fire_perimeter.at[0, 'Undrained_Fraction']
        df['Peatland_fraction'][index] = fire_perimeter.at[0, 'Peatland_fraction']
        df['DD_mean'][index] = fire_perimeter.at[0, 'DD_mean']
        df['DD_stdev'][index] = fire_perimeter.at[0, 'DD_stdev']
        df['DD_min'][index] = fire_perimeter.at[0, 'DD_min']
        df['DD_max'][index] = fire_perimeter.at[0, 'DD_max']

    else:
        print("===== NO PERIMETER FOUND =====")

    # Then the ignition file:
    if fire_ignition.shape[0] > 0:
        df['latitude_I'][index] = fire_ignition.at[0, 'latitude']
        df['longitude_I'][index] = fire_ignition.at[0, 'longitude']
        df['LUT_I'][index] = fire_ignition.at[0, 'LUT_Class']
        df['Drained_I'][index] = fire_ignition.at[0, 'Drained']
        df['DD_I'][index] = fire_ignition.at[0, 'Drainage_Density']
    else:
        print("===== NO IGNITION FOUND =====")

    # Of course, if the dates in both perimeter and ignition file don't match for the given ID, something went wrong:
    if (fire_ignition.shape[0] > 0) & (fire_perimeter.shape[0] > 0):
        if fire_perimeter.at[0, 'start_date'] != fire_ignition.at[0, 'start_date']:
            print('===== WARNING =====')
            print('Start dates for perimeter and ignition file don\'t agree!')
            print('fire_ID = ' + ID)
            print('start_date for perimeter: ' + str(fire_perimeter.at[0, 'start_date']))
            print('start_date for ignition: ' + str(fire_ignition.at[0, 'start_date']))
        elif fire_perimeter.at[0, 'end_date'] != fire_ignition.at[0, 'end_date']:
            print('===== WARNING =====')
            print('End dates for perimeter and ignition file don\'t agree!')
            print('fire_ID = ' + ID)
            print('end_date for perimeter: ' + str(fire_perimeter.at[0, 'end_date']))
            print('end_date for ignition: ' + str(fire_ignition.at[0, 'end_date']))
        elif fire_perimeter.at[0, 'duration'] != fire_ignition.at[0, 'duration']:
            print('===== WARNING =====')
            print('Duration for perimeter and ignition file don\'t agree!')
            print('fire_ID = ' + ID)
            print('duration for perimeter: ' + str(fire_perimeter.at[0, 'duration']))
            print('duration for ignition: ' + str(fire_ignition.at[0, 'duration']))

# endregion

# endregion

# endregion

'''--------------------------------------------------Save new table--------------------------------------------------'''
# region
df.to_csv(os.path.join(path_fires, 'Reformatted_table_Tim.csv'), index=False)
# endregion
