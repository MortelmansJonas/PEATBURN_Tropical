#!/usr/bin/env python
"""
This script is used to reformat Tim's master table. His excel file has 4 tabs: "Perimeters_for_Miettinen2007",
"Perimeters_for_Miettinen2015", "IgnitionPoint_for_Miettinen2007", and "IgnitionPoint_for_Miettinen2015".

All tabs have the same structure:
'fire_ID': unique fire identifier number (GFA)
'lat': latitude of the ignition location (GFA)
'lon': longitude of the ignition location (GFA)
'size': total area burned per fire [km2] (GFA)
'perimeter': final perimeter [km] (GFA)
'start_date': ignition date [YYYY-MM-DD] (GFA)
'start_DOY': ignition day of year (GFA)
'end_date': extinction date [YYYY-MM-DD] (GFA)
'end_DOY': extinction day of year (GFA)
'duration': fire duration [days] (GFA)
'expansion': average daily fire expansion [km2/day] (GFA)
'fire_line': average daily fire line length [km] (GFA)
'speed': average speed of the fire [km/day] (GFA)
'direction': dominant direction of spread (numerical), only provided for multiday fires: (0) indicates no data,
(1) north, (2) northeast, (3) east, (4) southeast, (5) south, (6) southwest, (7) west, and (8) northwest (GFA)
'direction_': dominant direction of spread (string), only provided for multiday fires: (0) indicates no data,
(1) north, (2) northeast, (3) east, (4) southeast, (5) south, (6) southwest, (7) west, and (8) northwest (GFA)
'landcover': dominant land cover type (numerical). The dominant land cover type was derived from MODIS MCD12Q1
collection 5.1 data for 2012 using the University of Maryland (UMD) classification (Friedl et al., 2002) (GFA)
'landcover_': dominant land cover type (string). The dominant land cover type was derived from MODIS MCD12Q1
collection 5.1 data for 2012 using the University of Maryland (UMD) classification (Friedl et al., 2002) (GFA)
'tile_ID': MODIS tile (h, v) (GFA)
'layer': I guess this is just the GFA file that contains that fire

===== The Perimeters files =====
'Row Labels': unique fire identifier (copy of 'fire_ID')
The following LUT classes are those from Miettinnen et al. (2016), see Tims thesis. The numbers in these columns show
the fraction of that grid cell (or probably the fire area) that contains this LULC
'LUT_1': Water [Permanent water bodies. This class includes fish adn crab farming ponds.]
'LUT_2': Seasonal Water [Areas that are inundated part of the year. Typically, either extremely degraded areas or
flood zones of rivers. This class also includes small-holder mining sites.]
'LUT_3': Pristine peat swamp forest [Pristine swamp forest (PSF) with no clear signs of human intervention.]
'LUT_4': Degraded PSF [PSF with clear signs of disturbance (e.g., logging), typically in the form of logging tracks
and canals and/or opened canopy.]
'LUT_5': Tall shrub/secondary forest [Shrub land or secondary forest with an average height above 2 m.]
'LUT_6': Ferns/low shrubs [Ferns and grass or shrub land with an average height of less than 2 m.]
'LUT_7': Small-holder area [Mosaic of housing, agricultural fields, plantations, gardens, fallow shrubland,
etc. Note that the name of the class refers to the patchy land scape patterns, typical in small-holder dominated
areas, but the actual land tenure of the areas is unknown.]
'LUT_8': Industrial plantations [Industrial plantations are assumed to have been already planted with the plantation
species. Mainly oil palm and pulp wood.]
'LUT_9': Built-up area [Towns, industrial areas, etc.]
'LUT_10': Clearance [Open area with no vegetation, including recently burned areas.]
'LUT_11': Mangrove [Areas that were mangrove forests in the satellite image interpretation although they were located
within the peatland maps used in this study.]
'LUT_Fraction_Total': Total fraction of all LUT (should be 1, but due to overlap etc., this can deviate slightly
'Dominant Class': The dominant LUT class (with the highest fraction
'Dominant Class Fraction': The fraction of the dominant LUT (is the same as the corresponding LUT_X column.
'Fire_ID': unique fire identifier (copy of 'fire_ID')
'Undrained_Fraction' or 'Undrained': Whether or not it is drained. This is based on the data of Mietinnen et al. (
2016) and, consequently, the different LUT present in the fire. Only LUT_1, LUT_2, LUT_3, and LUT_11 are considered
to be
undrained. So this column just shows the sum of these 3 columns.
'Drained_Fraction' or 'Drained': All other LUT_ columns are considered to be drained, so this column shows the sum of
these columns.
'Grand Total' or 'Total_Fraction': Again, just the total fraction of all LUT columns (sum). Same as LUT_Fraction_Total
'DD_mean': mean Dadap et al. (2021) drainage density in the fire
'DD_stdev': standard deviation of the Dadap et al. (2021) drainage density in the fire
'DD_min': min Dadap et al. (2021) drainage density in the fire
'DD_max': max Dadap et al. (2021) drainage density in the fire

===== The IgnitionPoint files =====
'LUT_Class': the Mietinnen LUT class in which the fire is ignited (see above for the different classes)
'D_or_UD': Drained or undrained, based on the LUT_Class. 1 = undrained, 2 = drained.
'Drainage_Density': Drainage density according to Dadap et al. (2021).
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
# REFORMAT THE TABLE
# ---------------------------------------------------------------------------------------------
# region
# Read the Excel file
Table_Tim = pd.read_excel(os.path.join(path_fires, 'Final_PointPerimeter_07_15_Database.xlsx'), sheet_name=None)

# Loop through all tabs in the file
for sheet_name, df in Table_Tim.items():
    # read the sheet into a pandas DataFrame:
    # df = pd.read_excel(Table_Tim, sheet_name=sheet_name)

    # Drop some unnecessary columns:
    if sheet_name == 'Perimeters_for_Miettinen2007':
        cols_to_drop = ['expansion', 'fire_line', 'speed', 'direction', 'direction_', 'landcover', 'landcover_',
                        'tile_ID', 'layer', 'Row Labels', 'Fire_ID']

        # Rename some columns to something more meaningful:
        df.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'LUT_1': 'Water', 'LUT_2': 'Seasonal_water',
                           'LUT_3': 'Pristine_PSF',
                           'LUT_4': 'Degraded_PSF', 'LUT_5': 'Tall_shrub', 'LUT_6': 'Low_shrub',
                           'LUT_7': 'Small-holder_area', 'LUT_8': 'Plantation', 'LUT_9': 'Built-up',
                           'LUT_10': 'Clearance', 'LUT_11': 'Mangrove', 'Grand Total': 'Peatland_fraction'},
                  inplace=True)

        # Rename the dominant class to something more meaningful so we don't always have to look it up:
        df.loc[df['Dominant Class'] == 'LUT_1', 'Dominant_Class'] = 'Water'
        df.loc[df['Dominant Class'] == 'LUT_2', 'Dominant_Class'] = 'Seasonal_water'
        df.loc[df['Dominant Class'] == 'LUT_3', 'Dominant_Class'] = 'Pristine_PSF'
        df.loc[df['Dominant Class'] == 'LUT_4', 'Dominant_Class'] = 'Degraded_PSF'
        df.loc[df['Dominant Class'] == 'LUT_5', 'Dominant_Class'] = 'Tall_shrub'
        df.loc[df['Dominant Class'] == 'LUT_6', 'Dominant_Class'] = 'Low_shrub'
        df.loc[df['Dominant Class'] == 'LUT_7', 'Dominant_Class'] = 'Small-holder_area'
        df.loc[df['Dominant Class'] == 'LUT_8', 'Dominant_Class'] = 'Plantation'
        df.loc[df['Dominant Class'] == 'LUT_9', 'Dominant_Class'] = 'Built-up'
        df.loc[df['Dominant Class'] == 'LUT_10', 'Dominant_Class'] = 'Clearance'
        df.loc[df['Dominant Class'] == 'LUT_11', 'Dominant_Class'] = 'Mangrove'

        df['Peatland_BA'] = df['Peatland_fraction'][:] * df['size'][:]

    elif sheet_name == 'Perimeters_for_Miettinen2015':
        cols_to_drop = ['expansion', 'fire_line', 'speed', 'direction', 'direction_', 'landcover', 'landcover_',
                        'tile_ID', 'layer', 'Row Labels', 'Fire_ID']

        # Rename some columns to something more meaningful:
        df.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'LUT_1': 'Water', 'LUT_2': 'Seasonal_water',
                           'LUT_3': 'Pristine_PSF', 'LUT_4': 'Degraded_PSF', 'LUT_5': 'Tall_shrub',
                           'LUT_6': 'Low_shrub', 'LUT_7': 'Small-holder_area', 'LUT_8': 'Plantation',
                           'LUT_9': 'Built-up', 'LUT_10': 'Clearance', 'LUT_11': 'Mangrove', 'Drained':
                           'Drained_Fraction', 'Undrained': 'Undrained_Fraction', 'Total_Fraction':
                           'Peatland_fraction'}, inplace=True)

        # Rename the dominant class to something more meaningful so we don't always have to look it up:
        df.loc[df['Dominant Class'] == 'LUT_1', 'Dominant_Class'] = 'Water'
        df.loc[df['Dominant Class'] == 'LUT_2', 'Dominant_Class'] = 'Seasonal_water'
        df.loc[df['Dominant Class'] == 'LUT_3', 'Dominant_Class'] = 'Pristine_PSF'
        df.loc[df['Dominant Class'] == 'LUT_4', 'Dominant_Class'] = 'Degraded_PSF'
        df.loc[df['Dominant Class'] == 'LUT_5', 'Dominant_Class'] = 'Tall_shrub'
        df.loc[df['Dominant Class'] == 'LUT_6', 'Dominant_Class'] = 'Low_shrub'
        df.loc[df['Dominant Class'] == 'LUT_7', 'Dominant_Class'] = 'Small-holder_area'
        df.loc[df['Dominant Class'] == 'LUT_8', 'Dominant_Class'] = 'Plantation'
        df.loc[df['Dominant Class'] == 'LUT_9', 'Dominant_Class'] = 'Built-up'
        df.loc[df['Dominant Class'] == 'LUT_10', 'Dominant_Class'] = 'Clearance'
        df.loc[df['Dominant Class'] == 'LUT_11', 'Dominant_Class'] = 'Mangrove'

        df['Peatland_BA'] = df['Peatland_fraction'][:] * df['size'][:]

    elif sheet_name == 'IgnitionPoint_for_Miettinen2007':
        cols_to_drop = ['UniqueID', 'expansion', 'fire_line', 'speed', 'direction', 'direction_', 'landcover',
                        'landcover_', 'tile_ID', 'layer']

        # Rename the LUT_Class to something more meaningful so we don't always have to look it up:
        df.loc[df['LUT_Class'] == 1, 'LUT_Class'] = 'Water'
        df.loc[df['LUT_Class'] == 2, 'LUT_Class'] = 'Seasonal_water'
        df.loc[df['LUT_Class'] == 3, 'LUT_Class'] = 'Pristine_PSF'
        df.loc[df['LUT_Class'] == 4, 'LUT_Class'] = 'Degraded_PSF'
        df.loc[df['LUT_Class'] == 5, 'LUT_Class'] = 'Tall_shrub'
        df.loc[df['LUT_Class'] == 6, 'LUT_Class'] = 'Low_shrub'
        df.loc[df['LUT_Class'] == 7, 'LUT_Class'] = 'Small-holder_area'
        df.loc[df['LUT_Class'] == 8, 'LUT_Class'] = 'Plantation'
        df.loc[df['LUT_Class'] == 9, 'LUT_Class'] = 'Built-up'
        df.loc[df['LUT_Class'] == 10, 'LUT_Class'] = 'Clearance'
        df.loc[df['LUT_Class'] == 11, 'LUT_Class'] = 'Mangrove'

        # Change the binary column D_or_UD to "Drained" and just a binary 1-0 with 1 being drained, 0 being undrained:
        df.rename(columns={'D_or_UD': 'Drained'}, inplace=True)

        df.loc[df['Drained'] == 1, 'Drained'] = 0
        df.loc[df['Drained'] == 2, 'Drained'] = 1

    elif sheet_name == 'IgnitionPoint_for_Miettinen2015':
        cols_to_drop = ['Unique_fire_ID', 'expansion', 'fire_line', 'speed', 'direction', 'direction_', 'landcover',
                        'landcover_', 'tile_ID', 'layer']

        # Rename the LUT_Class to something more meaningful so we don't always have to look it up:
        df.loc[df['LUT_Class'] == 1, 'LUT_Class'] = 'Water'
        df.loc[df['LUT_Class'] == 2, 'LUT_Class'] = 'Seasonal_water'
        df.loc[df['LUT_Class'] == 3, 'LUT_Class'] = 'Pristine_PSF'
        df.loc[df['LUT_Class'] == 4, 'LUT_Class'] = 'Degraded_PSF'
        df.loc[df['LUT_Class'] == 5, 'LUT_Class'] = 'Tall_shrub'
        df.loc[df['LUT_Class'] == 6, 'LUT_Class'] = 'Low_shrub'
        df.loc[df['LUT_Class'] == 7, 'LUT_Class'] = 'Small-holder_area'
        df.loc[df['LUT_Class'] == 8, 'LUT_Class'] = 'Plantation'
        df.loc[df['LUT_Class'] == 9, 'LUT_Class'] = 'Built-up'
        df.loc[df['LUT_Class'] == 10, 'LUT_Class'] = 'Clearance'
        df.loc[df['LUT_Class'] == 11, 'LUT_Class'] = 'Mangrove'

        # Change the binary column D_or_UD to "Drained" and just a binary 1-0 with 1 being drained, 0 being undrained:
        df.rename(columns={'D_or_UD': 'Drained'}, inplace=True)

        df.loc[df['Drained'] == 1, 'Drained'] = 0
        df.loc[df['Drained'] == 2, 'Drained'] = 1

    else:
        print('ERROR: there is an error in the if-statement. The sheet ' + sheet_name + ' is not in the options.')

    df.drop(cols_to_drop, axis=1, inplace=True)

    # Write the DataFrame to a csv file
    csv_file_name = sheet_name + '.csv'
    df.to_csv(os.path.join(path_fires, csv_file_name), index=False)

# endregion

# endregion
