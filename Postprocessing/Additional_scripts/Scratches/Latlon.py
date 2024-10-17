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
Tier = "Hortense"

if Tier == 'Hortense':
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Reference'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/CDF_matched'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures'
    path_peatlands = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATCLSM'
    path_PEATMAP = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATMAP/' \
                   'Miettinen2016-PeatLCC900715/crisp-sea-peatland-land-cover-data/data'

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
df = pd.read_csv(os.path.join(path_fires, 'Reformatted_table_Tim.csv'), header=0)
df.loc[pd.isna(df['latitude_P']), 'latitude_P'] = df['latitude_I']
df.loc[pd.isna(df['longitude_P']), 'longitude_P'] = df['longitude_I']
df.to_csv(os.path.join(path_fires, 'Reformatted_table_latlon.csv'), index=False)

# endregion
