#!/usr/bin/env python
"""
This script converts the table containing the fire data to a shapefile which then can be used to plot dots on the map
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import geopandas as gpd
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

elif Tier == "Genius":
    path_ref = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/FWI/Reference'
    path_out = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/FWI/CDF_matched'
    path_fires = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Firedata'
    path_figs = '/data/leuven/336/vsc33651/projects/PEATBURN/Figures'
    path_peat = '/staging/leuven/stg_00024/OUTPUT/jonasm/PEATBURN/Maps'

else:
    print('Error: Tier can only be Hortense or Genius.')
# endregion

'''--------------------------------------------------Load datasets--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------------------------------------------
# Natural fires
fires_nat_file = os.path.join(path_fires, 'Table_Natpixel.csv')

fires_nat = pd.read_csv(fires_nat_file, header=0)
fires_nat['start_date'] = pd.to_datetime(fires_nat['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_nat_data = fires_nat[fires_nat['Drained_I'] >= 0].reset_index(drop=True)

times = pd.date_range('2002-01-01', '2018-12-31', freq='D')

fire_nat_dates = pd.DatetimeIndex(fire_nat_data.start_date)
fire_nat_data = fire_nat_data[fire_nat_dates.year >= 2002].reset_index(drop=True)
fire_nat_dates = fire_nat_dates[fire_nat_dates.year >= 2002]

fire_nat_data['start_date'] = fire_nat_data['start_date'].dt.strftime('%Y-%m-%d')

# Drained fires
fires_dra_file = os.path.join(path_fires, 'Table_Drapixel.csv')

fires_dra = pd.read_csv(fires_dra_file, header=0)
fires_dra['start_date'] = pd.to_datetime(fires_dra['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_dra_data = fires_dra[fires_dra['Drained_I'] >= 0].reset_index(drop=True)

times = pd.date_range('2002-01-01', '2018-12-31', freq='D')

fire_dra_dates = pd.DatetimeIndex(fire_dra_data.start_date)
fire_dra_data = fire_dra_data[fire_dra_dates.year >= 2002].reset_index(drop=True)
fire_dra_dates = fire_dra_dates[fire_dra_dates.year >= 2002]

fire_dra_data['start_date'] = fire_dra_data['start_date'].dt.strftime('%Y-%m-%d')

# Combine the two datasets:
fire_data = pd.concat([fire_dra_data, fire_nat_data], axis=0, ignore_index=True)

gdf = gpd.GeoDataFrame(fire_data, geometry=gpd.points_from_xy(fire_data.longitude_I, fire_data.latitude_I))
gdf.to_file(os.path.join(path_fires, 'Fires_H2M.shp'))
