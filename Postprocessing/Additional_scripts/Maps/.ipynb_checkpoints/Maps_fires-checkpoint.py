#!/usr/bin/env python
"""
This script makes maps of the FWI for different days with pastel colors.
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm, Normalize
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import shapefile

# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
font_title = 25
font_subtitle = 20
font_axes = 16
font_ticklabels = 15
font_text = 15
font_legend = 15

# Set domain for maps
ll_lat = -7
ur_lat = 7.4
ll_lon = 95
ur_lon = 120

# Create a colorbar for the FWI:
colors = ['#8F92FF', '#A3FFA3', '#F8E396', '#FF8F8F']
# Define the value ranges corresponding to each color (Ranges defined for Indonesia)
boundaries = [0, 2, 7, 13, 100000]
cmap2 = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=cmap2.N, clip=True)

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
ds_ref = Dataset(os.path.join(path_ref, 'FWI_Ref_weighted.nc'), mode='r')
ds_exp3b = Dataset(os.path.join(path_out, 'FWI_EXP3b_weighted.nc'), mode='r')

# Extract the latitude and longitude to plot
lat = ds_ref['lat'][:].data
lon = ds_ref['lon'][:].data

lons, lats = np.meshgrid(lon, lat)

# ---------------------------------------------------------------------------------------------
# LOAD THE FIRE LOCATIONS
# ---------------------------------------------------------------------------------------------
# fires_shp = shapefile.Reader(os.path.join(path_fires, 'Fires_H2M.shp'))
fires_shp = shapefile.Reader(os.path.join(path_fires, 'Fires.shp'))

# Extract shapefile information
shapes = fires_shp.shapes()
records = fires_shp.records()

# times
years = [pd.to_datetime(record['start_date']).year for record in records]
norm = Normalize(vmin=min(years), vmax=max(years))
colormap = cm.viridis
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

'''--------------------------------------------------Make Maps--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# MAKE THE MAPS
# ---------------------------------------------------------------------------------------------
# Specify figure size and which parallels and meridians to plot
if np.nanmean(lats) > 30:
    fig_aspect_ratio = (0.1 * (np.nanmax(lons) - np.nanmin(lons))) / (
            0.18 * (np.nanmax(lats) - np.nanmin(lats)))
    figsize = (fig_aspect_ratio + 10, 12)
    parallels = np.arange(-80.0, 81, 5.)
    meridians = np.arange(0., 351., 20.)
else:
    figsize = (16, 10)
    parallels = np.arange(-80.0, 81, 5.)
    meridians = np.arange(0., 351., 5.)

# The actual making of a figure, with 2 rows and 2 columns
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=300)

m = Basemap(projection='mill', llcrnrlat=ll_lat, urcrnrlat=ur_lat, llcrnrlon=ll_lon, urcrnrlon=ur_lon,
            resolution='f', ax=ax)
m.drawcoastlines(linewidth=0.3)
m.drawcountries(linewidth=0.5)
m.drawparallels(parallels, labels=[False, True, True, False], color='grey', linewidth=0.3, fontsize=font_axes)
m.drawmeridians(meridians, labels=[True, False, False, True], color='grey', linewidth=0.3, fontsize=font_axes)

for record, shape in zip(records, shapes):
    date = pd.to_datetime(record['start_date'])
    year = date.year

    if year == 2014:
        marker = 'o'
        size = 5
        lon, lat = shape.points[0]
        x, y = m(lon, lat)
        # color = sm.to_rgba(year)
        #
        # m.scatter(x, y, c=[color], s=size, linewidth=0.1, marker=marker, edgecolor='k')
        m.scatter(x, y, c='tab:red', s=size, linewidth=0.1, marker=marker, edgecolor='k')

# cbar = plt.colorbar(sm, label='Year')
plt.subplots_adjust(hspace=0.1, wspace=0.12)
# plt.show()
plt.savefig(os.path.join(path_figs, "Fire_shapefile_2014.png"))
plt.close()