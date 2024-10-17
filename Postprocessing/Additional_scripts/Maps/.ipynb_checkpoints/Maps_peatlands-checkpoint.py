#!/usr/bin/env python
"""
This script makes maps of the peatland distribution in the SEA domain as well as the annual number of ignitions and
percentage of peat area that burned in each pixel per year.
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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap
import shapefile
from matplotlib.colors import Normalize

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

# create a new colorbar consisting of 2 colors:
cmap = LinearSegmentedColormap.from_list('my_cmap', ['navy', 'darkgoldenrod'], N=256)

# Adjust the hot_r colorbar
spectral_mod = cm.get_cmap('hot_r', 256)
newcolors = spectral_mod(np.linspace(0.3, 0.8, 8))
newcmp = ListedColormap(newcolors)
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

'''--------------------------------------------------Prepare Maps--------------------------------------------------'''
# Load Miettinen maps
sf_2007 = shapefile.Reader(os.path.join(path_PEATMAP, "Peatland_land_cover_2007.shp"))
sf_2015 = shapefile.Reader(os.path.join(path_PEATMAP, "Peatland_land_cover_2015.shp"))

# Extract shapefile information
shapes_2007 = sf_2007.shapes()
records_2007 = sf_2007.records()

# Extract the specific values from the shapefile to know the LUTs
values_2007 = np.array([record['Class'] for record in records_2007]).astype(float)

# Extract shapefile information
shapes_2015 = sf_2015.shapes()
records_2015 = sf_2015.records()

# Extract the specific values from the shapefile to know the LUTs
values_2015 = np.array([record['Class'] for record in records_2015]).astype(float)

# Then mask the LUTs based on drained or natural peatlands
'''
'Drained': All LUT classes except for water, seasonal_water, pristine_PSF, and mangrove are considered as drained
'Undrained': Water, seasonal_water, pristine_PSF, and mangrove
'''

values_2007[values_2007 == 0] = np.nan
values_2007[values_2007 <= 3] = 1
values_2007[(values_2007 > 3) & (values_2007 < 11)] = 2
values_2007[values_2007 == 11] = 1
values_2007[values_2007 > 11] = 2

values_2015[values_2015 == 0] = np.nan
values_2015[values_2015 <= 3] = 1
values_2015[(values_2015 > 3) & (values_2015 < 11)] = 2
values_2015[values_2015 == 11] = 1

# Normalize for later:
norm_2007 = Normalize(vmin=np.nanmin(values_2007), vmax=np.nanmax(values_2007))
norm_2015 = Normalize(vmin=np.nanmin(values_2015), vmax=np.nanmax(values_2015))

# ---------------------------------------------------------------------------------------------
# LOAD THE FIRE LOCATIONS
# ---------------------------------------------------------------------------------------------
# Load in a file to get the latitude and longitude
ds_ref = Dataset(os.path.join(path_ref, 'FWI_Ref_weighted.nc'), mode='r')

# Extract the latitude and longitude to plot
lat = ds_ref['lat'][:].data
lon = ds_ref['lon'][:].data

lons, lats = np.meshgrid(lon, lat)

# load the file with the fire data
fires_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')

fires = pd.read_csv(fires_file, header=0)
fires['start_date'] = pd.to_datetime(fires['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_data = fires[fires['Drained_I'] >= 0].reset_index(drop=True)

times = pd.date_range('2002-01-01', '2018-12-31', freq='D')
years = np.unique(times.year)

fire_dates = pd.DatetimeIndex(fire_data.start_date)
fire_data = fire_data[fire_dates.year >= 2002].reset_index(drop=True)
fire_dates = fire_dates[fire_dates.year >= 2002]

# ---------------------------------------------------------------------------------------------
# TURN FIRE TABLE INTO GRID THAT CAN BE PLOTTED
# ---------------------------------------------------------------------------------------------
ignitions_raster = np.zeros((len(lat), len(lon)))

for i in range(len(fire_data.latitude_I)):
    lat_diffs = abs(lats - fire_data['latitude_I'][i])
    lon_diffs = abs(lons - fire_data['longitude_I'][i])

    lat_inds, lon_inds = np.unravel_index(np.nanargmin(lat_diffs), lat_diffs.shape), np.unravel_index(
        np.nanargmin(lon_diffs), lon_diffs.shape)

    lat_inds = lat_inds[0]
    lon_inds = lon_inds[1]
    ignitions_raster[lat_inds, lon_inds] += 1

annual_ignitions = ignitions_raster / len(years)
annual_ignitions[annual_ignitions == 0] = np.nan
'''--------------------------------------------------Make Maps--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# MAKE THE MAPS
# ---------------------------------------------------------------------------------------------
# Specify figure size and which parallels and meridians to plot
figsize = (16, 6)
parallels = np.arange(-80.0, 81, 5.)
meridians = np.arange(0., 351., 5.)

# The actual making of a figure, with 3 rows and 2 columns
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=300)

# The extent of the maps is defined in the beginning of the script
m = Basemap(projection='mill', llcrnrlat=ll_lat, urcrnrlat=ur_lat,
            llcrnrlon=ll_lon, urcrnrlon=ur_lon, resolution='i', ax=ax[0])

# Plot shapefile polygons
## 2007
sc_2007 = None
for shape, value in zip(shapes_2007, values_2007):
    if shape.points:
        lons_shp, lats_shp = zip(*shape.points)
        x, y = m(lons_shp, lats_shp)
        color = cmap(value - 1)
        polygon = plt.Polygon(np.c_[x, y], edgecolor='none', linewidth=0.1, alpha=0.7)
        polygon.set_facecolor(color)
        ax[0].add_patch(polygon)

        if value == 1:
            polygon_nat_2007 = polygon
        elif value == 2:
            polygon_dra_2007 = polygon

        # update ScalarMappable for legend:
        if sc_2007 is None:
            sc_2007 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_2007)
            sc_2007.set_array([])
            sc_2007.set_clim(np.nanmin(values_2007), np.nanmax(values_2007))

# Fig 1: 2007 The peatland distribution
m.drawcoastlines(linewidth=0.3)
m.drawcountries(linewidth=0.5)

# Draw parallels and meridians with solid lines
m.drawparallels(parallels, labels=[False, True, True, False], color='grey', linewidth=0.3, fontsize=font_axes)
m.drawmeridians(meridians, labels=[True, False, False, True], color='grey', linewidth=0.3, fontsize=font_axes)

ax[0].set_title("2007", fontsize=font_subtitle)  # set title of subfigure

# Add legend
sc_2007.set_array(values_2007)
legend = ax[0].legend(handles=[polygon_nat_2007, polygon_dra_2007], labels=["Undrained", 'Drained'], loc='lower right',
                      fontsize=font_legend)

# The extent of the maps is defined in the beginning of the script
m = Basemap(projection='mill', llcrnrlat=ll_lat, urcrnrlat=ur_lat,
            llcrnrlon=ll_lon, urcrnrlon=ur_lon, resolution='i', ax=ax[1])

# Plot shapefile polygons
## 2015
sc_2015 = None
for shape, value in zip(shapes_2015, values_2015):
    if shape.points:
        lons_shp, lats_shp = zip(*shape.points)
        x, y = m(lons_shp, lats_shp)
        color = cmap(value - 1)
        polygon = plt.Polygon(np.c_[x, y], edgecolor='none', linewidth=0.1, alpha=0.7)
        polygon.set_facecolor(color)
        ax[1].add_patch(polygon)

        if value == 1:
            polygon_nat_2015 = polygon
        elif value == 2:
            polygon_dra_2015 = polygon

        # update ScalarMappable for legend:
        if sc_2015 is None:
            sc_2015 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_2015)
            sc_2015.set_array([])
            sc_2015.set_clim(np.nanmin(values_2015), np.nanmax(values_2015))

# Fig 2: The 2015 peatland distribution
m.drawcoastlines(linewidth=0.3)
m.drawcountries(linewidth=0.5)

# Draw parallels and meridians with solid lines
m.drawparallels(parallels, labels=[False, True, True, False], color='grey', linewidth=0.3, fontsize=font_axes)
m.drawmeridians(meridians, labels=[True, False, False, True], color='grey', linewidth=0.3, fontsize=font_axes)

ax[1].set_title("2015", fontsize=font_subtitle)  # set title of subfigure

# Add legend
sc_2015.set_array(values_2015)
legend = ax[1].legend(handles=[polygon_nat_2015, polygon_dra_2015], labels=["Undrained", 'Drained'], loc='lower right',
                      fontsize=font_legend)

# Add a, b, c, etc to the figures
first = '(a)'
second = '(b)'

plt.text(0.075, 0.9, first, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
         verticalalignment='bottom', transform=ax[0].transAxes, fontsize=font_text)
plt.text(0.075, 0.9, second, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
         verticalalignment='bottom', transform=ax[1].transAxes, fontsize=font_text)

# fig.tight_layout()
plt.savefig(os.path.join(path_figs, "Peatland_distribution.png"))
plt.close()

# Fig 3: the fire data
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=300)

m = Basemap(projection='mill', llcrnrlat=ll_lat, urcrnrlat=ur_lat,
            llcrnrlon=ll_lon, urcrnrlon=ur_lon, resolution='f', ax=ax)
m.drawcoastlines(linewidth=0.3)
m.drawcountries(linewidth=0.5, color='white')
m.drawparallels(parallels, labels=[False, True, True, False], color='grey', linewidth=0.3, fontsize=font_axes)
m.drawmeridians(meridians, labels=[True, False, False, True], color='grey', linewidth=0.3, fontsize=font_axes)

im = m.pcolormesh(lons, lats, annual_ignitions, latlon=True, cmap=newcmp)
cbar = plt.colorbar(im, ax=ax, extend='max', orientation='horizontal', shrink=0.5, aspect=50, pad=0.15)
ax.set_title('Ignitions on peatland [yr$^{-1}$]', fontsize=font_subtitle)  # set title of subfigure
cbar.ax.tick_params(labelsize=font_ticklabels)
cbar.cmap.set_over('b')
im.set_clim(0, 4) # Set the limits of the colorbar
cbar.set_ticks([0, 1, 2, 3, 4])

# add statistics
mstats = 'Avg. = %.2f yr$^{-1}$' % (np.nanmean(annual_ignitions))

if np.nanmean(lats) > 40:
    plt.text(1.0, 1.1, mstats, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
             fontsize=font_text)
else:
    plt.text(0.65, 0.9, mstats, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes, fontsize=font_text)

# Add a, b, c, etc to the figures
third = '(c)'

plt.text(0.075, 0.9, third, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
         verticalalignment='bottom', transform=ax.transAxes, fontsize=font_text)

# fig.tight_layout()
plt.savefig(os.path.join(path_figs, "Fires.png"))
plt.close()
