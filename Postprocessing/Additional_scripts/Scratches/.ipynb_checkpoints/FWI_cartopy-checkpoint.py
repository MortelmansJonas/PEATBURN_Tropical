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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import shapefile

# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
font_title = 20
font_subtitle = 16
font_axes = 14
font_ticklabels = 12
font_text = 12
font_legend = 12

# Set domain for maps
ll_lat = -7
ur_lat = 7.4
ll_lon = 95
ur_lon = 120

# create a new colorbar consisting of 2 colors:
cmap = LinearSegmentedColormap.from_list('my_cmap', ['navy', 'darkgoldenrod'], N=256)

# Adjust the hot_r colorbar
spectral_mod = cm.get_cmap('hot_r', 256)
newcolors = spectral_mod(np.linspace(0.3, 1, 5))
newcmp = ListedColormap(newcolors)

# Create a colorbar for the FWI:
colors = ['#8F92FF', '#A3FFA3', '#F8E396', '#FF8F8F']
# Define the value ranges corresponding to each color (Ranges defined for Indonesia)
boundaries = [0, 2, 7, 13, 100000]
cmap2 = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=cmap2.N, clip=True)

# Adjust the seismic colorbar
spectral_mod = cm.get_cmap('seismic', 256)
newcolors2 = spectral_mod(np.linspace(0.4, 1, 8))
seismic_dis = ListedColormap(newcolors2)

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
ds_exp1 = Dataset(os.path.join(path_out, 'FWI_EXP1_weighted.nc'), mode='r')

# Extract the latitude and longitude to plot
lat = ds_ref['lat'][:].data
lon = ds_ref['lon'][:].data

lons, lats = np.meshgrid(lon, lat)

# ---------------------------------------------------------------------------------------------
# LOAD THE FIRE LOCATIONS
# ---------------------------------------------------------------------------------------------
fires_shp = shapefile.Reader(os.path.join(path_fires, 'Fires_H2M.shp'))

# Extract shapefile information
shapes = fires_shp.shapes()
records = fires_shp.records()

# ---------------------------------------------------------------------------------------------
# LOAD THE FIRE DATASETS FROM THE HITS AND MISSES
# ---------------------------------------------------------------------------------------------
# Load the drained file from the hits and misses
fires_dra_file = os.path.join(path_fires, 'Table_Drapixel.csv')

fires_dra = pd.read_csv(fires_dra_file, header=0)
fires_dra['start_date'] = pd.to_datetime(fires_dra['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_dra_data = fires_dra[fires_dra['Drained_I'] >= 0].reset_index(drop=True)

times = pd.date_range('2002-01-01', '2018-12-31', freq='D')

fire_dra_dates2 = pd.DatetimeIndex(fire_dra_data.start_date)
fire_dra_data = fire_dra_data[fire_dra_dates2.year >= 2002].reset_index(drop=True)

# Load the natural file from the hits and misses
fires_nat_file = os.path.join(path_fires, 'Table_Natpixel.csv')

fires_nat = pd.read_csv(fires_nat_file, header=0)
fires_nat['start_date'] = pd.to_datetime(fires_nat['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_nat_data = fires_nat[fires_nat['Drained_I'] >= 0].reset_index(drop=True)

fire_nat_dates2 = pd.DatetimeIndex(fire_nat_data.start_date)
fire_nat_data = fire_nat_data[fire_nat_dates2.year >= 2002].reset_index(drop=True)

'''----------------------------------------------Find days for maps----------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# COMBINE TWO DATASETS
# ---------------------------------------------------------------------------------------------
# append the natural dataset to the drained dataset:
df = pd.concat([fire_dra_data, fire_nat_data], axis=0, ignore_index=True)
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Sum the number of hits and misses per day:
df_days_m2h = {}
df_days_h2m = {}
fires_per_day = np.zeros((len(times)))
i = 0
for day in times:
    # Filter the relevant rows where the day is between the start and end dates
    relevant_rows = df[(df['start_date'] <= day) & (df['end_date'] >= day)]

    # Calculate the daily number of hits and misses:
    df_days_m2h[day] = relevant_rows['M2H_EXP1'].sum()
    df_days_h2m[day] = relevant_rows['H2M_EXP1'].sum()

    fires_per_day[i] = len(relevant_rows)
    i += 1

# Convert it back to a dataframe:
result_days_m2h = pd.DataFrame(list(df_days_m2h.items()), columns=['Date', 'Daily_sum'])
result_days_m2h['Fires_per_day'] = fires_per_day

result_days_h2m = pd.DataFrame(list(df_days_h2m.items()), columns=['Date', 'Daily_sum'])
result_days_h2m['Fires_per_day'] = fires_per_day

# ---------------------------------------------------------------------------------------------
# CHECK IF ENOUGH FIRES ON THOSE DAYS
# ---------------------------------------------------------------------------------------------
# Get the top 20 days:
top_20_days = result_days_m2h.nlargest(20, 'Daily_sum').reset_index(drop=True)
top_20_days["Date"] = pd.to_datetime(top_20_days["Date"])
least_20_days = result_days_h2m.nlargest(20, 'Daily_sum').reset_index(drop=True)
least_20_days["Date"] = pd.to_datetime(least_20_days["Date"])

'''--------------------------------------------------Prepare Maps--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# SPECIFY DATES
# ---------------------------------------------------------------------------------------------
# First determine the dates to plot
date_1 = pd.to_datetime(top_20_days.at[0, 'Date'])
date_2 = pd.to_datetime(least_20_days.at[0, 'Date'])

# Extract the specific days from the shapefile with the fire points
records_day1 = []
records_day2 = []
for record, shape in zip(records, shapes):
    start_date = pd.to_datetime(record['start_date'])
    end_date = pd.to_datetime(record['end_date'])
    if (start_date <= date_1) & (end_date >= date_1):
        records_day1.append(record)
    if (start_date <= date_2) & (end_date >= date_2):
        records_day2.append(record)

# Define the index for the FWI data:
index_day_1 = np.where(times == date_1)[0][0]
index_day_2 = np.where(times == date_2)[0][0]
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

# Loop over the subplots (i and j), the datasets (ds) and the days (index_day):
m = Basemap(projection='mill', llcrnrlat=ll_lat, urcrnrlat=ur_lat, llcrnrlon=ll_lon, urcrnrlon=ur_lon,
            resolution='f', ax=ax)
m.drawcoastlines(linewidth=0.3)
m.drawcountries(linewidth=0.5)
m.drawparallels(parallels, labels=[False, True, True, False], color='grey', linewidth=0.3, fontsize=font_axes)
m.drawmeridians(meridians, labels=[True, False, False, True], color='grey', linewidth=0.3, fontsize=font_axes)

im = m.pcolormesh(lons, lats, np.nanmean(ds_exp1['MERRA2_FWI'][:] - ds_ref['MERRA2_FWI'][:], axis=0) /
                  (np.nanmean(ds_ref['MERRA2_FWI'][:], axis=0)), latlon=True, cmap=seismic_dis)
cbar = plt.colorbar(im, ax=ax, extend='max', orientation='horizontal', shrink=0.5, aspect=50, pad=0.15)
# im.set_clim(0, 13)

# Add titles
ax.set_title('mean(FWI$_{peat}$ - FWI$_{ref}$) / (mean(FWI$_{ref}$))', fontsize=font_subtitle)

# Add statistics:
mstats = 'Avg. = %.2f' % (np.nanmean(ds_exp1['MERRA2_FWI'][:] - ds_ref['MERRA2_FWI'][:]) /
                          np.nanmean(ds_ref['MERRA2_FWI'][:]))
date = str(times[index_day_1]).split()[0]

if np.nanmean(lats) > 40:
    plt.text(1.0, 1.1, mstats, horizontalalignment='right', verticalalignment='bottom',
             transform=ax.transAxes, fontsize=font_text)
else:
    plt.text(0.5, 0.90, mstats, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='center',
             verticalalignment='bottom', transform=ax.transAxes, fontsize=font_text)

plt.text(0.05, 0.05, date, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
         fontsize=font_text)

# Add fires in as points to the plot:
record_set = records_day1

for record, shape in zip(record_set, shapes):
    if record['M2H_EXP1'] == 0 & record['H2M_EXP1'] == 0:
        marker = '.'
        color = 'k'
        size = 10
        lon, lat = shape.points[0]
        x, y = m(lon, lat)
        m.scatter(x, y, c=color, s=size, linewidth=0.1, marker=marker, edgecolor=color)

for record, shape in zip(record_set, shapes):
    if record['M2H_EXP1'] == 1:
        marker = 'o'
        color = 'k'
        size = 10

        lon, lat = shape.points[0]
        x, y = m(lon, lat)
        m.scatter(x, y, c=color, s=size, linewidth=0.1, marker=marker, edgecolor=color)
    if record['H2M_EXP1'] == 1:
        marker = 'o'
        color = 'k'
        size = 10

        lon, lat = shape.points[0]
        x, y = m(lon, lat)
        m.scatter(x, y, c=color, s=size, linewidth=0.1, marker=marker, edgecolor=color)

# Add a, b, c, etc
plt.text(0.05, 0.9, '(a)', bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes, fontsize=font_text)

    # ax[i // 2, i % 2] itterates over the rows and columns:
    # "i // 2" is the integer division of i by 2. It gives the row index of the subplot in a 2x2 grid.
    # "i % 2" is the remainder of the division of i by 2. It gives the column index of the subplot in a 2x2 grid.
    # if i =0 -> ax[0 // 2, 0 % 2] equals ax[0, 0]
    # if i =1 -> ax[1 // 2, 1 % 2] equals ax[0, 1]
    # if i =2 -> ax[0 // 2, 0 % 2] equals ax[1, 0]
    # if i =3 -> ax[1 // 2, 1 % 2] equals ax[1, 1]

plt.subplots_adjust(hspace=0.1, wspace=0.12)
# plt.show()
plt.savefig(os.path.join(path_figs, "FWI_maps_percentage.png"))
plt.close()