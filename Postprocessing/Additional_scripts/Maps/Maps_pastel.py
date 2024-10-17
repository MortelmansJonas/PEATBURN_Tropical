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

# # Create a colorbar for the FWI:
# colors = ['#8F92FF', '#A3FFA3', '#F8E396', '#FF8F8F']
# # Define the value ranges corresponding to each color (Ranges defined for Indonesia)
# boundaries = [0, 2, 7, 13, 100]
# cmap2 = ListedColormap(colors)
# norm = BoundaryNorm(boundaries, ncolors=cmap2.N, clip=True)

# Create a continuous colorbar for the FWI:
colors = ['#8F92FF', '#A3FFA3', '#F8E396', '#FF8F8F']
# Define the value ranges corresponding to each color (Ranges defined for Indonesia)
# boundaries = [1, 4.5, 9.5, 13, 100]
fractions = [0.0, 0.3461538461538, 0.7307692307692, 1.0]
colors1 = [plt.cm.colors.hex2color(color) for color in colors]
cmap2 = LinearSegmentedColormap.from_list("my_cmap", list(zip(fractions, colors1)))
# cmap2 = ListedColormap(colors)
# norm = BoundaryNorm(boundaries, ncolors=cmap2.N, clip=True)

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
    df_days_m2h[day] = relevant_rows['M2H_EXP3b'].sum()
    df_days_h2m[day] = relevant_rows['H2M_EXP3b'].sum()

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
# Get the top 20 days with most misses to hits:
top_20_days = result_days_m2h.nlargest(20, 'Daily_sum').reset_index(drop=True)
top_20_days["Date"] = pd.to_datetime(top_20_days["Date"])
# Get the top 20 days with most hits to misses:
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
        print(record['fire_ID'])
    if (start_date <= date_2) & (end_date >= date_2):
        records_day2.append(record)

    if record['fire_ID'] == '857700_2014':
        print(start_date)
        print(end_date)
        print(record['latitude_I'])
        print(record['longitud_1'])
        print('hold')


# start_days = [pd.to_datetime(record['start_date']) for record in records_day1]
# print(start_days)
# end_days = [pd.to_datetime(record['end_date']) for record in records_day1]
# print(end_days)
# IDs = [pd.to_datetime(record['fire_ID']) for record in records_day1]
# print(IDs)

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
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=300)

# Loop over the subplots (i and j), the datasets (ds) and the days (index_day):
for i, j, ds, index_day in [(0, 0, ds_ref, index_day_1), (0, 1, ds_exp3b, index_day_1),
                            (1, 0, ds_ref, index_day_2), (1, 1, ds_exp3b, index_day_2)]:

    m = Basemap(projection='mill', llcrnrlat=ll_lat, urcrnrlat=ur_lat, llcrnrlon=ll_lon, urcrnrlon=ur_lon,
                resolution='f', ax=ax[i, j])
    m.drawcoastlines(linewidth=0.3)
    m.drawcountries(linewidth=0.5)
    m.drawparallels(parallels, labels=[False, True, True, False], color='grey', linewidth=0.3, fontsize=font_axes)
    m.drawmeridians(meridians, labels=[True, False, False, True], color='grey', linewidth=0.3, fontsize=font_axes)

    im = m.pcolormesh(lons, lats, ds['MERRA2_FWI'][index_day, :, :], latlon=True, cmap=cmap2)
    cbar = plt.colorbar(im, ax=ax[i, j], extend='max', orientation='horizontal', shrink=0.5, aspect=50, pad=0.15)
    im.set_clim(0, 14)
    cbar.remove()

    # Add titles
    if i == 0 and j == 0:
        ax[i, j].set_title('FWI$_{ref}$', fontsize=font_subtitle)
    elif i == 0 and j == 1:
        ax[i, j].set_title('FWI$_{peat}$', fontsize=font_subtitle)

    # Add statistics:
    mstats = 'Avg. = %.2f' % (np.nanmean(ds['MERRA2_FWI'][index_day, :, :]))
    date = str(times[index_day]).split()[0]

    if np.nanmean(lats) > 40:
        plt.text(1.0, 1.1, mstats, horizontalalignment='right', verticalalignment='bottom',
                 transform=ax[i, j].transAxes, fontsize=font_text)
    else:
        plt.text(0.5, 0.90, mstats, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='center',
                 verticalalignment='bottom', transform=ax[i, j].transAxes, fontsize=font_text)

    plt.text(0.05, 0.05, date, horizontalalignment='left', verticalalignment='bottom', transform=ax[i, j].transAxes,
             fontsize=font_text)

    record_set = []
    # Add fires in as points to the plot:
    if index_day == index_day_1:
        record_set = records_day1
    elif index_day == index_day_2:
        record_set = records_day2

    for record, shape in zip(record_set, shapes):
        marker = 'o'
        color = 'k'
        size = 5

        lon, lat = shape.points[0]
        x, y = m(record['longitud_1'], record['latitude_I'])
        m.scatter(x, y, c='none', s=size, linewidth=0.5, marker=marker, edgecolor=color)


    # # First add all fires that did not change as black dots
    # for record, shape in zip(record_set, shapes):
    #     if record['M2H_EXP3b'] == 0 & record['H2M_EXP3b'] == 0:
    #         marker = 'o'
    #         color = 'k'
    #         size = 10
    #         lon, lat = shape.points[0]
    #         x, y = m(lon, lat)
    #         m.scatter(x, y, c=color, s=size, linewidth=0.1, marker=marker, edgecolor=color)
    #
    # # Then add all fires for which the prediction changed
    # for record, shape in zip(record_set, shapes):
    #     if record['M2H_EXP3b'] == 1:
    #         marker = 'o'
    #         color = 'green'
    #         size = 15
    #
    #         lon, lat = shape.points[0]
    #         x, y = m(lon, lat)
    #         m.scatter(x, y, c=color, s=size, linewidth=0.2, marker=marker, edgecolor='white', zorder=3)
    #     if record['H2M_EXP3b'] == 1:
    #         marker = 'o'
    #         color = 'tab:red'
    #         size = 15
    #
    #         lon, lat = shape.points[0]
    #         x, y = m(lon, lat)
    #         m.scatter(x, y, c=color, s=size, linewidth=0.2, marker=marker, edgecolor='white', zorder=3)

# Add a, b, c, etc
for i, label in enumerate(['(a)', '(b)', '(c)', '(d)']):
    plt.text(0.075, 0.9, label, bbox=dict(facecolor='white', alpha=1.0), horizontalalignment='right',
             verticalalignment='bottom', transform=ax[i // 2, i % 2].transAxes, fontsize=font_text)

    # ax[i // 2, i % 2] itterates over the rows and columns:
    # "i // 2" is the integer division of i by 2. It gives the row index of the subplot in a 2x2 grid.
    # "i % 2" is the remainder of the division of i by 2. It gives the column index of the subplot in a 2x2 grid.
    # if i =0 -> ax[0 // 2, 0 % 2] equals ax[0, 0]
    # if i =1 -> ax[1 // 2, 1 % 2] equals ax[0, 1]
    # if i =2 -> ax[0 // 2, 0 % 2] equals ax[1, 0]
    # if i =3 -> ax[1 // 2, 1 % 2] equals ax[1, 1]

# Create one general colorbar:
cbar = fig.colorbar(im, ax=ax[1, :], orientation='horizontal', panchor=(0.0,1.0), shrink=0.8, aspect=50, pad=0.05)
cbar.set_ticks([0, 2, 7, 13])
# cbar.set_ticklabels(['Low', 'Medium', 'High', 'Extreme'])
# cbar.ax.tick_params(labelsize=font_ticklabels)

# plt.subplots_adjust(hspace=0.1, wspace=0.12)
plt.show()
# plt.savefig(os.path.join(path_figs, "FWI_maps_EXP3b_cbar_viridis.png"))
# plt.close()