#!/usr/bin/env python
"""
This script is used to calculate the weights for the tropical and drained peatlands.
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# LOAD MODULES
# ---------------------------------------------------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd
from tqdm import tqdm
import rasterio

# ---------------------------------------------------------------------------------------------
# SPECIFY TIER AND CORRESPONDING PATHS
# ---------------------------------------------------------------------------------------------
Tier = 'Hortense'

if Tier == 'Hortense':
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Reference'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/CDF_matched'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures'
    path_peatlands = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATCLSM'
    path_PEATMAP = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATMAP' \
                   '/Miettinen2016-PeatLCC900715'

elif Tier == 'Genius':
    path_ref = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_out = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_fires = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_figs = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_peatlands = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'
    path_PEATMAP = '/scratch/leuven/336/vsc33651/PEATBURN/Tropical/Weighted_average'

else:
    print('Error: Tier can only be Breniac, Hortense, or Genius.')

'''-------------------------------------------------Define functions-------------------------------------------------'''

def read_geotiff(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)  # Reading the first band


def extract_lat_lon_tiff(file):
    transform = file.transform
    width = file.width
    height = file.height

    # Calculate minx, miny, maxx, maxy
    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (width, height)

    # Alternatively, you can use the `bounds` property
    minx, miny, maxx, maxy = file.bounds

    # Create lons and lats arrays
    lons = np.linspace(minx, maxx, width)
    lats = np.linspace(miny, maxy, height)

    return lons, lats


'''--------------------------------------------------Load datasets--------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# PEATCLSM OUTPUT
# ---------------------------------------------------------------------------------------------
ds_natural = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TN_tchunk.nc'), 'r')
lons_PEATCLSM = ds_natural['lon'][:]
lats_PEATCLSM = ds_natural['lat'][:]
ds_drained = Dataset(os.path.join(path_peatlands, 'PEATCLSM_TD_tchunk.nc'), 'r')

times = pd.date_range("2002-01-01", "2018-12-31", freq="D")

years = [2007, 2015]
FWI_variables = ['MERRA2_DC', 'MERRA2_DMC', 'MERRA2_FFMC', 'MERRA2_BUI', 'MERRA2_ISI', 'MERRA2_FWI']

# Get the latitude and longitude from the Miettinen data:
lons, lats = extract_lat_lon_tiff(rasterio.open(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84_Dra_compressed.tif')))

lats = lats[::-1]

ds_nat = Dataset(os.path.join(path_ref, 'FWI_masked_Nat_tchunk.nc'), 'r')
ds_dra = Dataset(os.path.join(path_ref, 'FWI_masked_Dra_tchunk.nc'), 'r')

weight_nat = np.full((len(times), len(lats_PEATCLSM), len(lons_PEATCLSM)), np.nan)
weight_dra = np.full((len(times), len(lats_PEATCLSM), len(lons_PEATCLSM)), np.nan)
# Then loop over the 2 Miettinen datasets (years)
for year in years:
    print(year)
    # Load the necessary datasets
    if year == 2007:
        inds_time = np.where(times.year <= 2010)[0]
        Drained = read_geotiff(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84_Dra_compressed.tif'))
        Natural = read_geotiff(os.path.join(path_PEATMAP, 'Miettinen_2007_WGS84_Nat_compressed.tif'))

    elif year == 2015:
        inds_time = np.where(times.year > 2010)[0]
        Drained = read_geotiff(os.path.join(path_PEATMAP, 'Miettinen_2015_WGS84_Dra_compressed.tif'))
        Natural = read_geotiff(os.path.join(path_PEATMAP, 'Miettinen_2015_WGS84_Nat_compressed.tif'))

    for lat in tqdm(range(len(lats_PEATCLSM) - 1)):
        # Get all the Miettinen pixels (lat and lon) that fall within the PEATCLSM pixel
        lat_inds = np.asarray(np.where((lats <= lats_PEATCLSM[lat]) & (lats >= lats_PEATCLSM[lat + 1]))[0])

        if lat_inds.size == 0:
            print('empty lat, skipping')
            continue

        for lon in range(len(lons_PEATCLSM) - 1):
            lon_inds = np.asarray(np.where((lons >= lons_PEATCLSM[lon]) & (lons <= lons_PEATCLSM[lon + 1]))[0])
            print('lat: ' + str(lat) + ', lon: ' + str(lon))

            if lon_inds.size == 0:
                print('empty lon, skipping')
                continue

            # Count the number of pixels where there is data in the natural and drained peatland map.
            # Select data based on latitude and longitude indices:
            Natural_subset = Natural[lat_inds, :]
            Natural_subset = Natural_subset[:, lon_inds]
            Drained_subset = Drained[lat_inds, :]
            Drained_subset = Drained_subset[:, lon_inds]

            # Count the number of pixels:
            pix_nat = np.count_nonzero(Natural_subset[~np.isnan(Natural_subset)])
            pix_dra = np.count_nonzero(Drained_subset[~np.isnan(Drained_subset)])

            if (pix_nat + pix_dra) != 0:
                weight_nat[inds_time, lat, lon] = np.full((len(inds_time)), (pix_nat / (pix_nat + pix_dra)))
                weight_dra[inds_time, lat, lon] = np.full((len(inds_time)), (pix_dra / (pix_nat + pix_dra)))

ds_ref = Dataset(os.path.join(path_ref, 'FWI_Ref_weighted.nc'), mode='a')
ds_exp3b = Dataset(os.path.join(path_out, 'FWI_EXP3b_weighted.nc'), mode='a')

ds_ref['weight_natural'][:] = weight_nat
ds_ref['weight_drained'][:] = weight_dra

ds_exp3b['weight_natural'][:] = weight_nat
ds_exp3b['weight_drained'][:] = weight_dra
ds_ref.close()
ds_exp3b.close()
