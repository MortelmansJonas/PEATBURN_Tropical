#!/usr/bin/env python
"""
This script is used to calculate the ROC curve for the different FWI calculations. The stratification in time (early vs. late fires) will not be necessary over the tropics. Instead,
a stratification in drained and undrained peatlands will be more interested. This script can be easily adjusted to
those needs. This is, however, not done yet, as I have no idea how the master table of Tim looks like and how he made
the separation between drained and undrained fires.

EXP1 = the replacement of DC by zbar
EXP2 = the replacement of DC by zbar and DMC by sfmc
EXP3 = the replacement of DC by zbar and DMC and FFMC by sfmc
EXP4 = the direct replacement of FWI by zbar
"""

'''--------------------------------------------------Initialization--------------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# MODULES
# ---------------------------------------------------------------------------------------------
from python_functions_sklearn import Contingency_table, CT_metrics, compute_roc
import os
import seaborn2 as sns
from Functions import Breakpoint
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy

# ---------------------------------------------------------------------------------------------
# SET DETAILS FIGURES
# ---------------------------------------------------------------------------------------------
font_title = 20
font_subtitle = 16
font_axes = 12
font_ticklabels = 10
font_text = 10
font_legend = 10

# colorblind proof:
palette = sns.color_palette('colorblind')

# ---------------------------------------------------------------------------------------------
# SPECIFY TIER AND CORRESPONDING PATHS
# ---------------------------------------------------------------------------------------------
Tier = 'Genius'

if Tier == 'Hortense':
    path_ref = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Reference'
    path_out = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/CDF_matched'
    path_fires = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/Fire_data'
    path_figs = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/output/Figures'
    path_peatlands = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATCLSM'
    path_PEATMAP = '/dodrio/scratch/projects/2022_200/project_output/rsda/vsc33651/PEATBURN/Tropical/PEATMAP' \
                   '/Miettinen2016-PeatLCC900715'

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

peatland_types = ['TN', 'TD']
types = ['pixel']

'''----------------------------------------------Load datasets----------------------------------------------'''
# region
# ---------------------------------------------------------------------------------------------
# LOAD FIRE DATASET
# ---------------------------------------------------------------------------------------------
fires_file = os.path.join(path_fires, 'Reformatted_table_Tim.csv')

fires = pd.read_csv(fires_file, header=0)
fires['start_date'] = pd.to_datetime(fires['start_date'])
# only get those fires that are in our domain and in peatclsm pixels:
fire_data = fires[fires['Drained_I'] >= 0].reset_index(drop=True)

times = pd.date_range('2002-01-01', '2018-12-31', freq='D')

fire_dates = pd.DatetimeIndex(fire_data.start_date)
fire_data = fire_data[fire_dates.year >= 2002].reset_index(drop=True)
fire_dates = fire_dates[fire_dates.year >= 2002]

fires_nat = dcopy(fire_data)
fires_nat = fires_nat[fires_nat['Drained_I'] == 0].reset_index(drop=True)
fire_nat_dates = pd.DatetimeIndex(fires_nat.start_date)
fire_nat_dates = fire_nat_dates[fire_nat_dates.year >= 2002]
fires_dra = dcopy(fire_data)
fires_dra = fires_dra[fires_dra['Drained_I'] == 1].reset_index(drop=True)
fire_dra_dates = pd.DatetimeIndex(fires_dra.start_date)
fire_dra_dates = fire_dra_dates[fire_dra_dates.year >= 2002]

# endregion

'''-----------------------------------------------------For loop-----------------------------------------------------'''
# ---------------------------------------------------------------------------------------------
# FOR LOOP FOR ROC CALCULATIONS AND PLOTTING
# ---------------------------------------------------------------------------------------------
for type in types:  # Loop over domain-wise and pixel-wise CDF matching methods
    # ---------------------------------------------------------------------------------------------
    # LOAD FWI DATASETS
    # ---------------------------------------------------------------------------------------------
    ## Natural
    # Reference:
    ds_ref_nat = Dataset(os.path.join(path_ref, 'FWI_masked_Nat.nc'), 'r')
    FWI_M2_nat = ds_ref_nat['MERRA2_FWI'][0:6209, :, :].data

    # EXP1:
    ds_EXP1_nat = Dataset(os.path.join(path_out, 'FWI_Rescaled_zbar_DC_TN_' + type + '.nc'), 'r')
    FWI_EXP1_nat = ds_EXP1_nat['MERRA2_FWI'][0:6209, :, :].data

    # EXP2:
    ds_EXP2_nat = Dataset(os.path.join(path_out, 'FWI_Rescaled_sfmc_DMC_TN_' + type + '.nc'), 'r')
    FWI_EXP2_nat = ds_EXP2_nat['MERRA2_FWI'][0:6209, :, :].data

    # EXP3:
    ds_EXP3_nat = Dataset(os.path.join(path_out, 'FWI_Rescaled_sfmc_FFMC_TN_' + type + '.nc'), 'r')
    FWI_EXP3_nat = ds_EXP3_nat['MERRA2_FWI'][0:6209, :, :].data

    # EXP4:
    ds_EXP4_nat = Dataset(os.path.join(path_out, 'FWI_Rescaled_zbar_FWI_TN_' + type + '.nc'), 'r')
    FWI_EXP4_nat = ds_EXP4_nat['zbar'][0:6209, :, :].data

    ## Drained
    # Reference:
    ds_ref_dra = Dataset(os.path.join(path_ref, 'FWI_masked_Dra.nc'), 'r')
    FWI_M2_dra = ds_ref_dra['MERRA2_FWI'][0:6209, :, :].data

    # EXP1:
    ds_EXP1_dra = Dataset(os.path.join(path_out, 'FWI_Rescaled_zbar_DC_TD_' + type + '.nc'), 'r')
    FWI_EXP1_dra = ds_EXP1_dra['MERRA2_FWI'][0:6209, :, :].data

    # EXP2:
    ds_EXP2_dra = Dataset(os.path.join(path_out, 'FWI_Rescaled_sfmc_DMC_TD_' + type + '.nc'), 'r')
    FWI_EXP2_dra = ds_EXP2_dra['MERRA2_FWI'][0:6209, :, :].data

    # EXP3:
    ds_EXP3_dra = Dataset(os.path.join(path_out, 'FWI_Rescaled_sfmc_FFMC_TD_' + type + '.nc'), 'r')
    FWI_EXP3_dra = ds_EXP3_dra['MERRA2_FWI'][0:6209, :, :].data

    # EXP4:
    ds_EXP4_dra = Dataset(os.path.join(path_out, 'FWI_Rescaled_zbar_FWI_TD_' + type + '.nc'), 'r')
    FWI_EXP4_dra = ds_EXP4_dra['zbar'][0:6209, :, :].data


    '''-----------------------------------------------Prepare ROC-----------------------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # CREATE RASTER OF FIRE DATASET
    # ---------------------------------------------------------------------------------------------
    lats = ds_ref_nat['lat'][:]
    lons = ds_ref_nat['lon'][:]

    # Determine the dimensions of the dataset:
    dim_time_nat = FWI_M2_nat.shape[0]
    dim_lat_nat = FWI_M2_nat.shape[1]
    dim_lon_nat = FWI_M2_nat.shape[2]

    dim_time_dra = FWI_M2_dra.shape[0]
    dim_lat_dra = FWI_M2_dra.shape[1]
    dim_lon_dra = FWI_M2_dra.shape[2]

    # Create raster of natural fires
    fires_raster_nat = np.zeros((dim_time_nat, dim_lat_nat, dim_lon_nat))
    for i in range(len(fires_nat.latitude_I)):
        time_start = (pd.to_datetime(fires_nat['start_date'][i], format='%Y-%m-%d') - pd.to_datetime(
            '2010-01-01')).days
        lat_diffs = abs(lats - fires_nat['latitude_I'][i])
        lon_diffs = abs(lons - fires_nat['longitude_I'][i])

        lat_inds = np.where(lat_diffs == np.nanmin(lat_diffs))
        lon_inds = np.where(lon_diffs == np.nanmin(lon_diffs))

        fires_raster_nat[time_start, lat_inds, lon_inds] += 1

    fires_raster_nat[fires_raster_nat > 1] = 1  # For the calculation of the ROC curve, binary data is needed

    # Create raster of drained fires
    lat_diffs = []
    lon_diffs = []
    i = 0
    lat_inds = []
    lon_inds = []

    fires_raster_dra = np.zeros((dim_time_dra, dim_lat_dra, dim_lon_dra))
    for i in range(len(fires_dra.latitude_I)):
        time_start = (pd.to_datetime(fires_dra['start_date'][i], format='%Y-%m-%d') - pd.to_datetime(
            '2010-01-01')).days
        lat_diffs = abs(lats - fires_dra['latitude_I'][i])
        lon_diffs = abs(lons - fires_dra['longitude_I'][i])

        lat_inds = np.where(lat_diffs == np.nanmin(lat_diffs))[0][0]
        lon_inds = np.where(lon_diffs == np.nanmin(lon_diffs))[0][0]

        fires_raster_dra[time_start, lat_inds, lon_inds] += 1

    fires_raster_dra[fires_raster_dra > 1] = 1  # For the calculation of the ROC curve, binary data is needed

    # ---------------------------------------------------------------------------------------------
    # PREPARE DATASETS FOR CALCULATION OF ROC
    # ---------------------------------------------------------------------------------------------
    nonan_M2_nat = np.argwhere((~np.isnan(fires_raster_nat)) & (~np.isnan(FWI_M2_nat)))
    nonan_EXP1_nat = np.argwhere((~np.isnan(fires_raster_nat)) & (~np.isnan(FWI_EXP1_nat)))
    nonan_EXP2_nat = np.argwhere((~np.isnan(fires_raster_nat)) & (~np.isnan(FWI_EXP2_nat)))
    nonan_EXP3_nat = np.argwhere((~np.isnan(fires_raster_nat)) & (~np.isnan(FWI_EXP3_nat)))
    nonan_EXP4_nat = np.argwhere((~np.isnan(fires_raster_nat)) & (~np.isnan(FWI_EXP4_nat)))

    nonan_M2_dra = np.argwhere((~np.isnan(fires_raster_dra)) & (~np.isnan(FWI_M2_dra)))
    nonan_EXP1_dra = np.argwhere((~np.isnan(fires_raster_dra)) & (~np.isnan(FWI_EXP1_dra)))
    nonan_EXP2_dra = np.argwhere((~np.isnan(fires_raster_dra)) & (~np.isnan(FWI_EXP2_dra)))
    nonan_EXP3_dra = np.argwhere((~np.isnan(fires_raster_dra)) & (~np.isnan(FWI_EXP3_dra)))
    nonan_EXP4_dra = np.argwhere((~np.isnan(fires_raster_dra)) & (~np.isnan(FWI_EXP4_dra)))

    '''----------------------------------Determine location of 90th percentile----------------------------------'''
    # ---------------------------------------------------------------------------------------------
    # CALCULATE 90th PERCENTILE THRESHOLD
    # ---------------------------------------------------------------------------------------------
    threshold_M2_nat = np.nanquantile(FWI_M2_nat, 0.9)
    threshold_EXP1_nat = np.nanquantile(FWI_EXP1_nat, 0.9)
    threshold_EXP2_nat = np.nanquantile(FWI_EXP2_nat, 0.9)
    threshold_EXP3_nat = np.nanquantile(FWI_EXP3_nat, 0.9)
    threshold_EXP4_nat = np.nanquantile(FWI_EXP4_nat, 0.9)

    threshold_M2_dra = np.nanquantile(FWI_M2_dra, 0.9)
    threshold_EXP1_dra = np.nanquantile(FWI_EXP1_dra, 0.9)
    threshold_EXP2_dra = np.nanquantile(FWI_EXP2_dra, 0.9)
    threshold_EXP3_dra = np.nanquantile(FWI_EXP3_dra, 0.9)
    threshold_EXP4_dra = np.nanquantile(FWI_EXP4_dra, 0.9)

    # ---------------------------------------------------------------------------------------------
    # CALCULATE CORRESPONDING TPR AND FPR
    # ---------------------------------------------------------------------------------------------
    ## Natural
    # Start with the contingency table:
    A_M2_nat, B_M2_nat, C_M2_nat, D_M2_nat = Contingency_table(FWI_M2_nat, fires_raster_nat, threshold_M2_nat, 1)
    A_EXP1_nat, B_EXP1_nat, C_EXP1_nat, D_EXP1_nat = Contingency_table(FWI_EXP1_nat, fires_raster_nat,
                                                                       threshold_EXP1_nat, 1)
    A_EXP2_nat, B_EXP2_nat, C_EXP2_nat, D_EXP2_nat = Contingency_table(FWI_EXP2_nat, fires_raster_nat,
                                                                       threshold_EXP2_nat, 1)
    A_EXP3_nat, B_EXP3_nat, C_EXP3_nat, D_EXP3_nat = Contingency_table(FWI_EXP3_nat, fires_raster_nat,
                                                                       threshold_EXP3_nat, 1)
    A_EXP4_nat, B_EXP4_nat, C_EXP4_nat, D_EXP4_nat = Contingency_table(FWI_EXP4_nat, fires_raster_nat,
                                                                       threshold_EXP4_nat, 1)

    # Then the TPR and FPR:
    TPR_M2_nat, _, _, _, _, FPR_M2_nat = CT_metrics(A_M2_nat, B_M2_nat, C_M2_nat, D_M2_nat)
    TPR_EXP1_nat, _, _, _, _, FPR_EXP1_nat = CT_metrics(A_EXP1_nat, B_EXP1_nat, C_EXP1_nat, D_EXP1_nat)
    TPR_EXP2_nat, _, _, _, _, FPR_EXP2_nat = CT_metrics(A_EXP2_nat, B_EXP2_nat, C_EXP2_nat, D_EXP2_nat)
    TPR_EXP3_nat, _, _, _, _, FPR_EXP3_nat = CT_metrics(A_EXP3_nat, B_EXP3_nat, C_EXP3_nat, D_EXP3_nat)
    TPR_EXP4_nat, _, _, _, _, FPR_EXP4_nat = CT_metrics(A_EXP4_nat, B_EXP4_nat, C_EXP4_nat, D_EXP4_nat)

    ## Drained
    # Start with the contingency table:
    A_M2_dra, B_M2_dra, C_M2_dra, D_M2_dra = Contingency_table(FWI_M2_dra, fires_raster_dra, threshold_M2_dra, 1)
    A_EXP1_dra, B_EXP1_dra, C_EXP1_dra, D_EXP1_dra = Contingency_table(FWI_EXP1_dra, fires_raster_dra,
                                                                       threshold_EXP1_dra, 1)
    A_EXP2_dra, B_EXP2_dra, C_EXP2_dra, D_EXP2_dra = Contingency_table(FWI_EXP2_dra, fires_raster_dra,
                                                                       threshold_EXP2_dra, 1)
    A_EXP3_dra, B_EXP3_dra, C_EXP3_dra, D_EXP3_dra = Contingency_table(FWI_EXP3_dra, fires_raster_dra,
                                                                       threshold_EXP3_dra, 1)
    A_EXP4_dra, B_EXP4_dra, C_EXP4_dra, D_EXP4_dra = Contingency_table(FWI_EXP4_dra, fires_raster_dra,
                                                                       threshold_EXP4_dra, 1)

    # Then the TPR and FPR:
    TPR_M2_dra, _, _, _, _, FPR_M2_dra = CT_metrics(A_M2_dra, B_M2_dra, C_M2_dra, D_M2_dra)
    TPR_EXP1_dra, _, _, _, _, FPR_EXP1_dra = CT_metrics(A_EXP1_dra, B_EXP1_dra, C_EXP1_dra, D_EXP1_dra)
    TPR_EXP2_dra, _, _, _, _, FPR_EXP2_dra = CT_metrics(A_EXP2_dra, B_EXP2_dra, C_EXP2_dra, D_EXP2_dra)
    TPR_EXP3_dra, _, _, _, _, FPR_EXP3_dra = CT_metrics(A_EXP3_dra, B_EXP3_dra, C_EXP3_dra, D_EXP3_dra)
    TPR_EXP4_dra, _, _, _, _, FPR_EXP4_dra = CT_metrics(A_EXP4_dra, B_EXP4_dra, C_EXP4_dra, D_EXP4_dra)

    print(np.unique(FWI_EXP2_nat[nonan_EXP2_nat[:, 0], nonan_EXP2_nat[:, 1], nonan_EXP2_nat[:, 2]]))

    '''-------------------------------------------------Plotting-------------------------------------------------'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    # ---------------------------------------------------------------------------------------------
    # NATURAL PEATLANDS
    # ---------------------------------------------------------------------------------------------
    ## Reference:
    fpr_M2_nat, tpr_M2_nat, thresholds_M2_nat, auc_M2_nat = compute_roc(
        fires_raster_nat[nonan_M2_nat[:, 0], nonan_M2_nat[:, 1], nonan_M2_nat[:, 2]].flatten(),
        FWI_M2_nat[nonan_M2_nat[:, 0], nonan_M2_nat[:, 1], nonan_M2_nat[:, 2]].flatten())
    axes[0].plot(fpr_M2_nat, tpr_M2_nat, color=palette[0], label='FWI$_{ref}$ (AUC = %0.2f)' % auc_M2_nat)

    ## EXP1:
    fpr_EXP1_nat, tpr_EXP1_nat, thresholds_EXP1_nat, auc_EXP1_nat = \
        compute_roc(fires_raster_nat[nonan_EXP1_nat[:, 0], nonan_EXP1_nat[:, 1], nonan_EXP1_nat[:, 2]].flatten(),
                    FWI_EXP1_nat[nonan_EXP1_nat[:, 0], nonan_EXP1_nat[:, 1], nonan_EXP1_nat[:, 2]].flatten())
    axes[0].plot(fpr_EXP1_nat, tpr_EXP1_nat, color=palette[1], label='EXP1 (AUC = %0.2f)' % auc_EXP1_nat)

    ## EXP2:
    fpr_EXP2_nat, tpr_EXP2_nat, thresholds_EXP2_nat, auc_EXP2_nat = \
        compute_roc(fires_raster_nat[nonan_EXP2_nat[:, 0], nonan_EXP2_nat[:, 1], nonan_EXP2_nat[:, 2]].flatten(),
                    FWI_EXP2_nat[nonan_EXP2_nat[:, 0], nonan_EXP2_nat[:, 1], nonan_EXP2_nat[:, 2]].flatten())
    axes[0].plot(fpr_EXP2_nat, tpr_EXP2_nat, color=palette[2], label='EXP2 (AUC = %0.2f)' % auc_EXP2_nat)

    ## EXP3:
    fpr_EXP3_nat, tpr_EXP3_nat, thresholds_EXP3_nat, auc_EXP3_nat = \
        compute_roc(fires_raster_nat[nonan_EXP3_nat[:, 0], nonan_EXP3_nat[:, 1], nonan_EXP3_nat[:, 2]].flatten(),
                    FWI_EXP3_nat[nonan_EXP3_nat[:, 0], nonan_EXP3_nat[:, 1], nonan_EXP3_nat[:, 2]].flatten())
    axes[0].plot(fpr_EXP3_nat, tpr_EXP3_nat, color=palette[3], label='EXP3 (AUC = %0.2f)' % auc_EXP3_nat)

    ## EXP4:
    fpr_EXP4_nat, tpr_EXP4_nat, thresholds_EXP4_nat, auc_EXP4_nat = \
        compute_roc(fires_raster_nat[nonan_EXP4_nat[:, 0], nonan_EXP4_nat[:, 1], nonan_EXP4_nat[:, 2]].flatten(),
                    FWI_EXP4_nat[nonan_EXP4_nat[:, 0], nonan_EXP4_nat[:, 1], nonan_EXP4_nat[:, 2]].flatten())
    axes[0].plot(fpr_EXP4_nat, tpr_EXP4_nat, color=palette[4], label='EXP4 (AUC = %0.2f)' % auc_EXP4_nat)

    # Plot TPR and FPR of 90th percentile:
    ## Reference:
    axes[0].scatter(FPR_M2_nat, TPR_M2_nat, color=palette[0], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP1:
    axes[0].scatter(FPR_EXP1_nat, TPR_EXP1_nat, color=palette[1], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP2:
    axes[0].scatter(FPR_EXP2_nat, TPR_EXP2_nat, color=palette[2], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP3:
    axes[0].scatter(FPR_EXP3_nat, TPR_EXP3_nat, color=palette[3], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP4:
    axes[0].scatter(FPR_EXP4_nat, TPR_EXP4_nat, color=palette[4], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    # Beautify the plot:
    axes[0].set_title('Natural peatlands', fontsize=font_subtitle)
    axes[0].plot([0, 1], [0, 1], 'k--')  # Plot the 1:1 line

    axes[0].set_xlabel('False Positive Rate', fontsize=font_axes)
    axes[0].set_ylabel('True Positive Rate', fontsize=font_axes)
    axes[0].legend(fontsize=font_legend)
    axes[0].tick_params(axis='both', labelsize=font_ticklabels)

    # ---------------------------------------------------------------------------------------------
    # DRAINED PEATLANDS
    # ---------------------------------------------------------------------------------------------
    ## Reference:
    fpr_M2_dra, tpr_M2_dra, thresholds_M2_dra, auc_M2_dra = compute_roc(
        fires_raster_dra[nonan_M2_dra[:, 0], nonan_M2_dra[:, 1], nonan_M2_dra[:, 2]].flatten(),
        FWI_M2_dra[nonan_M2_dra[:, 0], nonan_M2_dra[:, 1], nonan_M2_dra[:, 2]].flatten())
    axes[1].plot(fpr_M2_dra, tpr_M2_dra, color=palette[0], label='FWI$_{ref}$ (AUC = %0.2f)' % auc_M2_dra)

    ## EXP1:
    fpr_EXP1_dra, tpr_EXP1_dra, thresholds_EXP1_dra, auc_EXP1_dra = \
        compute_roc(fires_raster_dra[nonan_EXP1_dra[:, 0], nonan_EXP1_dra[:, 1], nonan_EXP1_dra[:, 2]].flatten(),
                    FWI_EXP1_dra[nonan_EXP1_dra[:, 0], nonan_EXP1_dra[:, 1], nonan_EXP1_dra[:, 2]].flatten())
    axes[1].plot(fpr_EXP1_dra, tpr_EXP1_dra, color=palette[1], label='EXP1 (AUC = %0.2f)' % auc_EXP1_dra)

    ## EXP2:
    fpr_EXP2_dra, tpr_EXP2_dra, thresholds_EXP2_dra, auc_EXP2_dra = \
        compute_roc(fires_raster_dra[nonan_EXP2_dra[:, 0], nonan_EXP2_dra[:, 1], nonan_EXP2_dra[:, 2]].flatten(),
                    FWI_EXP2_dra[nonan_EXP2_dra[:, 0], nonan_EXP2_dra[:, 1], nonan_EXP2_dra[:, 2]].flatten())
    axes[1].plot(fpr_EXP2_dra, tpr_EXP2_dra, color=palette[2], label='EXP2 (AUC = %0.2f)' % auc_EXP2_dra)

    ## EXP3:
    fpr_EXP3_dra, tpr_EXP3_dra, thresholds_EXP3_dra, auc_EXP3_dra = \
        compute_roc(fires_raster_dra[nonan_EXP3_dra[:, 0], nonan_EXP3_dra[:, 1], nonan_EXP3_dra[:, 2]].flatten(),
                    FWI_EXP3_dra[nonan_EXP3_dra[:, 0], nonan_EXP3_dra[:, 1], nonan_EXP3_dra[:, 2]].flatten())
    axes[1].plot(fpr_EXP3_dra, tpr_EXP3_dra, color=palette[3], label='EXP3 (AUC = %0.2f)' % auc_EXP3_dra)

    ## EXP4:
    fpr_EXP4_dra, tpr_EXP4_dra, thresholds_EXP4_dra, auc_EXP4_dra = \
        compute_roc(fires_raster_dra[nonan_EXP4_dra[:, 0], nonan_EXP4_dra[:, 1], nonan_EXP4_dra[:, 2]].flatten(),
                    FWI_EXP4_dra[nonan_EXP4_dra[:, 0], nonan_EXP4_dra[:, 1], nonan_EXP4_dra[:, 2]].flatten())
    axes[1].plot(fpr_EXP4_dra, tpr_EXP4_dra, color=palette[4], label='EXP4 (AUC = %0.2f)' % auc_EXP4_dra)

    # Plot TPR and FPR of 90th percentile:
    ## Reference:
    axes[1].scatter(FPR_M2_dra, TPR_M2_dra, color=palette[0], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP1:
    axes[1].scatter(FPR_EXP1_dra, TPR_EXP1_dra, color=palette[1], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP2:
    axes[1].scatter(FPR_EXP2_dra, TPR_EXP2_dra, color=palette[2], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP3:
    axes[1].scatter(FPR_EXP3_dra, TPR_EXP3_dra, color=palette[3], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    ## EXP4:
    axes[1].scatter(FPR_EXP4_dra, TPR_EXP4_dra, color=palette[4], marker='*', s=70,
                    edgecolor='k', linewidths=0.5, zorder=2.5)

    # Beautify the plot:
    axes[1].set_title('Drained peatlands', fontsize=font_subtitle)
    axes[1].plot([0, 1], [0, 1], 'k--')  # Plot the 1:1 line

    axes[1].set_xlabel('False Positive Rate', fontsize=font_axes)
    axes[1].set_ylabel('True Positive Rate', fontsize=font_axes)
    axes[1].legend(fontsize=font_legend)
    axes[1].tick_params(axis='both', labelsize=font_ticklabels)

    fig.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(path_figs, 'ROC_rescaled' + type))

