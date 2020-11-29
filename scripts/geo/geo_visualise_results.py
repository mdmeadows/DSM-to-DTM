# Visualise: Performance of each modelling approach, with reference to test datasets & zones

# Import required packages
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from mpl_toolkits.basemap import Basemap
import pandas as pd
from osgeo import gdal, gdalconst
import pickle
gdal.UseExceptions()                       # Useful for trouble-shooting

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, array_to_geotiff

# List paths to GDAL scripts
ogr2ogr = 'C:/Anaconda3/envs/geo/Library/bin/ogr2ogr.exe'
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_rasterise = 'C:/Anaconda3/envs/geo/Library/bin/gdal_rasterize.exe'

# Define paths to relevant folders
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/proc'
folder_dtm = 'E:/mdm123/D/data/DTM/proc'
folder_logs = 'E:/mdm123/D/ML/logs'
folder_results_rf = 'E:/mdm123/D/ML/results/rf'
folder_results_densenet = 'E:/mdm123/D/ML/results/densenet'
folder_results_convnet = 'E:/mdm123/D/ML/results/convnet'
folder_fig = 'E:/mdm123/D/figures/All'
folder_maps = 'E:/mdm123/D/maps/PNG'
folder_input_1D = 'E:/mdm123/D/ML/inputs/1D'
folder_lcdb = 'E:/mdm123/D/data/LRIS/lris-lcdb-v50/proc'
folder_flha = 'E:/mdm123/D/data/NIWA/NZ_FLHA/proc'
folder_hand = 'E:/mdm123/D/maps/HAND/TIF'

# Define list of colours to be used for each ML model: in order of RF, DCN, FCN
models = ['rf','dcn','fcn']
model_colours = {'rf':'#fc8d62', 'dcn':'#66c2a5', 'fcn':'#8da0cb'}
label_colours = {'rf':'#d95f02', 'dcn':'#1b9e77', 'fcn':'#7570b3'}
dataset_colours = ['blue', 'green', 'firebrick']

# Define a list of the zones within which test areas were defined
test_zones = ['MRL18_WPE','MRL18_WVL','TSM16_ATG']

# Define various properties for the three test zones
test_zones_props = {'MRL18_WPE':{'label':'Wairau Plains East (Marlborough)',
                                 'elv_cbar_range':(0,12),
                                 'res_cbar_range':(-5,10)},
                    'MRL18_WVL':{'label':'Wairau Valley (Marlborough)',
                                 'elv_cbar_range':(230,300),
                                 'res_cbar_range':(-5,10)},
                    'TSM16_ATG':{'label':'Takaka (Tasman)',
                                 'elv_cbar_range':(0,75),
                                 'res_cbar_range':(-10,15)}}

# Define no_data value to be used
no_data = -9999


###############################################################################
# 1. Define additional helper functions specific to result visualisation      #
###############################################################################

# Define a function that retrieves the DTM, SRTM & DIFF arrays for a given zone, as well as a dictionary defining CRS properties for GeoTIFF generation
def get_base_raster_data(zone, no_data=-9999):
    
    # Import the SRTM raster for that zone - get array & geographic properties
    srtm_tif = '{}/{}/SRTM_{}_Z.tif'.format(folder_srtm, zone, zone)
    srtm_ds = gdal.Open(srtm_tif, gdalconst.GA_ReadOnly)
    srtm_proj, srtm_res_x, srtm_res_y, srtm_x_min, srtm_x_max, srtm_y_min, srtm_y_max, srtm_width, srtm_height = extract_projection_info(srtm_tif)
    srtm_props = {'proj':srtm_proj, 'res_x':srtm_res_x, 'res_y':srtm_res_y, 'x_min':srtm_x_min, 'y_max':srtm_y_max, 'x_max':srtm_x_max, 'y_min':srtm_y_min, 'width':srtm_width, 'height':srtm_height}
    srtm_array = np.array(srtm_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
    srtm_ds = None
    
    # Import the Resampled DTM raster for that zone - get array & geographic properties
    dtm_tif = '{}/{}/DTM_{}_30m_Median.tif'.format(folder_dtm, zone, zone)
    dtm_ds = gdal.Open(dtm_tif, gdalconst.GA_ReadOnly)
    dtm_array = np.array(dtm_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
    dtm_array[dtm_array==no_data] = np.nan
    dtm_ds = None
    
    # Import the SRTM-DTM DIFF raster for that zone - get array & geographic properties
    diff_tif = '{}/{}/SRTM_{}_Median_Diff.tif'.format(folder_srtm, zone, zone)
    diff_ds = gdal.Open(diff_tif, gdalconst.GA_ReadOnly)
    diff_array = np.array(diff_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
    diff_ds = None
    
    # Import raster describing which pixels in that zone were assigned to the test dataset
    test_tif = '{}/patches/TIF/{}_test_patches.tif'.format(folder_logs, zone)
    test_ds = gdal.Open(test_tif, gdalconst.GA_ReadOnly)
    test_array = np.array(test_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
    test_ds = None
    
    # Import the MERIT DEM raster for that zone - get array & geographic properties
    merit_tif = '{}/{}/MERIT_{}.tif'.format(folder_srtm.replace('SRTM','MERIT'), zone, zone)
    merit_ds = gdal.Open(merit_tif, gdalconst.GA_ReadOnly)
    merit_array = np.array(merit_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
    merit_ds = None
    
    # Import the FLHA (NIWA Flood Hazard) raster for that zone - get array & geographic properties
    flha_tif = '{}/FLHA_{}.tif'.format(folder_flha, zone)
    flha_ds = gdal.Open(flha_tif, gdalconst.GA_ReadOnly)
    flha_array = np.array(flha_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
    flha_ds = None
    # Replace no_data values (-9999) with 0 (to distinguish no_flood from no_data)
    flha_array = np.where(flha_array==no_data, 0, flha_array)
    
    # Import the HAND (Height Above Nearest Drainage) raster for that zone - get array & geographic properties
    hand_tif = '{}/DTM-SRTM_HAND_{}.tif'.format(folder_hand, zone)
    hand_ds = gdal.Open(hand_tif, gdalconst.GA_ReadOnly)
    hand_array = np.array(hand_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
    hand_ds = None
    
    # Return arrays for the DTM, SRTM & their DIFF, as well as the SRTM projection property dict
    return dtm_array, srtm_array, diff_array, test_array, merit_array, flha_array, hand_array, srtm_props

# Define a function that takes a prediction vector (pixel-based models) & returns three 2D arrays: corrections, corrected SRTM, and residuals
def process_1D_predictions(zone, prediction_vector, output_format, no_data=-9999):
    
    # Get 2D arrays for that zone's DTM, SRTM, DIFF & MERIT data, as well as the SRTM GeoTIFF's geographical projection properties
    dtm_array, srtm_array, diff_array, test_array, merit_array, flha_array, hand_array, srtm_props = get_base_raster_data(zone)
    
    # Check that length of predictions vector matches expectations
    if diff_array.size != len(prediction_vector): print('Prediction vector does not match expectation')
    if diff_array.shape != srtm_array.shape: print('SRTM & DIFF arrays are of different shapes')
    
    # Initialise new numpy arrays of the same dimensions as the SRTM array, for the full zone & also limited to the test patches
    pred_corrections = np.zeros(srtm_array.shape)
    pred_corrections_test = np.zeros(srtm_array.shape)
    pred_elevations = np.zeros(srtm_array.shape)
    pred_elevations_test = np.zeros(srtm_array.shape)
    
    # Make copies of other rasters, in which pixels out of test extent will be set to np.nan
    dtm_array_test = dtm_array.copy()
    srtm_array_test = srtm_array.copy()
    diff_array_test = diff_array.copy()
    merit_array_test = merit_array.copy()
    flha_array_test = flha_array.copy()
    hand_array_test = hand_array.copy()
    
    # Iterate through all cells, filling the new arrays as appropriate
    i = 0
    array_height = srtm_array.shape[0]
    array_width = srtm_array.shape[1]
    # Starting at the top & moving down:
    for j in range(array_height):
        # Starting at the left & moving right:
        for k in range(array_width):
            # If corresponding DIFF pixel is no_data, assign no_data pixels to all new arrays
            if diff_array[j,k] == no_data:
                pred_corrections[j,k] = np.nan
                pred_corrections_test[j,k] = np.nan
                pred_elevations[j,k] = np.nan
                pred_elevations_test[j,k] = np.nan
                dtm_array_test[j,k] = np.nan
                srtm_array_test[j,k] = np.nan
                diff_array_test[j,k] = np.nan
                merit_array_test[j,k] = np.nan
                flha_array_test[j,k] = np.nan
                hand_array_test[j,k] = np.nan
            # If corresponding DIFF pixel is valid, use the predictions available in the input vector
            else:
                # Full arrays processed the same regardless of whether or not this pixel belongs to a test patch
                correction = prediction_vector[i]
                elevation = srtm_array[j,k]
                pred_corrections[j,k] = correction
                pred_elevations[j,k] = elevation - correction
                # For test arrays, assign predicted values if patch is a test patch, otherwise no_data value
                if test_array[j,k]:
                    pred_corrections_test[j,k] = correction
                    pred_elevations_test[j,k] = elevation - correction
                else:
                    pred_corrections_test[j,k] = np.nan
                    pred_elevations_test[j,k] = np.nan
                    dtm_array_test[j,k] = np.nan
                    srtm_array_test[j,k] = np.nan
                    diff_array_test[j,k] = np.nan
                    merit_array_test[j,k] = np.nan
                    flha_array_test[j,k] = np.nan
                    hand_array_test[j,k] = np.nan
            # Increment the vector counter
            i += 1
    
    # Calculate residuals (diff - predictions)
    pred_residuals = diff_array - pred_corrections
    pred_residuals_test = pred_residuals.copy()
    pred_residuals_test[test_array==0] = np.nan
    
    # Get array extents of test data, for export of clipped arrays
    x_min = np.where(test_array==1)[1].min()
    x_max = np.where(test_array==1)[1].max()
    y_min = np.where(test_array==1)[0].min()
    y_max = np.where(test_array==1)[0].max()
    
    # Arrays returned will depend on the output_format argument provided
    if output_format == 'full_zone':
        return pred_corrections, pred_elevations, pred_residuals, dtm_array, srtm_array, diff_array, merit_array, flha_array, hand_array
    elif output_format == 'test_pad':
        return pred_corrections_test, pred_elevations_test, pred_residuals_test, dtm_array_test, srtm_array_test, diff_array_test, merit_array_test, flha_array_test, hand_array_test
    elif output_format == 'test_clip':
        return pred_corrections_test[y_min:y_max+1, x_min:x_max+1], pred_elevations_test[y_min:y_max+1, x_min:x_max+1], pred_residuals_test[y_min:y_max+1, x_min:x_max+1], dtm_array_test[y_min:y_max+1, x_min:x_max+1], srtm_array_test[y_min:y_max+1, x_min:x_max+1], diff_array_test[y_min:y_max+1, x_min:x_max+1], merit_array_test[y_min:y_max+1, x_min:x_max+1], flha_array_test[y_min:y_max+1, x_min:x_max+1], hand_array_test[y_min:y_max+1, x_min:x_max+1]
    else:
        print('Unknown input value for output_format!')
        return None

# Define a function that takes a prediction array (patch-based models) & returns three 2D arrays: corrections, corrected SRTM, and residuals
def process_2D_predictions(zone, prediction_array, output_format, no_data=-9999):
    
    # Get 2D arrays for that zone's DTM, SRTM, DIFF & MERIT data, as well as the SRTM GeoTIFF's geographical projection properties
    dtm_array, srtm_array, diff_array, test_array, merit_array, flha_array, hand_array, srtm_props = get_base_raster_data(zone)
    
    # Limit predictions array extent to same dimensions as other arrays (as they have same origin in upper-left)
    pred_corrections = prediction_array[0, :srtm_array.shape[0], :srtm_array.shape[1]]
    
    # Calculate elevations & residuals arrays
    pred_elevations = srtm_array - pred_corrections
    pred_residuals = diff_array - pred_corrections
    
    # Calculate another set of arrays, with nan values outside of the test patches extent
    dtm_array_test = dtm_array.copy()
    srtm_array_test = srtm_array.copy()
    diff_array_test = diff_array.copy()
    merit_array_test = merit_array.copy()
    flha_array_test = flha_array.copy()
    hand_array_test = hand_array.copy()
    pred_corrections_test = pred_corrections.copy()
    pred_elevations_test = pred_elevations.copy()
    pred_residuals_test = pred_residuals.copy()
    dtm_array_test[test_array==0] = np.nan
    srtm_array_test[test_array==0] = np.nan
    diff_array_test[test_array==0] = np.nan
    merit_array_test[test_array==0] = np.nan
    flha_array_test[test_array==0] = np.nan
    hand_array_test[test_array==0] = np.nan
    pred_corrections_test[test_array==0] = np.nan
    pred_elevations_test[test_array==0] = np.nan
    pred_residuals_test[test_array==0] = np.nan
    
    # Get array extents of test data, for export of clipped arrays
    x_min = np.where(test_array==1)[1].min()
    x_max = np.where(test_array==1)[1].max()
    y_min = np.where(test_array==1)[0].min()
    y_max = np.where(test_array==1)[0].max()
    
    # Arrays returned will depend on the output_format argument provided
    if output_format == 'full_zone':
        return pred_corrections, pred_elevations, pred_residuals, dtm_array, srtm_array, diff_array, merit_array, flha_array, hand_array
    elif output_format == 'test_pad':
        return pred_corrections_test, pred_elevations_test, pred_residuals_test, dtm_array_test, srtm_array_test, diff_array_test, merit_array_test, flha_array_test, hand_array_test
    elif output_format == 'test_clip':
        return pred_corrections_test[y_min:y_max+1, x_min:x_max+1], pred_elevations_test[y_min:y_max+1, x_min:x_max+1], pred_residuals_test[y_min:y_max+1, x_min:x_max+1], dtm_array_test[y_min:y_max+1, x_min:x_max+1], srtm_array_test[y_min:y_max+1, x_min:x_max+1], diff_array_test[y_min:y_max+1, x_min:x_max+1], merit_array_test[y_min:y_max+1, x_min:x_max+1], flha_array_test[y_min:y_max+1, x_min:x_max+1], hand_array_test[y_min:y_max+1, x_min:x_max+1]
    else:
        print('Unknown input value for output_format!')
        return None


###############################################################################
# 2. Generate rasters indicating test patch coverage within each zone         #
###############################################################################

# Define path to SHP of all patches, split up according to usage (train, dev, test)
patches_all = '{}/patches/SHP/patches_target_split.shp'.format(folder_logs)

# Loop through each of the test zones, generating a new SHP & GeoTIFF showing coverage of test patches for each
for zone in test_zones:
    
    print('Processing {} zone...'.format(zone))
    
    # Save a filtered version of the patch SHP, containing only those in the appropriate zone
    patches_zone_shp = '{}/patches/SHP/patches_target_split_{}.shp'.format(folder_logs, zone)
    filter_query = "zone = '{}' and usage = 'test'".format(zone)
    filter_command = [ogr2ogr, patches_zone_shp, patches_all, '-sql', 'SELECT * FROM patches_target_split WHERE {}'.format(filter_query), '-overwrite']
    filter_result = subprocess.run(filter_command, stdout=subprocess.PIPE)
    if filter_result.returncode != 0: print(filter_result.stdout)
    
    # Read that zone's SRTM GeoTIFF's geographical projection properties, to use as a template
    diff_tif = '{}/{}/SRTM_{}_Median_Diff.tif'.format(folder_srtm, zone, zone)
    diff_proj, diff_res_x, diff_res_y, diff_x_min, diff_x_max, diff_y_min, diff_y_max, diff_width, diff_height = extract_projection_info(diff_tif)
    
    # Convert new SHP to a GeoTIFF, using that zone's DIFF raster as a template
    patches_zone_tif = '{}/patches/TIF/{}_test_patches.tif'.format(folder_logs, zone)
    rasterise_command = [gdal_rasterise, '-te', str(diff_x_min), str(diff_y_min), str(diff_x_max), str(diff_y_max), '-tr', str(diff_res_x), str(-diff_res_y), '-burn', '1', '-ot', 'Int16', '-a_nodata', '-9999', '-init', '0', patches_zone_shp, patches_zone_tif]
    rasterise_result = subprocess.run(rasterise_command, stdout=subprocess.PIPE)
    if rasterise_result.returncode != 0: print(rasterise_result.stdout)


###############################################################################
# 3. Generate prediction GeoTIFFs for each test zone, for each ML model       #
###############################################################################

# Random Forest predictions
for zone in test_zones:
    # Import zone predictions as vector
    rf_predictions = np.load('{}/RF_Predictions_ByZone_{}.npy'.format(folder_results_rf, zone))
    # Get general 2D arrays for this zone
    dtm_array, srtm_array, diff_array, test_array, merit_array, flha_array, hand_array, srtm_props = get_base_raster_data(zone)
    # Get 2D result arrays covering FULL zone & save to GeoTIFFs
    rf_corrections_array, rf_elevations_array, rf_residuals_array, _,_,_,_,_,_ = process_1D_predictions(zone, rf_predictions, 'full_zone', no_data=-9999)
    rf_corrections_tif = '{}/TIF/rf_corrections_{}.tif'.format(folder_results_rf, zone)
    rf_elevations_tif = '{}/TIF/rf_elevations_{}.tif'.format(folder_results_rf, zone)
    rf_residuals_tif = '{}/TIF/rf_residuals_{}.tif'.format(folder_results_rf, zone)
    array_to_geotiff(rf_corrections_array, rf_corrections_tif, -9999, srtm_props)
    array_to_geotiff(rf_elevations_array, rf_elevations_tif, -9999, srtm_props)
    array_to_geotiff(rf_residuals_array, rf_residuals_tif, -9999, srtm_props)
    # Get 2D result arrays covering FULL zone (with non-test pixels set to np.nan) & save to GeoTIFFs
    rf_corrections_test_array, rf_elevations_test_array, rf_residuals_test_array, _,_,_,_,_,_ = process_1D_predictions(zone, rf_predictions, 'test_pad', no_data=-9999)
    rf_corrections_test_tif = '{}/TIF/rf_corrections_{}_test.tif'.format(folder_results_rf, zone)
    rf_elevations_test_tif = '{}/TIF/rf_elevations_{}_test.tif'.format(folder_results_rf, zone)
    rf_residuals_test_tif = '{}/TIF/rf_residuals_{}_test.tif'.format(folder_results_rf, zone)
    array_to_geotiff(rf_corrections_test_array, rf_corrections_test_tif, -9999, srtm_props)
    array_to_geotiff(rf_elevations_test_array, rf_elevations_test_tif, -9999, srtm_props)
    array_to_geotiff(rf_residuals_test_array, rf_residuals_test_tif, -9999, srtm_props)

# Densely-connected neural network predictions
for zone in test_zones:
    # Import zone predictions as vector
    densenet_predictions = np.load('{}/predictions/densenet_ensemble_{}_prediction.npy'.format(folder_results_densenet, zone))
    # Get general 2D arrays for this zone
    dtm_array, srtm_array, diff_array, test_array, merit_array, flha_array, hand_array, srtm_props = get_base_raster_data(zone)
    # Get 2D result arrays covering FULL zone & save to GeoTIFFs
    densenet_corrections_array, densenet_elevations_array, densenet_residuals_array,_,_,_,_,_,_ = process_1D_predictions(zone, densenet_predictions, 'full_zone', no_data=-9999)
    densenet_corrections_tif = '{}/TIF/densenet_corrections_{}.tif'.format(folder_results_densenet, zone)
    densenet_elevations_tif = '{}/TIF/densenet_elevations_{}.tif'.format(folder_results_densenet, zone)
    densenet_residuals_tif = '{}/TIF/densenet_residuals_{}.tif'.format(folder_results_densenet, zone)
    array_to_geotiff(densenet_corrections_array, densenet_corrections_tif, -9999, srtm_props)
    array_to_geotiff(densenet_elevations_array, densenet_elevations_tif, -9999, srtm_props)
    array_to_geotiff(densenet_residuals_array, densenet_residuals_tif, -9999, srtm_props)
    # Get 2D result arrays covering FULL zone (with non-test pixels set to np.nan) & save to GeoTIFFs
    densenet_corrections_test_array, densenet_elevations_test_array, densenet_residuals_test_array,_,_,_,_,_,_ = process_1D_predictions(zone, densenet_predictions, 'test_pad', no_data=-9999)
    densenet_corrections_test_tif = '{}/TIF/densenet_corrections_{}_test.tif'.format(folder_results_densenet, zone)
    densenet_elevations_test_tif = '{}/TIF/densenet_elevations_{}_test.tif'.format(folder_results_densenet, zone)
    densenet_residuals_test_tif = '{}/TIF/densenet_residuals_{}_test.tif'.format(folder_results_densenet, zone)
    array_to_geotiff(densenet_corrections_test_array, densenet_corrections_test_tif, -9999, srtm_props)
    array_to_geotiff(densenet_elevations_test_array, densenet_elevations_test_tif, -9999, srtm_props)
    array_to_geotiff(densenet_residuals_test_array, densenet_residuals_test_tif, -9999, srtm_props)

# Fully-convolutional neural network predictions
for zone in test_zones:
    # Import zone predictions as array
    convnet_predictions = np.load('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results_convnet, zone))
    # Get general 2D arrays for this zone
    dtm_array, srtm_array, diff_array, test_array, merit_array, flha_array, hand_array, srtm_props = get_base_raster_data(zone)
    # Get 2D result arrays covering FULL zone & save to GeoTIFFs
    convnet_corrections_array, convnet_elevations_array, convnet_residuals_array,_,_,_,_,_,_ = process_2D_predictions(zone, convnet_predictions, 'full_zone', no_data=-9999)
    convnet_corrections_tif = '{}/TIF/convnet_corrections_{}.tif'.format(folder_results_convnet, zone)
    convnet_elevations_tif = '{}/TIF/convnet_elevations_{}.tif'.format(folder_results_convnet, zone)
    convnet_residuals_tif = '{}/TIF/convnet_residuals_{}.tif'.format(folder_results_convnet, zone)
    array_to_geotiff(convnet_corrections_array, convnet_corrections_tif, -9999, srtm_props)
    array_to_geotiff(convnet_elevations_array, convnet_elevations_tif, -9999, srtm_props)
    array_to_geotiff(convnet_residuals_array, convnet_residuals_tif, -9999, srtm_props)
    # Get 2D result arrays covering FULL zone (with non-test pixels set to np.nan) & save to GeoTIFFs
    convnet_corrections_test_array, convnet_elevations_test_array, convnet_residuals_test_array,_,_,_,_,_,_ = process_2D_predictions(zone, convnet_predictions, 'test_pad', no_data=-9999)
    convnet_corrections_test_tif = '{}/TIF/convnet_corrections_{}_test.tif'.format(folder_results_convnet, zone)
    convnet_elevations_test_tif = '{}/TIF/convnet_elevations_{}_test.tif'.format(folder_results_convnet, zone)
    convnet_residuals_test_tif = '{}/TIF/convnet_residuals_{}_test.tif'.format(folder_results_convnet, zone)
    array_to_geotiff(convnet_corrections_test_array, convnet_corrections_test_tif, -9999, srtm_props)
    array_to_geotiff(convnet_elevations_test_array, convnet_elevations_test_tif, -9999, srtm_props)
    array_to_geotiff(convnet_residuals_test_array, convnet_residuals_test_tif, -9999, srtm_props)


###############################################################################
# 4. Map elevations & residuals in each zone, showing results of all models   #
###############################################################################

# Loop through each test zone
for zone in test_zones:
    
    # Import Random Forest arrays - ONLY test dataset pixels
    rf_predictions = np.load('{}/RF_Predictions_ByZone_{}.npy'.format(folder_results_rf, zone))
    rf_cor, rf_elv, rf_res, _,_,_,_,_,_ = process_1D_predictions(zone, rf_predictions, 'test_clip', no_data=-9999)
    
    # Import Densenet arrays - ONLY test dataset pixels
    dn_predictions = np.load('{}/predictions/densenet_ensemble_{}_prediction.npy'.format(folder_results_densenet, zone))
    dn_cor, dn_elv, dn_res, _,_,_,_,_,_ = process_1D_predictions(zone, dn_predictions, 'test_clip', no_data=-9999)
    
    # Import Convnet arrays - ONLY test dataset pixels
    cn_predictions = np.load('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results_convnet, zone))
    cn_cor, cn_elv, cn_res, dtm, srtm, diff, merit, flha, hand = process_2D_predictions(zone, cn_predictions, 'test_clip', no_data=-9999)
    
    # Read satellite imagery for that test zone into memory
    sat_img = plt.imread('{}/SatImg_{}.png'.format(folder_maps, zone))
        
    # Evaluate RMSE of all DSMs, with reference to the LiDAR DTM available
    srtm_RMSE = np.sqrt(np.nanmean(np.square(dtm - srtm)))
    merit_RMSE = np.sqrt(np.nanmean(np.square(dtm - merit)))
    rf_RMSE = np.sqrt(np.nanmean(np.square(dtm - rf_elv)))
    dn_RMSE = np.sqrt(np.nanmean(np.square(dtm - dn_elv)))
    cn_RMSE = np.sqrt(np.nanmean(np.square(dtm - cn_elv)))
    
    # Get corresponding improvement, with respect to SRTM's RMSE
    merit_RMSE_reduction = (srtm_RMSE - merit_RMSE)/srtm_RMSE * 100
    rf_RMSE_reduction = (srtm_RMSE - rf_RMSE)/srtm_RMSE * 100
    dn_RMSE_reduction = (srtm_RMSE - dn_RMSE)/srtm_RMSE * 100
    cn_RMSE_reduction = (srtm_RMSE - cn_RMSE)/srtm_RMSE * 100
    
    # Get elevation range & build appropriate colourmap
    elv_min, elv_max = test_zones_props[zone]['elv_cbar_range']
    elv_cmap = cm.terrain
    elv_cmap.set_bad(color='whitesmoke')
    elv_norm = colors.Normalize(vmin=elv_min, vmax=elv_max)
    
    # Get residual range & build appropriate colourmap
    res_min, res_max = test_zones_props[zone]['res_cbar_range']
    res_cmap = cm.coolwarm
    res_cmap.set_bad(color='whitesmoke')
    res_norm = colors.Normalize(vmin=res_min, vmax=res_max)
    res_norm = colors.TwoSlopeNorm(vmin=res_min, vcenter=0.0, vmax=res_max)
    
    # For the 'TSM16_ATG' zone, rotate arrays for easier plotting
    if zone == 'TSM16_ATG':
        dtm, srtm, merit, diff, sat_img, rf_elv, rf_res, dn_elv, dn_res, cn_elv, cn_res = [np.rot90(raster, axes=(1,0)) for raster in [dtm, srtm, merit, diff, sat_img, rf_elv, rf_res, dn_elv, dn_res, cn_elv, cn_res]]
    
    # Determine figure size based on desired width & array dimensions
    width = 8
    scale = width/(2*srtm.shape[1])
    height = 1.05 * 5 * srtm.shape[0] * scale
    
    # Generate plot showing both elevations & residuals
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(width,height))
    # Row 1A: LiDAR DTM
    axes[0,0].imshow(dtm, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[0,0].set_title('a) LiDAR (resampled to SRTM resolution)', x=0, ha='left', size=9, pad=4)
    axes[0,0].axis('off')
    # Row 1B: Satellite imagery
    axes[0,1].imshow(sat_img, aspect='equal')
    axes[0,1].set_title("b) 'NZ Imagery' basemap (LINZ Data Service)", x=0, ha='left', size=9, pad=4)
    axes[0,1].axis('off')
    # Row 2A: SRTM DSM
    axes[1,0].imshow(srtm, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[1,0].set_title('c) SRTM: RMSE={:.2f}m (compared to LiDAR)'.format(srtm_RMSE), x=0, ha='left', size=9, pad=4)
    axes[1,0].axis('off')
    # Row 2B: SRTM Residuals
    axes[1,1].imshow(diff, aspect='equal', cmap=res_cmap, norm=res_norm)
    axes[1,1].set_title('d) SRTM residuals: \u03BC={:.2f}m, \u03C3={:.2f}m'.format(np.nanmean(diff), np.nanstd(diff)), x=0, ha='left', size=9, pad=4)
    axes[1,1].axis('off')
    # Row 3A: Random Forest - elevations
    axes[2,0].imshow(rf_elv, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[2,0].set_title('e) RF-corrected SRTM: RMSE={:.2f}m (-{:.1f}%)'.format(rf_RMSE, rf_RMSE_reduction), x=0, ha='left', size=9, pad=4)
    axes[2,0].axis('off')
    # Row 3B: Random Forest - residuals
    axes[2,1].imshow(rf_res, aspect='equal', cmap=res_cmap, norm=res_norm)
    axes[2,1].set_title('f) RF-corrected SRTM residuals: \u03BC={:.2f}m, \u03C3={:.2f}m'.format(np.nanmean(rf_res), np.nanstd(rf_res)), x=0, ha='left', size=9, pad=4)
    axes[2,1].axis('off')
    # Row 4A: Densenet - elevations
    axes[3,0].imshow(dn_elv, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[3,0].set_title('g) DCN-corrected SRTM: RMSE={:.2f}m (-{:.1f}%)'.format(dn_RMSE, dn_RMSE_reduction), x=0, ha='left', size=9, pad=4)
    axes[3,0].axis('off')
    # Row 4B: Densenet - residuals
    axes[3,1].imshow(dn_res, aspect='equal', cmap=res_cmap, norm=res_norm)
    axes[3,1].set_title('h) DCN-corrected SRTM: \u03BC={:.2f}m, \u03C3={:.2f}m'.format(np.nanmean(dn_res), np.nanstd(dn_res)), x=0, ha='left', size=9, pad=4)
    axes[3,1].axis('off')
    # Row 5A: Convnet - elevations
    axes[4,0].imshow(cn_elv, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[4,0].set_title('i) FCN-corrected SRTM: RMSE={:.2f}m (-{:.1f}%)'.format(cn_RMSE, cn_RMSE_reduction), x=0, ha='left', size=9, pad=4)
    axes[4,0].axis('off')
    # Row 5B: Convnet - residuals
    axes[4,1].imshow(cn_res, aspect='equal', cmap=res_cmap, norm=res_norm)
    axes[4,1].set_title('j) FCN-corrected SRTM residuals: \u03BC={:.2f}m, \u03C3={:.2f}m'.format(np.nanmean(cn_res), np.nanstd(cn_res)), x=0, ha='left', size=9, pad=4)
    axes[4,1].axis('off')
    # Add a small north arrow indicator to each map
    arrowprops = dict(facecolor='black', width=1.5, headwidth=4, headlength=4)
    if zone == 'TSM16_ATG':
        x, y, arrow_length = 0.1, 0.95, 0.06
        xytext = (x-arrow_length, y)
    else:
        x, y, arrow_length = 0.97, 0.96, 0.11
        xytext = (x, y-arrow_length)
    for ax in axes.ravel():
        ax.annotate('N', xy=(x,y), xycoords='axes fraction', xytext=xytext, textcoords='axes fraction', arrowprops=arrowprops, ha='center', va='center', fontsize=8)
    # Add a simple scale bar to the DTM map, assuming that each grid cell is approx. 23m (SRTM at this latitude)
    ncells_1km = 1000/23
    offset = 8
    adjust_y = 10 if zone == 'TSM16_ATG' else 0
    axes[0,0].plot([offset, offset + ncells_1km], [offset + adjust_y, offset + adjust_y], color='black', linewidth=0.8)
    axes[0,0].plot([offset, offset], [offset-1+adjust_y, offset+1+adjust_y], color='black', linewidth=0.8)
    axes[0,0].plot([offset + ncells_1km, offset + ncells_1km], [offset-1+adjust_y, offset+1+adjust_y], color='black', linewidth=0.8)
    axes[0,0].annotate('1km', xy=(offset + 0.5*ncells_1km, 1.3*offset + adjust_y), ha='center', va='top', size=8)
    # Tighten layout
    fig.tight_layout(pad=1)
    # Adjust layout to fit two colourbars at the bottom
    fig.subplots_adjust(top=0.98, bottom=0.06, wspace=0.05, hspace=0.12)
    # Add colourbar for elevations
    elv_cbar = fig.add_axes([0.03, 0.04, 0.4, 0.01])                          # [left, bottom, width, height]
    fig.colorbar(cm.ScalarMappable(cmap=elv_cmap, norm=elv_norm), cax=elv_cbar, orientation='horizontal').set_label(label='Elevation [m]', size=9)
    elv_cbar.tick_params(labelsize=8)
    # Add colourbar for residuals
    res_cbar = fig.add_axes([0.5, 0.04, 0.4, 0.01])                          # [left, bottom, width, height]
    fig.colorbar(cm.ScalarMappable(cmap=res_cmap, norm=res_norm), cax=res_cbar, orientation='horizontal').set_label(label='Residuals [m]', size=9)
    res_cbar.tick_params(labelsize=8)
    # Save figure
    fig.savefig('{}/maps_elv_res_{}.png'.format(folder_fig, zone), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # Generate plot showing only elevations, and only the convnet results
    width = 8
    scale = width/(srtm.shape[1])
    height = 1.05 * 3 * srtm.shape[0] * scale
    fig, axes = plt.subplots(nrows=3, figsize=(width,height))
    # Row 1: LiDAR DTM
    axes[0].imshow(dtm, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[0].set_title('a) LiDAR (resampled to SRTM resolution)', x=0, ha='left', size=9, pad=4)
    axes[0].axis('off')
    # Row 2: SRTM DSM
    axes[1].imshow(srtm, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[1].set_title('b) SRTM: RMSE={:.2f}m (compared to LiDAR)'.format(srtm_RMSE), x=0, ha='left', size=9, pad=4)
    axes[1].axis('off')
    # Row 3: Convnet - elevations
    axes[2].imshow(cn_elv, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[2].set_title('c) FCN-corrected SRTM: RMSE={:.2f}m (an improvement of {:.1f}% over raw SRTM)'.format(cn_RMSE, cn_RMSE_reduction), x=0, ha='left', size=9, pad=4)
    axes[2].axis('off')
    # Add a small north arrow indicator to each map
    arrowprops = dict(facecolor='black', width=1.5, headwidth=4, headlength=4)
    if zone == 'TSM16_ATG':
        x, y, arrow_length = 0.07, 0.95, 0.04
        xytext = (x-arrow_length, y)
    else:
        x, y, arrow_length = 0.97, 0.96, 0.07
        xytext = (x, y-arrow_length)
    for ax in axes.ravel():
        ax.annotate('N', xy=(x,y), xycoords='axes fraction', xytext=xytext, textcoords='axes fraction', arrowprops=arrowprops, ha='center', va='center', fontsize=10)
    # Add a simple scale bar to the DTM map, assuming that each grid cell is approx. 23m (SRTM at this latitude)
    ncells_1km = 1000/23
    offset = 8
    adjust_y = 10 if zone == 'TSM16_ATG' else 0
    axes[0].plot([offset, offset + ncells_1km], [offset + adjust_y, offset + adjust_y], color='black', linewidth=0.8)
    axes[0].plot([offset, offset], [offset-1+adjust_y, offset+1+adjust_y], color='black', linewidth=0.8)
    axes[0].plot([offset + ncells_1km, offset + ncells_1km], [offset-1+adjust_y, offset+1+adjust_y], color='black', linewidth=0.8)
    axes[0].annotate('1km', xy=(offset + 0.5*ncells_1km, 1.3*offset + adjust_y), ha='center', va='top', size=8)
    # Tighten layout
    fig.tight_layout(pad=1)
    # Adjust layout to fit two colourbars at the bottom
    fig.subplots_adjust(bottom=0.07, wspace=0.05, hspace=0.07)
    # Add colourbar for elevations
    elv_cbar = fig.add_axes([0.06, 0.04, 0.88, 0.015])                          # [left, bottom, width, height]
    fig.colorbar(cm.ScalarMappable(cmap=elv_cmap, norm=elv_norm), cax=elv_cbar, orientation='horizontal').set_label(label='Elevation [m]', size=9)
    elv_cbar.tick_params(labelsize=8)
    # Save figure
    fig.savefig('{}/maps_elv_{}_convnet.png'.format(folder_fig, zone), dpi=300, bbox_inches='tight')
    plt.close()


###############################################################################
# 5. Compare overall residuals using boxplots                                 #
###############################################################################

# Read in error residuals calculated earlier for the test dataset
residuals_dict_rf = pickle.load(open('{}/rf_residuals.p'.format(folder_results_rf), 'rb'))
residuals_dict_densenet = pickle.load(open('{}/densenet_residuals_models.p'.format(folder_results_densenet), 'rb'))
residuals_dict_convnet = pickle.load(open('{}/convnet_residuals_models.p'.format(folder_results_convnet), 'rb'))

# Check that initial residuals are the same (in each dictionary)
fig, axes = plt.subplots(figsize=(9,5))
axes.boxplot([d['test']['initial'] for d in [residuals_dict_rf, residuals_dict_densenet, residuals_dict_convnet]], showfliers=False)

# Get residuals to plot
res_initial = residuals_dict_convnet['test']['initial']
res_baseline = residuals_dict_convnet['test']['naive']
res_rf = residuals_dict_rf['test']['rf']
res_densenet = residuals_dict_densenet['test']['densenet_ensemble']
res_convnet = residuals_dict_convnet['test']['convnet_ensemble']

# Boxplots of error residuals
bp_data = [res_initial, res_baseline, res_rf, res_densenet, res_convnet]
bp_colours = ['dimgrey', 'darkgrey'] + [model_colours[m] for m in models]
bp_label_colours = ['dimgrey', 'darkgrey'] + [label_colours[m] for m in models]
# Add boxplots to the figure
fig, axes = plt.subplots(figsize=(9,5))
bps = axes.boxplot(bp_data, showfliers=False, medianprops={'color':'black'}, patch_artist=True)
for patch, colour in zip(bps['boxes'], bp_colours):
    patch.set_facecolor(colour)
# Add axis ticks & labels
axes.set_xticks(range(1,6))
axes.set_xticklabels(['Initial','Baseline\ncorrection','RF\ncorrection','DCN\ncorrection','FCN\ncorrection'])
axes.set_ylabel('Residual error before/after correction [m]')
# Turn spines off
[axes.spines[edge].set_visible(False) for edge in ['top','right']]
# Add a horizontal line for zero error
axes.axhline(y=0, linestyle='dashed', color='black', linewidth=0.8, alpha=0.3)
# Add labels for medians & IQR
iqr_label_y = 0
for i, data in enumerate(bp_data):
    median = np.median(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    iqr_label_y = max(1.02*(q75 + 1.5*iqr), iqr_label_y)
    axes.annotate('{:.3f}m'.format(median), xy=(i+1.28, median), ha='left', va='center')
    axes.annotate('IQR = {:.3f}m'.format(iqr), xy=(i+1, iqr_label_y), color=bp_label_colours[i], fontweight='bold', ha='center', va='bottom')
fig.tight_layout()
fig.savefig('{}/residuals_bymodel_boxplots.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 6. Assess correction efficacy by zone, land cover, FLHA class & HAND range  #
###############################################################################

# Set up a dictionary to contain SRTM-LiDAR difference values corresponding to each Manaaki Whenua landclass type present in that LiDAR zone coverage
diff_by_landcover = {1:{'label':'Artificial\nsurfaces', 'data':[], 'colour':(78/255, 78/255, 78/255)},
                     2:{'label':'Bare/lightly-\nvegetated\nsurfaces', 'data':[], 'colour':(255/255, 235/255, 190/255)},
                     3:{'label':'Water\nbodies', 'data':[], 'colour':(0/255, 197/255, 255/255)},
                     4:{'label':'Cropland', 'data':[], 'colour':(255/255, 170/255, 0/255)},
                     5:{'label':'Grassland,\nSedgeland\n& Marshland', 'data':[], 'colour':(255/255, 255/255, 115/255)},
                     6:{'label':'Scrub &\nShrubland', 'data':[], 'colour':(137/255, 205/255, 102/255)},
                     7:{'label':'Forest', 'data':[], 'colour':(38/255, 115/255, 0/255)},
                     8:{'label':'Other', 'data':[], 'colour':'#FF0000'}}

# Initalise dictionary to hold test residuals, classed in different ways
res = {'initial':{'by_zone':{zone:[] for zone in ['All'] + test_zones}, 'by_lcdb':{i:[] for i in range(1,8)}, 'by_flha':{'flood':[], 'noflood':[]}, 'by_hand':{'hand_{}'.format(h):[] for h in range(1,6)}},
       'rf':{'by_zone':{zone:[] for zone in ['All'] + test_zones}, 'by_lcdb':{i:[] for i in range(1,8)}, 'by_flha':{'flood':[], 'noflood':[]}, 'by_hand':{'hand_{}'.format(h):[] for h in range(1,6)}},
       'dn':{'by_zone':{zone:[] for zone in ['All'] + test_zones}, 'by_lcdb':{i:[] for i in range(1,8)}, 'by_flha':{'flood':[], 'noflood':[]}, 'by_hand':{'hand_{}'.format(h):[] for h in range(1,6)}},
       'cn':{'by_zone':{zone:[] for zone in ['All'] + test_zones}, 'by_lcdb':{i:[] for i in range(1,8)}, 'by_flha':{'flood':[], 'noflood':[]}, 'by_hand':{'hand_{}'.format(h):[] for h in range(1,6)}}}

# Loop through the three test zones
for i, zone in enumerate(test_zones):
    
    print('Processing {} zone...'.format(zone))
    
    # Get the test array & SRTM props dictionary
    _, _, _, test, _, _, _, srtm_props = get_base_raster_data(zone)
    
    # Land cover classes
    lcdb_tif = '{}/LCDB_GroupID_{}.tif'.format(folder_lcdb, zone)
    lcdb_ds = gdal.Open(lcdb_tif, gdalconst.GA_ReadOnly)
    lcdb = np.array(lcdb_ds.ReadAsArray())
    lcdb_ds = None
    
    # Import Random Forest arrays - ONLY test dataset pixels
    rf_predictions = np.load('{}/RF_Predictions_ByZone_{}.npy'.format(folder_results_rf, zone))
    rf_cor, rf_elv, rf_res, _,_,_,_,_,_ = process_1D_predictions(zone, rf_predictions, 'test_clip', no_data=-9999)
    
    # Import Densenet arrays - ONLY test dataset pixels
    dn_predictions = np.load('{}/predictions/densenet_ensemble_{}_prediction.npy'.format(folder_results_densenet, zone))
    dn_cor, dn_elv, dn_res, _,_,_,_,_,_ = process_1D_predictions(zone, dn_predictions, 'test_clip', no_data=-9999)
    
    # Import Convnet arrays - ONLY test dataset pixels
    cn_predictions = np.load('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results_convnet, zone))
    cn_cor, cn_elv, cn_res, dtm, srtm, diff, merit, flha, hand = process_2D_predictions(zone, cn_predictions, 'test_clip', no_data=-9999)
    
    # Check extent of the test patch coverage, with reference to the zone coverage as a whole
    x_min = np.where(test==1)[1].min()
    x_max = np.where(test==1)[1].max()
    y_min = np.where(test==1)[0].min()
    y_max = np.where(test==1)[0].max()
    
    # For the LCDB array, set to np.nan any pixels which aren't in the test patches & clip it to test data extent
    lcdb[test==0] = np.nan
    lcdb = lcdb[y_min:y_max+1, x_min:x_max+1]
    
    # Mask all arrays wherever no_data values are present
    dtm = np.ma.masked_equal(dtm, no_data)
    srtm = np.ma.masked_equal(srtm, no_data)
    diff = np.ma.masked_equal(diff, no_data)
    test = np.ma.masked_equal(test[y_min:y_max+1, x_min:x_max+1], no_data)
    lcdb = np.ma.masked_equal(lcdb, no_data)
    flha = np.ma.masked_equal(flha, no_data)
    hand = np.ma.masked_equal(hand, no_data)
    rf_cor = np.ma.masked_equal(rf_cor, no_data)
    rf_elv = np.ma.masked_equal(rf_elv, no_data)
    rf_res = np.ma.masked_equal(rf_res, no_data)
    dn_cor = np.ma.masked_equal(dn_cor, no_data)
    dn_elv = np.ma.masked_equal(dn_elv, no_data)
    dn_res = np.ma.masked_equal(dn_res, no_data)
    cn_cor = np.ma.masked_equal(cn_cor, no_data)
    cn_elv = np.ma.masked_equal(cn_elv, no_data)
    cn_res = np.ma.masked_equal(cn_res, no_data)
    
    # Check that all arrays have the same shape
    if not (dtm.shape == srtm.shape == diff.shape == test.shape == lcdb.shape == flha.shape == hand.shape == rf_cor.shape == dn_cor.shape == cn_cor.shape):
        print('Different test array dimensions!')
        break
    
    # Class residuals by test zone
    # Get list of residuals for that zone (for each model)
    res_initial_byzone = diff.flatten().tolist()
    res_rf_byzone = rf_res.flatten().tolist()
    res_dn_byzone = dn_res.flatten().tolist()
    res_cn_byzone = cn_res.flatten().tolist()
    # Filter out any None values (masked)
    res_initial_byzone = [r for r in res_initial_byzone if (not np.isnan(r) and r != None)]
    res_rf_byzone = [r for r in res_rf_byzone if (not np.isnan(r) and r != None)]
    res_dn_byzone = [r for r in res_dn_byzone if (not np.isnan(r) and r != None)]
    res_cn_byzone = [r for r in res_cn_byzone if (not np.isnan(r) and r != None)]
    # Update dictionary of all test residuals
    res['initial']['by_zone'][zone] = res_initial_byzone
    res['initial']['by_zone']['All'] = np.append(res['initial']['by_zone']['All'], res_initial_byzone)
    res['rf']['by_zone'][zone] = res_rf_byzone
    res['rf']['by_zone']['All'] = np.append(res['rf']['by_zone']['All'], res_rf_byzone)
    res['dn']['by_zone'][zone] = res_dn_byzone
    res['dn']['by_zone']['All'] = np.append(res['dn']['by_zone']['All'], res_dn_byzone)
    res['cn']['by_zone'][zone] = res_cn_byzone
    res['cn']['by_zone']['All'] = np.append(res['cn']['by_zone']['All'], res_cn_byzone)
    
    # Class residuals by land cover class
    # Loop through each potential land cover class (as defined in proc_LCDB.py) and calculate elevation residuals for that particular class
    for i in range(1,8):
        # Get lists of residuals for that land cover class - for each of the input residual arrays
        res_initial_byclass = diff[lcdb==i].flatten().tolist()
        res_rf_byclass = rf_res[lcdb==i].flatten().tolist()
        res_dn_byclass = dn_res[lcdb==i].flatten().tolist()
        res_cn_byclass = cn_res[lcdb==i].flatten().tolist()
        # Filter out any None values (masked)
        res_initial_byclass = [r for r in res_initial_byclass if (not np.isnan(r) and r != None)]
        res_rf_byclass = [r for r in res_rf_byclass if (not np.isnan(r) and r != None)]
        res_dn_byclass = [r for r in res_dn_byclass if (not np.isnan(r) and r != None)]
        res_cn_byclass = [r for r in res_cn_byclass if (not np.isnan(r) and r != None)]
        # Update dictionary of all test residuals
        res['initial']['by_lcdb'][i] = np.append(res['initial']['by_lcdb'][i], res_initial_byclass)
        res['rf']['by_lcdb'][i] = np.append(res['rf']['by_lcdb'][i], res_rf_byclass)
        res['dn']['by_lcdb'][i] = np.append(res['dn']['by_lcdb'][i], res_dn_byclass)
        res['cn']['by_lcdb'][i] = np.append(res['cn']['by_lcdb'][i], res_cn_byclass)
    
    # Class residuals by NIWA's Flood Hazard susceptibility map
    # Loop through each potential land cover class (as defined in proc_LCDB.py) and calculate elevation residuals for that particular class
    for flha_code, flha_label in zip([1,0], ['flood','noflood']):
        # Get lists of residuals for that flood susceptibility - for each of the input residual arrays
        res_initial_byflha = diff[flha==flha_code].flatten().tolist()
        res_rf_byflha = rf_res[flha==flha_code].flatten().tolist()
        res_dn_byflha = dn_res[flha==flha_code].flatten().tolist()
        res_cn_byflha = cn_res[flha==flha_code].flatten().tolist()
        # Filter out any None values (masked)
        res_initial_byflha = [r for r in res_initial_byflha if (not np.isnan(r) and r != None)]
        res_rf_byflha = [r for r in res_rf_byflha if (not np.isnan(r) and r != None)]
        res_dn_byflha = [r for r in res_dn_byflha if (not np.isnan(r) and r != None)]
        res_cn_byflha = [r for r in res_cn_byflha if (not np.isnan(r) and r != None)]
        # Update dictionary of all test residuals
        res['initial']['by_flha'][flha_label] = np.append(res['initial']['by_flha'][flha_label], res_initial_byflha)
        res['rf']['by_flha'][flha_label] = np.append(res['rf']['by_flha'][flha_label], res_rf_byflha)
        res['dn']['by_flha'][flha_label] = np.append(res['dn']['by_flha'][flha_label], res_dn_byflha)
        res['cn']['by_flha'][flha_label] = np.append(res['cn']['by_flha'][flha_label], res_cn_byflha)
    
    # Class residuals by HAND (height above nearest drainage) range
    # Define breaks for each of the five HAND ranges
    hand_breaks = [(0,2), (2,5), (5,10), (10,20), (20, max(50, np.nanmax(hand))+1)]
    # Loop through each HAND range
    for j, breaks in enumerate(hand_breaks):
        hand_class = 'hand_{}'.format(j+1)
        # Get lists of residuals corresponding to that range of HAND values - for each of the input residual arrays
        res_initial_byhand = diff[(hand >= breaks[0]) & (hand < breaks[1])].flatten().tolist()
        res_rf_byhand = rf_res[(hand >= breaks[0]) & (hand < breaks[1])].flatten().tolist()
        res_dn_byhand = dn_res[(hand >= breaks[0]) & (hand < breaks[1])].flatten().tolist()
        res_cn_byhand = cn_res[(hand >= breaks[0]) & (hand < breaks[1])].flatten().tolist()
        # Filter out any None values (masked)
        res_initial_byhand = [r for r in res_initial_byhand if (not np.isnan(r) and r != None)]
        res_rf_byhand = [r for r in res_rf_byhand if (not np.isnan(r) and r != None)]
        res_dn_byhand = [r for r in res_dn_byhand if (not np.isnan(r) and r != None)]
        res_cn_byhand = [r for r in res_cn_byhand if (not np.isnan(r) and r != None)]
        # Update dictionary of all test residuals
        res['initial']['by_hand'][hand_class] = np.append(res['initial']['by_hand'][hand_class], res_initial_byhand)
        res['rf']['by_hand'][hand_class] = np.append(res['rf']['by_hand'][hand_class], res_rf_byhand)
        res['dn']['by_hand'][hand_class] = np.append(res['dn']['by_hand'][hand_class], res_dn_byhand)
        res['cn']['by_hand'][hand_class] = np.append(res['cn']['by_hand'][hand_class], res_cn_byhand)

# Define some common properties for the boxplots of residuals
bp_width = 0.1
bp_offset = 0.2
bp_colours = ['dimgrey'] + [model_colours[m] for m in models]

# Define labels for the four test zones/subsets
test_zones_classes = ['All'] + test_zones
test_zones_labels = ['Overall (all test\nzones combined)', 'Wairau Plains East\n(Marlborough)', 'Wairau Valley\n(Marlborough)', 'Takaka\n(Tasman)']

# Define colours for the six HAND ranges
hand_cmap = cm.Blues_r
hand_colours = [hand_cmap(i/5) for i in range(5)]


# 6a. Set up a figure summarising all residuals by test zone & LCDB class
fig, axes = plt.subplots(nrows=2, figsize=(9,9))

# Plot 1: Residuals by test zone
for i, test_zone in enumerate(test_zones_classes):
    i += 1
    # Add boxplots to the figure
    bps = axes[0].boxplot([res['initial']['by_zone'][test_zone], res['rf']['by_zone'][test_zone], res['dn']['by_zone'][test_zone], res['cn']['by_zone'][test_zone]], positions=[i-1.5*bp_offset, i-0.5*bp_offset, i+0.5*bp_offset, i+1.5*bp_offset], showfliers=False, medianprops={'color':'black'}, widths=bp_width, patch_artist=True)
    for patch, colour in zip(bps['boxes'], bp_colours):
        patch.set_facecolor(colour)
    # Get RMSE associated with each set of residuals
    RMSE_initial = np.sqrt(np.nanmean(np.square(res['initial']['by_zone'][test_zone])))
    RMSE_rf = np.sqrt(np.nanmean(np.square(res['rf']['by_zone'][test_zone])))
    RMSE_dn = np.sqrt(np.nanmean(np.square(res['dn']['by_zone'][test_zone])))
    RMSE_cn = np.sqrt(np.nanmean(np.square(res['cn']['by_zone'][test_zone])))
    # Calculate percentage improvement made by method
    RMSE_rf_improve = (RMSE_initial - RMSE_rf)/RMSE_initial * 100.
    RMSE_dn_improve = (RMSE_initial - RMSE_dn)/RMSE_initial * 100.
    RMSE_cn_improve = (RMSE_initial - RMSE_cn)/RMSE_initial * 100.
    # Determine which method achieved the highest improvement & label that point
    RMSEs_ordered = [RMSE_rf, RMSE_dn, RMSE_cn]
    best_index = np.argmin(RMSEs_ordered)
    best_RMSE = RMSEs_ordered[best_index]
    best_improve = [RMSE_rf_improve, RMSE_dn_improve, RMSE_cn_improve][best_index]
    best_model = models[best_index]
    best_colour = label_colours[best_model]
    axes[0].annotate(u'RMSE \u2193\nby {:.0f}%'.format(best_improve), xy=(i/4 - 1/8, 0.98), xycoords='axes fraction', ha='center', va='top', size=10, color=best_colour)
# Add axis ticks & labels, and turn some spines off
axes[0].set_xticks(range(1,5))
axes[0].set_xticklabels(test_zones_labels)
axes[0].set_xticklabels(['{}\n[{:,} pixels]'.format(label, len(res['initial']['by_zone'][z])) for label, z in zip(test_zones_labels, test_zones_classes)])
axes[0].set_ylabel('Residuals before/after correction [m]')
[axes[0].spines[edge].set_visible(False) for edge in ['top','right']]
# Colour background based on land cover class & add a horizontal line for zero error
[axes[0].axvspan(j-0.45, j+0.45, alpha=0.25, facecolor='lightgrey', edgecolor='none') for j in range(1,5)]
axes[0].axhline(y=0, linestyle='dashed', color='black', linewidth=0.8, alpha=0.3)
axes[0].set_xlim(0.5,4.5)
# Add title
axes[0].set_title('a) By test zone/subset', x=0, ha='left', color='dimgrey', weight='bold', alpha=0.8)

# Plot 2: Residuals by land cover class (LCDB), as defined in geo_process_LCDB.py
for j in range(1,8):
    # Add boxplots to the figure
    bps = axes[1].boxplot([res['initial']['by_lcdb'][j], res['rf']['by_lcdb'][j], res['dn']['by_lcdb'][j], res['cn']['by_lcdb'][j]], positions=[j-1.5*bp_offset, j-0.5*bp_offset, j+0.5*bp_offset, j+1.5*bp_offset], showfliers=False, medianprops={'color':'black'}, widths=bp_width, patch_artist=True)
    for patch, colour in zip(bps['boxes'], bp_colours):
        patch.set_facecolor(colour)
    # Get RMSE associated with each set of residuals
    RMSE_initial = np.sqrt(np.nanmean(np.square(res['initial']['by_lcdb'][j])))
    RMSE_rf = np.sqrt(np.nanmean(np.square(res['rf']['by_lcdb'][j])))
    RMSE_dn = np.sqrt(np.nanmean(np.square(res['dn']['by_lcdb'][j])))
    RMSE_cn = np.sqrt(np.nanmean(np.square(res['cn']['by_lcdb'][j])))
    # Calculate percentage improvement made by method
    RMSE_rf_improve = (RMSE_initial - RMSE_rf)/RMSE_initial * 100.
    RMSE_dn_improve = (RMSE_initial - RMSE_dn)/RMSE_initial * 100.
    RMSE_cn_improve = (RMSE_initial - RMSE_cn)/RMSE_initial * 100.
    # Determine which method achieved the highest improvement & label that point
    RMSEs_ordered = [RMSE_rf, RMSE_dn, RMSE_cn]
    best_index = np.argmin(RMSEs_ordered)
    best_RMSE = RMSEs_ordered[best_index]
    best_improve = [RMSE_rf_improve, RMSE_dn_improve, RMSE_cn_improve][best_index]
    best_model = models[best_index]
    best_colour = label_colours[best_model]
    axes[1].annotate(u'RMSE \u2193\nby {:.0f}%'.format(best_improve), xy=(j/7 - 1/14, 0.98), xycoords='axes fraction', ha='center', va='top', size=10, color=best_colour)
# Add axis ticks & labels, and turn some spines off
axes[1].set_xticks(range(1,8))
axes[1].set_xticklabels(['{}\n[{:,} pixels]'.format(diff_by_landcover[j]['label'], len(res['initial']['by_lcdb'][j])) for j in range(1,8)])
axes[1].set_ylabel('Residuals before/after correction [m]')
[axes[1].spines[edge].set_visible(False) for edge in ['top','right']]
# Colour background based on land cover class & add a horizontal line for zero error
[axes[1].axvspan(j-0.45, j+0.45, alpha=0.1, facecolor=diff_by_landcover[j]['colour'], edgecolor='none') for j in range(1,8)]
axes[1].axhline(y=0, linestyle='dashed', color='black', linewidth=0.8, alpha=0.3)
axes[1].set_xlim(0.5,7.5)
# Add title
axes[1].set_title('b) By land cover class', x=0, ha='left', color='dimgrey', weight='bold', alpha=0.8)

# Tighten up layout, add a common legend at the top & save figure
fig.tight_layout(pad=0, h_pad=1.8)
fig.subplots_adjust(top=0.93)
fig.legend(handles=bps['boxes'], labels=['Initial','RF','DCN','FCN'], frameon=False, ncol=4, loc='upper center', prop={'size':11})
fig.savefig('{}/residuals_boxplots_by_zone-lcdb.png'.format(folder_fig), dpi=300, bbox='tight')
plt.close()


# 6b. Set up a figure summarising all residuals by FLHA & HAND zones
fig, axes = plt.subplots(nrows=2, figsize=(9,9))

# Plot 1: Residuals by FLHA zone
for i, flha_label in enumerate(['flood','noflood']):
    i += 1
    # Add boxplots to the figure
    bps = axes[0].boxplot([res['initial']['by_flha'][flha_label], res['rf']['by_flha'][flha_label], res['dn']['by_flha'][flha_label], res['cn']['by_flha'][flha_label]], positions=[i-1.5*bp_offset, i-0.5*bp_offset, i+0.5*bp_offset, i+1.5*bp_offset], showfliers=False, medianprops={'color':'black'}, widths=bp_width, patch_artist=True)
    for patch, colour in zip(bps['boxes'], bp_colours):
        patch.set_facecolor(colour)
    # Get RMSE associated with each set of residuals
    RMSE_initial = np.sqrt(np.nanmean(np.square(res['initial']['by_flha'][flha_label])))
    RMSE_rf = np.sqrt(np.nanmean(np.square(res['rf']['by_flha'][flha_label])))
    RMSE_dn = np.sqrt(np.nanmean(np.square(res['dn']['by_flha'][flha_label])))
    RMSE_cn = np.sqrt(np.nanmean(np.square(res['cn']['by_flha'][flha_label])))
    # Calculate percentage improvement made by method
    RMSE_rf_improve = (RMSE_initial - RMSE_rf)/RMSE_initial * 100.
    RMSE_dn_improve = (RMSE_initial - RMSE_dn)/RMSE_initial * 100.
    RMSE_cn_improve = (RMSE_initial - RMSE_cn)/RMSE_initial * 100.
    # Determine which method achieved the highest improvement & label that point
    RMSEs_ordered = [RMSE_rf, RMSE_dn, RMSE_cn]
    best_index = np.argmin(RMSEs_ordered)
    best_RMSE = RMSEs_ordered[best_index]
    best_improve = [RMSE_rf_improve, RMSE_dn_improve, RMSE_cn_improve][best_index]
    best_model = models[best_index]
    best_colour = label_colours[best_model]
    axes[0].annotate(u'RMSE \u2193\nby {:.0f}%'.format(best_improve), xy=(i/2 - 1/4, 0.98), xycoords='axes fraction', ha='center', va='top', size=10, color=best_colour)
# Add axis ticks & labels, and turn some spines off
axes[0].set_xticks(range(1,3))
axes[0].set_xticklabels(['Flood-prone\n[{:,} pixels]'.format(len(res['initial']['by_flha']['flood'])), 'Not flood-prone\n({:,} pixels)'.format(len(res['initial']['by_flha']['noflood']))])
axes[0].set_ylabel('Residuals before/after correction [m]')
[axes[0].spines[edge].set_visible(False) for edge in ['top','right']]
# Colour background based on flood proneness & add a horizontal line for zero error
[axes[0].axvspan(j-0.48+1, j+0.48+1, alpha=0.08, facecolor=flha_colour, edgecolor='none') for j, flha_colour in enumerate(['red','green'])]
axes[0].axhline(y=0, linestyle='dashed', color='black', linewidth=0.8, alpha=0.3)
axes[0].set_xlim(0.5,2.5)
# Add legend & title
axes[0].set_title('a) By flood susceptibility, based on resampled NIWA raster [39]', x=0, ha='left', color='dimgrey', weight='bold', alpha=0.8)

# Plot 2: Residuals by HAND range
for k in range(1,6):
    # Add boxplots to the figure
    h = 'hand_{}'.format(k)
    bps = axes[1].boxplot([res['initial']['by_hand'][h], res['rf']['by_hand'][h], res['dn']['by_hand'][h], res['cn']['by_hand'][h]], positions=[k-1.5*bp_offset, k-0.5*bp_offset, k+0.5*bp_offset, k+1.5*bp_offset], showfliers=False, medianprops={'color':'black'}, widths=bp_width, patch_artist=True)
    for patch, colour in zip(bps['boxes'], bp_colours):
        patch.set_facecolor(colour)
    # Get RMSE associated with each set of residuals
    RMSE_initial = np.sqrt(np.nanmean(np.square(res['initial']['by_hand'][h])))
    RMSE_rf = np.sqrt(np.nanmean(np.square(res['rf']['by_hand'][h])))
    RMSE_dn = np.sqrt(np.nanmean(np.square(res['dn']['by_hand'][h])))
    RMSE_cn = np.sqrt(np.nanmean(np.square(res['cn']['by_hand'][h])))
    # Calculate percentage improvement made by method
    RMSE_rf_improve = (RMSE_initial - RMSE_rf)/RMSE_initial * 100.
    RMSE_dn_improve = (RMSE_initial - RMSE_dn)/RMSE_initial * 100.
    RMSE_cn_improve = (RMSE_initial - RMSE_cn)/RMSE_initial * 100.
    # Determine which method achieved the highest improvement & label that point
    RMSEs_ordered = [RMSE_rf, RMSE_dn, RMSE_cn]
    best_index = np.argmin(RMSEs_ordered)
    best_RMSE = RMSEs_ordered[best_index]
    best_improve = [RMSE_rf_improve, RMSE_dn_improve, RMSE_cn_improve][best_index]
    best_model = models[best_index]
    best_colour = label_colours[best_model]
    axes[1].annotate(u'RMSE \u2193\nby {:.0f}%'.format(best_improve), xy=(k/5 - 1/10, 0.98), xycoords='axes fraction', ha='center', va='top', size=10, color=best_colour)
# Add axis ticks & labels, and turn some spines off
axes[1].set_xticks(range(1,6))
axes[1].set_xticklabels(['{}\n[{:,} pixels]'.format(label, len(res['initial']['by_hand'][h])) for label, h in zip(['0 - 2 m','2 - 5 m','5 - 10 m','10 - 20 m','> 20 m'], ['hand_1','hand_2','hand_3','hand_4','hand_5'])])
axes[1].set_ylabel('Residuals before/after correction [m]')
[axes[1].spines[edge].set_visible(False) for edge in ['top','right']]
# Colour background based on graded blues & add a horizontal line for zero error
[axes[1].axvspan(k-0.45, k+0.45, alpha=0.1, facecolor=hand_colours[k-1], edgecolor='none') for k in range(1,6)]
axes[1].axhline(y=0, linestyle='dashed', color='black', linewidth=0.8, alpha=0.3)
axes[1].set_xlim(0.5,5.5)
# Add legend & title
axes[1].set_title('b) By height above nearest drainage (HAND) [68]', x=0, ha='left', color='dimgrey', weight='bold', alpha=0.8)

# Tighten up layout, add a common legend at the top & save figure
fig.tight_layout(pad=0, h_pad=1.8)
fig.subplots_adjust(top=0.93)
fig.legend(handles=bps['boxes'], labels=['Initial','RF','DCN','FCN'], frameon=False, ncol=4, loc='upper center', prop={'size':11})
fig.savefig('{}/residuals_boxplots_by_flha-hand.png'.format(folder_fig), dpi=300, bbox='tight')
plt.close()


###############################################################################
# 7. Check distribution of land cover, elevations & slopes in each dataset    #
###############################################################################

# Read in 1D inputs for each dataset (training, validation, testing)
vectors_train = pd.read_csv('{}/Input1D_Train.csv'.format(folder_input_1D))
vectors_dev = pd.read_csv('{}/Input1D_Dev.csv'.format(folder_input_1D))
vectors_test = pd.read_csv('{}/Input1D_Test.csv'.format(folder_input_1D))

# 7a. Generate plot showing distribution of elevations & slope (by SRTM) for each dataset
fig, axes = plt.subplots(nrows=2, figsize=(8, 7))
# Extract elevation values from each dataset
z_train = vectors_train['srtm_z'].values
z_dev = vectors_dev['srtm_z'].values
z_test = vectors_test['srtm_z'].values
# Extract slope values from each dataset
slope_train = vectors_train['srtm_slope'].values
slope_dev = vectors_dev['srtm_slope'].values
slope_test = vectors_test['srtm_slope'].values
# Add elevation data as histograms
axes[0].hist([z_train, z_dev, z_test], color=dataset_colours, label=['Train','Validation','Test'], linewidth=2, alpha=0.5, bins=50, density=True, histtype='step', log=True)
axes[0].set_xlabel('SRTM Elevations [m]')
axes[0].set_ylabel('Relative frequency (log-scale)')
axes[0].set_title('a) Distribution of elevation values, by input dataset', x=0, size=10, ha='left', color='dimgrey', weight='bold', alpha=0.8)
[axes[0].spines[edge].set_visible(False) for edge in ['top','right']]
h0, l0 = axes[0].get_legend_handles_labels()
axes[0].legend(h0[::-1], l0[::-1], frameon=False, loc='upper right')
# Add slope data as histograms
axes[1].hist([slope_train, slope_dev, slope_test], color=dataset_colours, label=['Train','Validation','Test'], linewidth=2, alpha=0.5, bins=50, density=True, histtype='step', log=True)
axes[1].set_xlabel('Slopes - derived from SRTM elevations [%]')
axes[1].set_ylabel('Relative frequency (log-scale)')
axes[1].set_title('b) Distribution of slope values, by input dataset', x=0, size=10, ha='left', color='dimgrey', weight='bold', alpha=0.8)
[axes[1].spines[edge].set_visible(False) for edge in ['top','right']]
h1, l1 = axes[1].get_legend_handles_labels()
axes[1].legend(h1[::-1], l1[::-1], frameon=False, loc='upper right')
# Save figure
fig.tight_layout(h_pad=1.8)
fig.savefig('{}/inputdata_distribution_srtm_z-slope.png'.format(folder_fig), dpi=300)
plt.close()


# 7b. Generate plot showing distribution of land cover groups for each dataset
fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8,5))
lc_colours = [diff_by_landcover[i]['colour'] for i in range(1,8)]
lc_labels = [diff_by_landcover[i]['label'] for i in range(1,8)]
# Plot 1: Training data
lc_train_counts = [len(vectors_train[vectors_train['lcdb']==i].index) for i in range(1,8)]
lc_train_freq = [count/sum(lc_train_counts) for count in lc_train_counts]
axes[0].bar(x=range(1,8), height=lc_train_freq, color=lc_colours, alpha=0.7)
[axes[0].annotate('{:.1f}%'.format(freq*100), xy=(i, freq), ha='center', va='bottom') for i, freq in zip(range(1,8), lc_train_freq)]
axes[0].annotate('a) Training dataset', xy=(0.02, 0.98), xycoords='axes fraction', ha='left', va='top', fontweight='bold', color='dimgrey', alpha=0.8)
# Plot 2: Validation data
lc_dev_counts = [len(vectors_dev[vectors_dev['lcdb']==i].index) for i in range(1,8)]
lc_dev_freq = [count/sum(lc_dev_counts) for count in lc_dev_counts]
axes[1].bar(x=range(1,8), height=lc_dev_freq, color=lc_colours, alpha=0.7)
[axes[1].annotate('{:.1f}%'.format(freq*100), xy=(i, freq), ha='center', va='bottom') for i, freq in zip(range(1,8), lc_dev_freq)]
axes[1].annotate('b) Validation dataset', xy=(0.02, 0.98), xycoords='axes fraction', ha='left', va='top', fontweight='bold', color='dimgrey', alpha=0.8)
# Plot 3: Testing data
lc_test_counts = [len(vectors_test[vectors_test['lcdb']==i].index) for i in range(1,8)]
lc_test_freq = [count/sum(lc_test_counts) for count in lc_test_counts]
axes[2].bar(x=range(1,8), height=lc_test_freq, color=lc_colours, alpha=0.7)
[axes[2].annotate('{:.1f}%'.format(freq*100), xy=(i, freq), ha='center', va='bottom') for i, freq in zip(range(1,8), lc_test_freq)]
axes[2].annotate('c) Testing dataset', xy=(0.02, 0.98), xycoords='axes fraction', ha='left', va='top', fontweight='bold', color='dimgrey', alpha=0.8)
# Add figure labels, etc
[[[axes[i].spines[edge].set_visible(False)] for edge in ['top','right']] for i in [0,1,2]]
[axes[i].set_xticks(range(1,8)) for i in [0,1,2]]
[axes[i].set_ylabel('Frequency') for i in [0,1,2]]
axes[2].set_xticklabels(lc_labels)
fig.tight_layout()
fig.savefig('{}/inputdata_distribution_lcdb.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 8. 'Graphical abstract' image for Remote Sensing journal requirements       #
###############################################################################

# Instructions from Remote Sensing journal
#  - The minimum required size for the GA is 560 × 1100 pixels (height × width)
#  - When submitting larger images, please make sure to keep to the same ratio
#  - High-quality illustration or diagram (PNG, JPEG, EPS, SVG, PSD or AI)
#  - Written text in a GA should be clear and easy to read, using Times/Arial/Courier/Helvetica/Ubuntu/Calibri

# Set figure height & width based on required dimensions
width = 16
height = 560/1100 * width

# Read in MRL18_WPE data for the convolutional network results (ONLY test dataset pixels)
zone = 'MRL18_WPE'
cn_predictions = np.load('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results_convnet, zone))
cn_cor, cn_elv, cn_res, dtm, srtm, diff, merit, flha, hand = process_2D_predictions(zone, cn_predictions, 'test_clip', no_data=-9999)

# Set up grid for 3D plot
ny, nx = cn_elv.shape
x = range(nx)
y = range(ny)
X, Y = np.meshgrid(x, y)

# Develop consistent colourmap (same as that used previously for this zone)
elv_min, elv_max = test_zones_props[zone]['elv_cbar_range']
elv_cmap = cm.terrain
elv_cmap.set_bad(color='whitesmoke')
elv_norm = colors.Normalize(vmin=elv_min, vmax=elv_max)

# Calculate RMSE for the SRTM & FCN-corrected SRTM (compared to DTM)
RMSE_srtm = np.sqrt(np.mean(np.square(diff)))
RMSE_fcn = np.sqrt(np.mean(np.square(cn_res)))
improve = (RMSE_srtm-RMSE_fcn)/RMSE_srtm * 100

# Define plotting parameters for the 3D visualisation
stride = 1
offset = 50

# Define colours to be used for map inset
colour_land = (215/255, 194/255, 158/255)
colour_ocean = (190/255, 232/255, 255/255)


# 8a. 3D terrain visualisations, with horizontal bar charts summarising change in RMSE
fig = plt.figure(figsize=(width, height))
gs = gridspec.GridSpec(3, 2, width_ratios=[6, 1], height_ratios=[1,1,1])
ax1 = fig.add_subplot(gs[0], projection='3d')
ax2 = fig.add_subplot(gs[2], projection='3d')
ax3 = fig.add_subplot(gs[4], projection='3d')
ax4_hold = fig.add_subplot(gs[1])
ax5_hold = fig.add_subplot(gs[3])
ax6_hold = fig.add_subplot(gs[5])

# Top: DTM
ax1.plot_surface(X, Y, dtm, cmap=elv_cmap, norm=elv_norm, linewidth=0, antialiased=False, rstride=stride, cstride=stride)
# Middle: SRTM
ax2.plot_surface(X, Y, srtm, cmap=elv_cmap, norm=elv_norm, linewidth=0, antialiased=False, rstride=stride, cstride=stride)
# Bottom: FCN-corrected SRTM
ax3.plot_surface(X, Y, cn_elv, cmap=elv_cmap, norm=elv_norm, linewidth=0, antialiased=False, rstride=stride, cstride=stride)
# General properties to be applied to each ax
for ax in [ax1, ax2, ax3]:
    ax.set_xlim((0,nx))
    ax.set_ylim((0,ny))
    ax.set_zlim((0,20))
    ax.view_init(65, 70)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
# Leave the others empty - essentially just placeholders to keep space open for the map & violin plots
for ax in [ax4_hold, ax5_hold, ax6_hold]:
    ax.set_axis_off()
fig.tight_layout(pad=0, h_pad=-7, w_pad=-7)
# Add axes for the horizontal bar chart
ax_bars = fig.add_axes([0.81, 0.1, 0.19, 0.58])
ax_bars.barh(y=[0,1], width=[RMSE_fcn, RMSE_srtm], height=0.7, color='firebrick', alpha=0.5)
ax_bars.grid(axis='x', which='major', color='dimgrey', alpha=0.25)
ax_bars.set_xlabel('Root Mean Square Error [m]', size=13)
ax_bars.tick_params(axis='x', which='both', labelsize=11)
ax_bars.set_yticks([0,1])
ax_bars.set_yticklabels(['Corrected\nSRTM','Original\nSRTM'], size=13)
[ax_bars.spines[side].set_visible(False) for side in ['top','right','bottom']]
ax_bars.set_title('Impact of correcting SRTM\nusing model predictions', size=14)
ax_bars.annotate('Applying corrections\npredicted by model\nreduces RMSE from\n{:.2f}m to {:.2}m\n(-{:.1f}%)'.format(RMSE_srtm, RMSE_fcn, improve), xy=(1.05, 0.35), xycoords='data', ha='left', va='top', color='firebrick', size=12)
ax_bars.annotate('', xy=(-0.6, 0.23), xycoords='axes fraction', xytext=(-0.35, 0.23), arrowprops=dict(width=3, headwidth=3, headlength=1, linestyle='--', fc='dimgrey', ec='none', alpha=0.5), ha='left', va='center')
ax_bars.annotate('', xy=(-0.6, 0.77), xycoords='axes fraction', xytext=(-0.3, 0.77), arrowprops=dict(width=3, headwidth=3, headlength=1, linestyle='--', fc='dimgrey', ec='none', alpha=0.5), ha='left', va='center')

# Add axes for the inset map
ax_map = fig.add_axes([0.91, 0.79, 0.09, 0.2])
m = Basemap(projection='merc', resolution='h', llcrnrlat=-48.1, urcrnrlat=-33.6, llcrnrlon=164.9, urcrnrlon=179.9, ax=ax_map)
m.drawcoastlines(linewidth=0.1, color='none')
m.drawmapboundary(fill_color=colour_ocean, color='none')
m.fillcontinents(color=colour_land, lake_color=colour_land)
# Map coordinates for test zone to (x, y) for plotting
zone_x, zone_y = m(174.0, -41.4743)
ax_map.plot(zone_x, zone_y, marker='o', markeredgecolor='black', markeredgewidth=1, markerfacecolor='white', markersize=5)
plt.annotate('Test zone (data unseen during model training)', xy=(0.91, 0.985), xycoords='figure fraction', ha='right', va='top', size=14)
plt.annotate('Wairau Plains East, Marlborough Region\nAotearoa New Zealand', xy=(0.91, 0.95), xycoords='figure fraction', ha='right', va='top', size=13, color='dimgrey')

# Add general explanatory labels
plt.annotate('We trained a fully-convolutional neural network to convert a Digital Surface Model (DSM)\nto a Digital Terrain Model (DTM), by predicting vertical biases due to vegetation and\nbuilt-up areas, using two DSMs, multi-spectral imagery (Landsat-7 & -8),\nnight-time light, and maps of forest cover & canopy height\n(all free & global datasets).', xy=(0.01, 0.985), xycoords='figure fraction', ha='left', va='top', size=14)
plt.annotate('a) DTM (LiDAR)\nReference data', xy=(0.01, 0.72), xycoords='figure fraction', ha='left', va='bottom', size=14)
plt.annotate('LiDAR-derived DTM, resampled from\n1m resolution to match SRTM grid (~23m),\nwith river channel clearly visible.', xy=(0.01, 0.63), xycoords='figure fraction', ha='left', va='bottom', size=14, color='dimgrey')
plt.annotate('', xy=(0.235, 0.732), xycoords='figure fraction', xytext=(0.107, 0.732), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')
plt.annotate('b) DSM (SRTM)\nBefore correction', xy=(0.01, 0.461), xycoords='figure fraction', ha='left', va='bottom', size=14)
plt.annotate("Original SRTM DSM, showing large\nvertical biases along river channel and\nrandom 'speckle' noise spread across the\nfloodplain.", xy=(0.01, 0.345), xycoords='figure fraction', ha='left', va='bottom', size=14, color='dimgrey')
plt.annotate('', xy=(0.205, 0.475), xycoords='figure fraction', xytext=(0.12, 0.475), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')
plt.annotate('c) DSM (SRTM)\nAfter correction', xy=(0.01, 0.17), xycoords='figure fraction', ha='left', va='bottom', size=14)
plt.annotate('SRTM after applying corrections\npredicted by the fully-convolutional\nneural network, with a {:.0f}% reduction\nin RMSE and river channels better resolved.'.format(improve), xy=(0.01, 0.05), xycoords='figure fraction', ha='left', va='bottom', size=14, color='dimgrey')
plt.annotate('', xy=(0.2, 0.182), xycoords='figure fraction', xytext=(0.11, 0.182), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')

# Save figure
fig.savefig('{}/graphical_abstract_barh.png'.format(folder_fig), dpi=300)
plt.close()


# 8b. 3D terrain visualisations, with horizontal boxplots summarising change in residuals
fig = plt.figure(figsize=(width, height))
gs = gridspec.GridSpec(3, 2, width_ratios=[6, 1], height_ratios=[1,1,1])
ax1 = fig.add_subplot(gs[0], projection='3d')
ax2 = fig.add_subplot(gs[2], projection='3d')
ax3 = fig.add_subplot(gs[4], projection='3d')
ax4_hold = fig.add_subplot(gs[1])
ax5_hold = fig.add_subplot(gs[3])
ax6_hold = fig.add_subplot(gs[5])

# Top: DTM
ax1.plot_surface(X, Y, dtm, cmap=elv_cmap, norm=elv_norm, linewidth=0, antialiased=False, rstride=stride, cstride=stride)
# Middle: SRTM
ax2.plot_surface(X, Y, srtm, cmap=elv_cmap, norm=elv_norm, linewidth=0, antialiased=False, rstride=stride, cstride=stride)
# Bottom: FCN-corrected SRTM
ax3.plot_surface(X, Y, cn_elv, cmap=elv_cmap, norm=elv_norm, linewidth=0, antialiased=False, rstride=stride, cstride=stride)
# General properties to be applied to each ax
for ax in [ax1, ax2, ax3]:
    ax.set_xlim((0,nx))
    ax.set_ylim((0,ny))
    ax.set_zlim((0,20))
    ax.view_init(65, 70)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
# Leave the others empty - essentially just placeholders to keep space open for the map & violin plots
for ax in [ax4_hold, ax5_hold, ax6_hold]:
    ax.set_axis_off()
fig.tight_layout(pad=0, h_pad=-7, w_pad=-7)
# Add axes for the boxplots
ax_box = fig.add_axes([0.77, 0.1, 0.22, 0.58])
ax_box.boxplot(x=[cn_res.flatten(), diff.flatten()], positions=[0,1.1], boxprops={'linewidth':1.5}, whiskerprops={'linewidth':1.5}, medianprops={'color':'white', 'linewidth':2}, vert=False, showfliers=False)
# Add jittered points to plot
diff_jitter = np.random.normal(loc=1.1, scale=0.08, size=diff.size)
ax_box.scatter(diff.flatten(), diff_jitter, color='steelblue', s=8, alpha=0.15)
cn_res_jitter = np.random.normal(loc=0.0, scale=0.08, size=cn_res.size)
ax_box.scatter(cn_res.flatten(), cn_res_jitter, color='steelblue', s=8, alpha=0.15)
ax_box.set_yticks([])
[ax_box.spines[side].set_visible(False) for side in ['left','top','right']]
ax_box.set_title('Change in residuals\nafter applying corrections', size=14)
ax_box.grid(axis='x', which='major', color='dimgrey', alpha=0.25)
ax_box.set_xlabel('Residuals [m]\n(compared to reference DTM)', size=13)
ax_box.tick_params(axis='x', which='both', labelsize=11)
ax_box.patch.set_alpha(0)
ax_box.annotate('', xy=(0, 0.23), xycoords='axes fraction', xytext=(-0.33, 0.23), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')
ax_box.annotate('', xy=(0, 0.76), xycoords='axes fraction', xytext=(-0.33, 0.76), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')
ax_box.annotate('Applying corrections predicted\nby model reduces RMSE from\n{:.2f}m to {:.2}m (-{:.1f}%)'.format(RMSE_srtm, RMSE_fcn, improve), xy=(-2, 0.66), xycoords='data', ha='left', va='top', color='dimgrey', size=13)

# Add axes for the inset map
ax_map = fig.add_axes([0.91, 0.79, 0.09, 0.2])
m = Basemap(projection='merc', resolution='h', llcrnrlat=-48.1, urcrnrlat=-33.6, llcrnrlon=164.9, urcrnrlon=179.9, ax=ax_map)
m.drawcoastlines(linewidth=0.1, color='none')
m.drawmapboundary(fill_color=colour_ocean, color='none')
m.fillcontinents(color=colour_land, lake_color=colour_land)
# Map coordinates for test zone to (x, y) for plotting
zone_x, zone_y = m(174.0, -41.4743)
ax_map.plot(zone_x, zone_y, marker='o', markeredgecolor='black', markeredgewidth=1, markerfacecolor='white', markersize=5)
plt.annotate('Test zone (data unseen during model training)', xy=(0.91, 0.985), xycoords='figure fraction', ha='right', va='top', size=14)
plt.annotate('Wairau Plains East, Marlborough Region\nAotearoa New Zealand', xy=(0.91, 0.95), xycoords='figure fraction', ha='right', va='top', size=13, color='dimgrey')

# Add general explanatory labels
plt.annotate('We trained a fully-convolutional neural network to convert a Digital Surface Model (DSM)\nto a Digital Terrain Model (DTM), by predicting vertical biases due to vegetation and\nbuilt-up areas, using two DSMs, multi-spectral imagery (Landsat-7 & -8),\nnight-time light, and maps of forest cover & canopy height\n(all free & global datasets).', xy=(0.01, 0.985), xycoords='figure fraction', ha='left', va='top', size=14)
plt.annotate('a) DTM (LiDAR)\nReference data', xy=(0.01, 0.72), xycoords='figure fraction', ha='left', va='bottom', size=14)
plt.annotate('LiDAR-derived DTM, resampled from\n1m resolution to match SRTM grid (~23m),\nwith river channel clearly visible.', xy=(0.01, 0.63), xycoords='figure fraction', ha='left', va='bottom', size=14, color='dimgrey')
plt.annotate('', xy=(0.235, 0.732), xycoords='figure fraction', xytext=(0.107, 0.732), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')
plt.annotate('b) DSM (SRTM)\nBefore correction', xy=(0.01, 0.461), xycoords='figure fraction', ha='left', va='bottom', size=14)
plt.annotate("Original SRTM DSM, showing large\nvertical biases along river channel and\nrandom 'speckle' noise spread across the\nfloodplain.", xy=(0.01, 0.345), xycoords='figure fraction', ha='left', va='bottom', size=14, color='dimgrey')
plt.annotate('', xy=(0.205, 0.475), xycoords='figure fraction', xytext=(0.12, 0.475), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')
plt.annotate('c) DSM (SRTM)\nAfter correction', xy=(0.01, 0.17), xycoords='figure fraction', ha='left', va='bottom', size=14)
plt.annotate('SRTM after applying corrections\npredicted by the fully-convolutional\nneural network, with a {:.0f}% reduction\nin RMSE and river channels better resolved.'.format(improve), xy=(0.01, 0.05), xycoords='figure fraction', ha='left', va='bottom', size=14, color='dimgrey')
plt.annotate('', xy=(0.2, 0.182), xycoords='figure fraction', xytext=(0.11, 0.182), arrowprops=dict(fc='black', ec='none', width=1.5, headwidth=1.5, headlength=1, alpha=0.5), ha='left', va='center')

# Save figure
fig.savefig('{}/graphical_abstract_boxplots.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 9. Extract cross-sections as part of results illustration                   #
###############################################################################

# Define a function that takes a filepath (to a corrected-top GeoTIFF) & two lon-lat coordinate pairs, and returns the elevation profile between those two points
def get_cross_section(tif_path, point_a, point_b, n_points):
    
    # Read the raster into memory, extracting its geotransform & the inverse
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    array = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()              # maps raster grid to lon-lat coordinates
    gt_inv = gdal.InvGeoTransform(gt)      # maps lon-lat coordinates to raster grid
    ds = None
    
    # Calculate x & y steps between the points (in degrees)
    dx = (point_b[0] - point_a[0]) / n_points
    dy = (point_b[1] - point_a[1]) / n_points
    
    # Initialise a list to hold elevations extracted from the topography raster
    elevations = []
    
    # Loop through all steps along the line between the two points, getting the underlying raster value
    for i in range(n_points):
        # Get the step coordinates in terms of lon-lat
        step_coords = (point_a[0] + i*dx, point_a[1] + i*dy)
        # Convert these coordinates to array indices    
        array_x = int(gt_inv[0] + gt_inv[1]*step_coords[0] + gt_inv[2]*step_coords[1])
        array_y = int(gt_inv[3] + gt_inv[4]*step_coords[0] + gt_inv[5]*step_coords[1])
        elevation = array[array_y, array_x]
        elevations.append(elevation)
    return elevations

# Define a dictionary describing the cross-sections of interest, for each test zone
cs_dict = {'MRL18_WPE':{'A':{'start':(173.98786, -41.46714), 'end':(173.97545, -41.47835)},
                        'B':{'start':(174.00127, -41.47945), 'end':(173.99717, -41.48526)}},
           'MRL18_WVL':{'C':{'start':(173.31079, -41.61877), 'end':(173.31722, -41.63086)},
                        'D':{'start':(173.36445, -41.60312), 'end':(173.36751, -41.61092)}},
           'TSM16_ATG':{'E':{'start':(172.79475, -40.85864), 'end':(172.79853, -40.85309)},
                        'F':{'start':(172.79357, -40.82249), 'end':(172.79936, -40.82627)}}}

# Set up a figure to show two cross-sections for each test zone
fig, axes = plt.subplots(nrows=6, sharex=True, figsize=(4.4, 9.25))
# Define the number of points to extract for each section
n_points = 1000
# Loop through each cross-section in turn
for zone in test_zones:
    # Define a dictionary describing the topography datasets available
    dem_dict = {'srtm':{'label':'SRTM', 'colour':'black', 'lw':0.8, 'path':'{}/{}/SRTM_{}_Z.tif'.format(folder_srtm, zone, zone)},
                'dtm':{'label':'LiDAR DTM', 'colour':'red', 'lw':0.9, 'path':'{}/{}/DTM_{}_30m_Median.tif'.format(folder_dtm, zone, zone)},
                'rf':{'label':'RF', 'colour':label_colours['rf'], 'lw':1.2, 'path':'{}/TIF/rf_elevations_{}_test.tif'.format(folder_results_rf, zone)},
                'dcn':{'label':'DCN', 'colour':label_colours['dcn'], 'lw':1.2, 'path':'{}/TIF/densenet_elevations_{}_test.tif'.format(folder_results_densenet, zone)},
                'fcn':{'label':'FCN', 'colour':label_colours['fcn'], 'lw':1.2, 'path':'{}/TIF/convnet_elevations_{}_test.tif'.format(folder_results_convnet, zone)}}
    # Loop through the two cross-sections for each zone
    for i, cs in enumerate(['A','B','C','D','E','F']):
        if cs in cs_dict[zone].keys():
            # Get the start & end points of that cross-section
            cs_start = cs_dict[zone][cs]['start']
            cs_end = cs_dict[zone][cs]['end']
            # Loop through the DEMs available, adding its cross-section to the figure
            for dem in ['srtm','dtm','rf','dcn','fcn']:
                dem_path = dem_dict[dem]['path']
                dem_label = dem_dict[dem]['label']
                dem_colour = dem_dict[dem]['colour']
                dem_lw = dem_dict[dem]['lw']
                # Extract cross-section
                cs_elevs = get_cross_section(dem_path, cs_start, cs_end, n_points)
                # Add profile to appropriate axes
                axes[i].plot(cs_elevs, label=dem_label, color=dem_colour, linewidth=dem_lw)
            # Force a 5m interval in the y-axis ticks used
            yticks = axes[i].get_yticks()
            axes[i].yaxis.set_ticks(np.arange(yticks[1], yticks[-1], 5))
            # Turn off all spines, the x-axis & the y-axis tick marks
            axes[i].xaxis.set_visible(False)
            [axes[i].spines[edge].set_visible(False) for edge in ['top','right','bottom','left']]
            axes[i].yaxis.set_tick_params(length=0)
            axes[i].tick_params(axis='y', labelsize=7)
            # Add faded y-axis grid lines & y-axis label
            axes[i].grid(axis='y', which='major', color='dimgrey', alpha=0.1)
            axes[i].set_ylabel('Elevation [m]', fontsize=8)
            # Add annotations indicating section name/code
            axes[i].annotate(cs, xy=(0.05, 0.0), xycoords='axes fraction', ha='left', va='bottom', fontweight='bold', alpha=0.8)
            axes[i].annotate("{} '".format(cs), xy=(0.95, 0.0), xycoords='axes fraction', ha='right', va='bottom', fontweight='bold', alpha=0.8)
# Tighten layout & make space for the legend
fig.tight_layout(h_pad=1, w_pad=0)
plt.subplots_adjust(top=0.97)
# Add overall legend & align labels
legend_handles, legend_labels = axes[i-1].get_legend_handles_labels()
fig.legend(legend_handles, legend_labels, frameon=False, loc='upper center', ncol=5, columnspacing=1.5, handletextpad=0.3, prop={'size':8})
fig.align_labels()
# Save figure
fig.savefig('{}/results_sections.png'.format(folder_fig), dpi=300, bbox_inches='tight')
plt.close()


###############################################################################
# 10. Assess RMSE for SRTM 1-20m zone, following Kulp & Strauss (CoastalDEM)  #
###############################################################################

# Loop through each test zone
for zone in test_zones:
    
    # Import Random Forest arrays - ONLY test dataset pixels
    rf_predictions = np.load('{}/RF_Predictions_ByZone_{}.npy'.format(folder_results_rf, zone))
    rf_cor, rf_elv, rf_res, _,_,_,_,_,_ = process_1D_predictions(zone, rf_predictions, 'test_clip', no_data=-9999)
    
    # Import Densenet arrays - ONLY test dataset pixels
    dn_predictions = np.load('{}/densenet_Predictions_ByZone_{}.npy'.format(folder_results_densenet, zone))
    dn_cor, dn_elv, dn_res, _,_,_,_,_,_ = process_1D_predictions(zone, dn_predictions, 'test_clip', no_data=-9999)
    
    # Import Convnet arrays - ONLY test dataset pixels
    cn_predictions = np.load('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results_convnet, zone))
    cn_cor, cn_elv, cn_res, dtm, srtm, diff, merit, flha, hand = process_2D_predictions(zone, cn_predictions, 'test_clip', no_data=-9999)
    
    # Mask all arrays wherever no_data values are present
    srtm_m = np.ma.masked_equal(srtm, no_data)
    dtm_m = np.ma.masked_equal(dtm, no_data)
    diff_m = np.ma.masked_equal(diff, no_data)
    rf_elv_m = np.ma.masked_equal(rf_elv, no_data)
    dn_elv_m = np.ma.masked_equal(dn_elv, no_data)
    cn_elv_m = np.ma.masked_equal(cn_elv, no_data)
    
    # Process only if that zone's SRTM array contains elevations between 1-20 m
    if np.any(srtm[(srtm_m >= 1) & (srtm_m <= 20)]):
    
        # Filter above arrays to only include values for which SRTM is between 1 and 20 m
        srtm_m_f = srtm_m[np.where((srtm_m >= 1) & (srtm_m <= 20))]
        dtm_m_f = dtm_m[np.where((srtm_m >= 1) & (srtm_m <= 20))]
        diff__f = diff_m[np.where((srtm_m >= 1) & (srtm_m <= 20))]
        rf_elv_m_f = rf_elv_m[np.where((srtm_m >= 1) & (srtm_m <= 20))]
        dn_elv_m_f = dn_elv_m[np.where((srtm_m >= 1) & (srtm_m <= 20))]
        cn_elv_m_f = cn_elv_m[np.where((srtm_m >= 1) & (srtm_m <= 20))]
        
        # Get all RMSE values, for areas where SRTM is between 1 and 20m
        RMSE_initial = np.sqrt(np.mean((srtm_m_f - dtm_m_f)**2))
        RMSE_rf = np.sqrt(np.mean((rf_elv_m_f - dtm_m_f)**2))
        RMSE_dn = np.sqrt(np.mean((dn_elv_m_f - dtm_m_f)**2))
        RMSE_cn = np.sqrt(np.mean((cn_elv_m_f - dtm_m_f)**2))
        
        # Calculate improvements
        improv_rf = (RMSE_rf - RMSE_initial)/RMSE_initial * 100.
        improv_dn = (RMSE_dn - RMSE_initial)/RMSE_initial * 100.
        improv_cn = (RMSE_cn - RMSE_initial)/RMSE_initial * 100.
        
        # Print results
        print(zone)
        print('RMSE_initial:', RMSE_initial)
        print('RMSE_rf:', RMSE_rf)
        print('RMSE_dn:', RMSE_dn)
        print('RMSE_cn:', RMSE_cn)
        print('improv_rf:', improv_rf)
        print('improv_dn:', improv_dn)
        print('improv_cn:', improv_cn)

# Only two test zones have elevations within the 1-20m range (MRL18_WPE & TSM16_ATG)
# Results were found to be very similar to the overall results already presented

# MRL18_WPE
#  - RMSE_initial: 2.8066373
#  - RMSE_rf: 1.1405739737630056
#  - RMSE_dn: 0.9603291958237794
#  - RMSE_cn: 0.6547264
#  - improv_rf: -59.36154703747597
#  - improv_dn: -65.78363722936197
#  - improv_cn: -76.67220830917358
# TSM16_ATG
#  - RMSE_initial: 4.651809
#  - RMSE_rf: 2.0713059254273296
#  - RMSE_dn: 2.218630549176275
#  - RMSE_cn: 1.5238178
#  - improv_rf: -55.47311101011337
#  - improv_dn: -52.30607175887735
#  - improv_cn: -67.24247336387634


###############################################################################
# 11. Assess test zone RMSEs compared with MERIT DEM (Yamazaki et al. 2017)   #
###############################################################################

# Initalise dictionary to hold test residuals for each available DSM (compared to the DTM)
res = {'srtm':[], 'merit':[], 'rf':[], 'dn':[], 'cn':[]}

# Loop through each test zone, appending the residuals for each DSM source to the relevant list
for zone in test_zones:
    
    # Import Random Forest arrays - ONLY test dataset pixels
    rf_predictions = np.load('{}/RF_Predictions_ByZone_{}.npy'.format(folder_results_rf, zone))
    rf_cor, rf_elv, rf_res, _,_,_,_,_,_ = process_1D_predictions(zone, rf_predictions, 'test_clip', no_data=-9999)
    
    # Import Densenet arrays - ONLY test dataset pixels
    dn_predictions = np.load('{}/densenet_Predictions_ByZone_{}.npy'.format(folder_results_densenet, zone))
    dn_cor, dn_elv, dn_res, _,_,_,_,_,_ = process_1D_predictions(zone, dn_predictions, 'test_clip', no_data=-9999)
    
    # Import Convnet arrays - ONLY test dataset pixels
    cn_predictions = np.load('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results_convnet, zone))
    cn_cor, cn_elv, cn_res, dtm, srtm, diff, merit, flha, hand = process_2D_predictions(zone, cn_predictions, 'test_clip', no_data=-9999)
    
    # Get list of residuals for that zone & model
    res_srtm = (srtm - dtm).flatten().tolist()
    res_merit = (merit - dtm).flatten().tolist()
    res_rf = (rf_elv - dtm).flatten().tolist()
    res_dn = (dn_elv - dtm).flatten().tolist()
    res_cn = (cn_elv - dtm).flatten().tolist()
    
    # Filter out any None or nan values
    res_srtm = [r for r in res_srtm if (not np.isnan(r) and r != None)]
    res_merit = [r for r in res_merit if (not np.isnan(r) and r != None)]
    res_rf = [r for r in res_rf if (not np.isnan(r) and r != None)]
    res_dn = [r for r in res_dn if (not np.isnan(r) and r != None)]
    res_cn = [r for r in res_cn if (not np.isnan(r) and r != None)]
    
    # Update dictionary of all test residuals
    res['srtm'] = np.append(res['srtm'], res_srtm)
    res['merit'] = np.append(res['merit'], res_merit)
    res['rf'] = np.append(res['rf'], res_rf)
    res['dn'] = np.append(res['dn'], res_dn)
    res['cn'] = np.append(res['cn'], res_cn)


# Check that arrays are showing up as expected
fig, axes = plt.subplots(ncols=2, figsize=(9,6))
axes[0].imshow(merit, vmin=0, vmax=15)
axes[1].imshow(dtm, vmin=0, vmax=15)

# Calculate overall RMSE for each available DSM
RMSE_srtm = np.sqrt(np.nanmean((res['srtm'])**2))
RMSE_merit = np.sqrt(np.nanmean((res['merit'])**2))
RMSE_rf = np.sqrt(np.nanmean((res['rf'])**2))
RMSE_dn = np.sqrt(np.nanmean((res['dn'])**2))
RMSE_cn = np.sqrt(np.nanmean((res['cn'])**2))

# RMSE: Generate summary plots for the test dataset results
fig, axes = plt.subplots(figsize=(9,4.5))
axes.bar(range(5), [RMSE_srtm, RMSE_merit, RMSE_rf, RMSE_dn, RMSE_cn], color=dataset_colours[2], alpha=0.5)
axes.set_xticks(range(5))
axes.yaxis.set_tick_params(length=0)
axes.set_xticklabels(['SRTM','MERIT DEM\n[27]','RF\ncorrection','DCN\ncorrection','FCN\ncorrection'])
axes.set_ylabel('Root Mean Square Error [m]')
axes.grid(axis='y', which='major', color='dimgrey', alpha=0.1)
[axes.spines[edge].set_visible(False) for edge in ['left','top','right']]
# Add a horizontal line showing the initial error & a label
axes.axhline(y=RMSE_srtm, color=dataset_colours[2], linestyle='dashed', alpha=0.5)
axes.annotate('{:.3f}m'.format(RMSE_srtm), xy=(0, RMSE_srtm), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
# Add labels indicating improvement achieved by each method
for j, RMSE_new in enumerate([RMSE_merit, RMSE_rf, RMSE_dn, RMSE_cn]):
    # Add downward arrow from initial RMSE to improved RMSE
    axes.annotate('', xy=(j+1, RMSE_new), xytext=(j+1, RMSE_srtm), arrowprops=dict(arrowstyle='->'))
    # Add label indicating new RMSE and the percentage improvement it equates to
    improvement_percentage = (RMSE_new-RMSE_srtm)/RMSE_srtm * 100.
    axes.annotate('{:.3f}m\n({:.1f}%)'.format(RMSE_new, improvement_percentage), xy=(j+1, RMSE_new), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
axes.set_title('Performance on test dataset')
fig.tight_layout()
fig.savefig('{}/results_RMSE_MERIT.png'.format(folder_fig), dpi=300)
plt.close()


# Figure showing topo maps for each test zone (DTM, MERIT & FCN-SRTM)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9,5.83))
# Loop through each zone, adding three topography maps for each
for i, zone in enumerate(test_zones):
    
    # Import Random Forest arrays - ONLY test dataset pixels
    rf_predictions = np.load('{}/RF_Predictions_ByZone_{}.npy'.format(folder_results_rf, zone))
    rf_cor, rf_elv, rf_res, _,_,_,_,_,_ = process_1D_predictions(zone, rf_predictions, 'test_clip', no_data=-9999)
    
    # Import Densenet arrays - ONLY test dataset pixels
    dn_predictions = np.load('{}/densenet_Predictions_ByZone_{}.npy'.format(folder_results_densenet, zone))
    dn_cor, dn_elv, dn_res, _,_,_,_,_,_ = process_1D_predictions(zone, dn_predictions, 'test_clip', no_data=-9999)
    
    # Import Convnet arrays - ONLY test dataset pixels
    cn_predictions = np.load('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results_convnet, zone))
    cn_cor, cn_elv, cn_res, dtm, srtm, diff, merit, flha, hand = process_2D_predictions(zone, cn_predictions, 'test_clip', no_data=-9999)
    
    # Calculate RMSE for SRTM, MERIT & FCN DSMs
    RMSE_srtm = np.sqrt(np.nanmean((srtm - dtm)**2))
    RMSE_merit = np.sqrt(np.nanmean((merit - dtm)**2))
    RMSE_cn = np.sqrt(np.nanmean((cn_elv - dtm)**2))
    
    # Calculate improvement (over SRTM) for MERIT & FCN DSMs
    improve_merit = (RMSE_merit - RMSE_srtm)/RMSE_srtm * 100.
    improve_cn = (RMSE_cn - RMSE_srtm)/RMSE_srtm * 100.
    
    # For the 'TSM16_ATG' zone, rotate arrays for easier plotting
    if zone == 'TSM16_ATG':
        dtm, merit, cn_elv = [np.rot90(raster, axes=(1,0)) for raster in [dtm, merit, cn_elv]]
        
    # Get elevation range & build appropriate colourmap
    elv_min, elv_max = test_zones_props[zone]['elv_cbar_range']
    elv_cmap = cm.terrain
    elv_cmap.set_bad(color='whitesmoke')
    elv_norm = colors.Normalize(vmin=elv_min, vmax=elv_max)
    
    # Column 1: DTM
    axes[0,i].imshow(dtm, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[0,i].axis('off')
    axes[0,i].set_anchor('N')
    # Column 2: MERIT
    axes[1,i].imshow(merit, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[1,i].axis('off')
    axes[1,i].set_anchor('N')
    axes[1,i].annotate('RMSE={:.3f}m ({:.1f}%)'.format(RMSE_merit, improve_merit), xy=(0.02,0.98), xycoords='axes fraction', ha='left', va='top', size=9)
    # Column 3: FCN
    axes[2,i].imshow(cn_elv, aspect='equal', cmap=elv_cmap, norm=elv_norm)
    axes[2,i].axis('off')
    axes[2,i].set_anchor('N')
    axes[2,i].annotate('RMSE={:.3f}m ({:.1f}%)'.format(RMSE_cn, improve_cn), xy=(0.02,0.98), xycoords='axes fraction', ha='left', va='top', size=9)
    
    # Add a simple scale bar, assuming that each grid cell is approx. 23m (SRTM at this latitude)
    ncells_1km = 1000/23
    offset = 8
    axes[0,i].plot([offset, offset + ncells_1km], [offset, offset], color='black', linewidth=0.8)
    axes[0,i].plot([offset, offset], [offset-1, offset+1], color='black', linewidth=0.8)
    axes[0,i].plot([offset + ncells_1km, offset + ncells_1km], [offset-1, offset+1], color='black', linewidth=0.8)
    axes[0,i].annotate('1km', xy=(offset + 0.5*ncells_1km, 1.5*offset), ha='center', va='top', size=9)
    
# Tighten layout & make space for labels
fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig.subplots_adjust(left=0.03, top=0.96)
# Add annotations for the zone names
for i, zone in enumerate(test_zones):
    zone_label = test_zones_props[zone]['label']
    axes[0,i].annotate(zone_label, xy=([0.19,0.51,0.83][i], 0.99), xycoords='figure fraction', ha='center', va='top', weight='bold', color='dimgrey')
# Add annotations for the DSM names
for j, DSM in enumerate(['LiDAR (resampled)','MERIT (resampled)','FCN-corrected']):
    axes[j,0].annotate(DSM, xy=(0.015, [0.955,0.64,0.3][j]), xycoords='figure fraction', ha='center', va='top', rotation=90, weight='bold', color='dimgrey')
# Save figure
fig.savefig('{}/results_elv_MERIT.png'.format(folder_fig), dpi=300)
plt.close()