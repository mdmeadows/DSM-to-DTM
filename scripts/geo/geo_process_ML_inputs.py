# Process: All input data for machine learning (including division into train/dev/test sets)

# Import required packages
import sys
import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import geotiff_to_array, get_geotiff_props, get_target_patch_geom, get_feature_patch_geom, get_target_patch_array, get_feature_patch_array, pad_array, fill_array_nodata

# Define paths to data folders
folder_dtm = 'E:/mdm123/D/data/DTM/proc'                            # LiDAR DTM (reference data)
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/proc'                      # SRTM DSM & derivatives
folder_aster = 'E:/mdm123/D/data/DSM/ASTER/proc'                    # ASTER DSM & derivatives
folder_aw3d30 = 'E:/mdm123/D/data/DSM/AW3D30/proc'                  # AW3D30 DSM & derivatives
folder_ls7 = 'E:/mdm123/D/data/Landsat7/proc/cloudfree/Resampled'   # Landsat 7 - bands & derivatives
folder_ls8 = 'E:/mdm123/D/data/Landsat8/proc/cloudfree/Resampled'   # Landsat 8 - bands & derivatives
folder_gsw = 'E:/mdm123/D/data/GSW/proc'                            # Global Surface Water
folder_osm = 'E:/mdm123/D/data/OSM/proc'                            # OpenStreetMap
folder_gch = 'E:/mdm123/D/data/GCH/proc'                            # Global canopy heights
folder_gfc = 'E:/mdm123/D/data/GFC/proc'                            # Global forest cover
folder_viirs = 'E:/mdm123/D/data/NTL/VIIRS/proc'                    # Night-time light from VIIRS (2012-2019)
folder_dmsp = 'E:/mdm123/D/data/NTL/DMSP/proc'                      # Night-time light from DMSP (1993-2012)
folder_lcdb = 'E:/mdm123/D/data/LRIS/lris-lcdb-v50/proc'            # Land Cover Database v5.0 (Manaaki Whenua)

# Define paths to ML input & log folders
folder_input_1D = 'E:/mdm123/D/ML/inputs/1D'
folder_input_2D = 'E:/mdm123/D/ML/inputs/2D'
folder_logs = 'E:/mdm123/D/ML/logs'
folder_fig = 'E:/mdm123/D/figures'

# Define a list of the spectral index products calculated for each set of multi-spectral imagery
spectral_index_products = ['NDVI','EVI','AVI','SAVI','MSAVI','SI','BSI','NDMI','MNDWI','AWEInsh','AWEIsh','NDBI']

# Define lists of the variations available within each feature product group
srtm_vars = ['Z','Slope','Aspect','Roughness','TPI','TRI']
aster_vars = ['z','slope','aspect','roughness','tpi','tri']
aw3d30_vars = ['z','slope','aspect','roughness','tpi','tri']
ls7_vars = ['B1','B2','B3','B4','B5','B6_VCID_1','B6_VCID_2','B7','B8'] + spectral_index_products
ls8_vars = ['B{}'.format(i) for i in range(1,12)] + spectral_index_products
gsw_vars = ['occurrence']
osm_vars = ['Bld', 'Rds', 'Brd']
dmsp_vars = ['avg_vis', 'avg_vis_stable', 'pct_lights', 'avg_lights_pct']

# Specify the temporal query window to be used (relevant to the Landsat 7 & Landsat 8 multi-spectral imagery)
query_window = '120d'

# Define parameters relating to the image 'patches' used for the convolutional neural network
pad = 44                    # Number of padding cells/pixels along each edge (w.r.t. output/target data)
dim_out = 12                # Desired width/height of output/target data (based on architecture of the modified U-net convolutional neural network)
dim_in = dim_out + 2*pad    # Required width/height of input/feature data (for standard U-net to produce output of dimensions dim_out x dim_out)

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define the no_data value to be used/assumed
no_data_value = -9999


###############################################################################
# 1. Process feature & target rasters into vectors (RF, DCN) & patches (FCN)  #
###############################################################################

# Initialise dictionaries to track the target & feature patch generation
target_patch_dict = {'zone':[], 'idx_all':[], 'idx_valid':[], 'i':[], 'j':[], 'geometry':[], 'target_pixels':[], 'valid':[]}
feature_patch_dict = {'zone':[], 'idx_all':[], 'idx_valid':[], 'i':[], 'j':[], 'geometry':[], 'target_pixels':[], 'valid':[]}

# Initialise lists to hold arrays of all extracted patch data
target_patches_list = []
feature_patches_list = []

# Initialise another list to track variable names as they're processed & appended
feature_names = []

# Initialise dictionary of zone dataframes to hold 1D feature & target data for each zone
all_patch_vectors = pd.DataFrame()

# Initialise tracker of patch index numbers (one for all patches, one for only those patches with target data)
idx_all = 0
idx_valid = 0

# 1a. Loop through the available zones, processing all feature & target data as patches
for zone in zones:
    
    print('\nProcessing data from {}...'.format(zone))
    
    # Target: SRTM-DTM DIFF (using resampled 30m DTM based on median of 1m cells)
    diff_path = '{}/{}/SRTM_{}_Median_Diff_Pad44.tif'.format(folder_srtm, zone, zone)
    diff_array = geotiff_to_array(diff_path)
    diff_props = get_geotiff_props(diff_path)
    
    # Calculate the number of patches available in the x & y dimensions (for that particular zone)
    n_patches_x = np.ceil((diff_props['width']-2*pad)/dim_out).astype(np.int16)
    n_patches_y = np.ceil((diff_props['height']-2*pad)/dim_out).astype(np.int16)
    
    # Starting at the top & moving down:
    for j in range(n_patches_y):
        
        # Starting at the left & moving right:
        for i in range(n_patches_x):
            
            # Get the geometry of the target patch (smaller)
            target_patch_wkt = get_target_patch_geom(i, j, diff_props, pad, dim_out)
            target_patch_dict['geometry'].append(target_patch_wkt)
            target_patch_dict['zone'].append(zone)
            target_patch_dict['idx_all'].append(idx_all)
            target_patch_dict['i'].append(i)
            target_patch_dict['j'].append(j)
                    
            # Get the geometry of the feature patch (larger)
            feature_patch_wkt = get_feature_patch_geom(i, j, diff_props, pad, dim_in, dim_out)
            feature_patch_dict['geometry'].append(feature_patch_wkt)
            feature_patch_dict['zone'].append(zone)
            feature_patch_dict['idx_all'].append(idx_all)
            feature_patch_dict['i'].append(i)
            feature_patch_dict['j'].append(j)
            
            # Get the target data for that patch
            target_patch_array = get_target_patch_array(i, j, diff_array, pad, dim_out)
            
            # Check if the target patch is entirely no_data values
            patch_invalid = np.all(target_patch_array == no_data_value)
            
            # If the patch is invalid (no data available), update target & feature dictionaries accordingly and move on to next patch
            if patch_invalid:
                
                # Update target patch properties
                target_patch_dict['idx_valid'].append(np.nan)
                target_patch_dict['target_pixels'].append(0)
                target_patch_dict['valid'].append(0)
                
                # Update feature patch properties
                feature_patch_dict['idx_valid'].append(np.nan)
                feature_patch_dict['target_pixels'].append(0)
                feature_patch_dict['valid'].append(0)
                
                # Only update the overall count (not the valid count)
                idx_all += 1
                    
            # If the patch is not invalid (contains at least one valid pixel), process further
            else:
                
                # Pad target_patch_array with no_data values if necessary
                if target_patch_array.shape != (dim_out, dim_out): target_patch_array = pad_array(target_patch_array, dim_out, no_data_value)
                
                # Append array of target data available for this patch to the list
                target_patches_list.append(target_patch_array)
                
                # Update target patch properties
                target_patch_dict['idx_valid'].append(idx_valid)
                target_patch_dict['target_pixels'].append(np.sum(target_patch_array != no_data_value))
                target_patch_dict['valid'].append(1)
                
                # Update feature patch properties
                feature_patch_dict['idx_valid'].append(idx_valid)
                feature_patch_dict['target_pixels'].append(np.sum(target_patch_array != no_data_value))
                feature_patch_dict['valid'].append(1)
                
                # 1D data: Initialise a dataframe to hold feature & target data for this particular patch
                patch_vectors = pd.DataFrame()
                patch_vectors['idx_all'] = list(np.repeat(idx_all, dim_out*dim_out))
                patch_vectors['idx_valid'] = list(np.repeat(idx_valid, dim_out*dim_out))
                patch_vectors['zone'] = list(np.repeat(zone, dim_out*dim_out))
                patch_vectors['i'] = list(np.repeat(i, dim_out*dim_out))
                patch_vectors['j'] = list(np.repeat(j, dim_out*dim_out))
                patch_vectors['target_pixels'] = list(np.repeat(np.sum(target_patch_array != no_data_value), dim_out*dim_out))
                patch_vectors['diff'] = target_patch_array.flatten()
                
                # Increment both index counts, now that all counts/trackers have been recorded for this patch
                idx_all += 1
                idx_valid += 1
                                
                # 2D data: Initialise a list to hold all feature arrays available for this particular patch
                feature_patch_arrays = []
                
                # Feature Set 1: SRTM data & derivatives
                for var in srtm_vars:
                    srtm_name = 'srtm_{}'.format(var.lower())
                    srtm_path = '{}/{}/SRTM_{}_{}_Pad44.tif'.format(folder_srtm, zone, zone, var)
                    srtm_array = geotiff_to_array(srtm_path)
                    if np.any(srtm_array==no_data_value): srtm_array = fill_array_nodata(srtm_array, no_data_value)
                    if srtm_array.shape != diff_array.shape: print('ERROR! SRTM {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, srtm_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, srtm_array, pad, dim_out)
                    patch_vectors[srtm_name] = feature_patch_array_unpadded.flatten()
                    if srtm_name not in feature_names: feature_names.append(srtm_name)
                
                # Feature Set 2: ASTER data & derivatives
                for var in aster_vars:
                    aster_name = 'aster_{}'.format(var.lower())
                    aster_path = '{}/{}/ASTER_{}_{}_Pad44.tif'.format(folder_aster, zone, zone, var.capitalize())
                    aster_array = geotiff_to_array(aster_path)
                    if np.any(aster_array==no_data_value): aster_array = fill_array_nodata(aster_array, no_data_value)
                    if aster_array.shape != diff_array.shape: print('ERROR! ASTER {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, aster_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, aster_array, pad, dim_out)
                    patch_vectors[aster_name] = feature_patch_array_unpadded.flatten()
                    if aster_name not in feature_names: feature_names.append(aster_name)
                
                # Feature Set 3: AW3D30 data & derivatives
                for var in aw3d30_vars:
                    aw3d30_name = 'aw3d30_{}'.format(var.lower())
                    aw3d30_path = '{}/{}/AW3D30_{}_{}_Pad44.tif'.format(folder_aw3d30, zone, zone, var.capitalize())
                    aw3d30_array = geotiff_to_array(aw3d30_path)
                    if np.any(aw3d30_array==no_data_value): aw3d30 = fill_array_nodata(aw3d30_array, no_data_value)
                    if aw3d30_array.shape != diff_array.shape: print('ERROR! AW3D30 {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, aw3d30_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, aw3d30_array, pad, dim_out)
                    patch_vectors[aw3d30_name] = feature_patch_array_unpadded.flatten()
                    if aw3d30_name not in feature_names: feature_names.append(aw3d30_name)
                
                # Feature Set 4: Landsat 7 - nine (9) bands (incl both Band 6 options) & twelve (12) derivatives available - all based on median composite rasters (gap-filled using lower-quality composite, resampled using cubicspline)
                for var in ls7_vars:
                    ls7_name = 'ls7_{}'.format(var.lower())
                    ls7_path = '{}/{}/LS7_{}_{}_{}_Pad44_Bounded.tif'.format(folder_ls7, zone, var, query_window, zone)
                    ls7_array = geotiff_to_array(ls7_path)
                    if np.any(ls7_array==no_data_value): ls7_array = fill_array_nodata(ls7_array, no_data_value)
                    if ls7_array.shape != diff_array.shape: print('ERROR! LS7 {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, ls7_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, ls7_array, pad, dim_out)
                    patch_vectors[ls7_name] = feature_patch_array_unpadded.flatten()
                    if ls7_name not in feature_names: feature_names.append(ls7_name)
                
                # Feature Set 5: Landsat 8 - eleven (11) bands & twelve (12) derivatives available - all based on median composite rasters (gap-filled using lower-quality composite, resampled using cubicspline)
                for var in ls8_vars:
                    ls8_name = 'ls8_{}'.format(var.lower())
                    ls8_path = '{}/{}/LS8_{}_{}_{}_Pad44_Bounded.tif'.format(folder_ls8, zone, var, query_window, zone)
                    ls8_array = geotiff_to_array(ls8_path)
                    if np.any(ls8_array==no_data_value): ls8_array = fill_array_nodata(ls8_array, no_data_value)
                    if ls8_array.shape != diff_array.shape: print('ERROR! LS8 {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, ls8_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, ls8_array, pad, dim_out)
                    patch_vectors[ls8_name] = feature_patch_array_unpadded.flatten()
                    if ls8_name not in feature_names: feature_names.append(ls8_name)
                
                # Feature Set 6: Global Surface Water - only the 'occurrence' raster seems to be relevant (using cubicspline for resampling, as above)
                for var in gsw_vars:
                    gsw_name = 'gsw_{}'.format(var.lower())
                    gsw_path = '{}/{}/GSW_{}_{}_cubicspline_Bounded_Pad44.tif'.format(folder_gsw, zone, var, zone)
                    gsw_array = geotiff_to_array(gsw_path)
                    if np.any(gsw_array==no_data_value): gsw_array = fill_array_nodata(gsw_array, no_data_value)
                    if gsw_array.shape != diff_array.shape: print('ERROR! GSW {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, gsw_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, gsw_array, pad, dim_out)
                    patch_vectors[gsw_name] = feature_patch_array_unpadded.flatten()
                    if gsw_name not in feature_names: feature_names.append(gsw_name)
                
                # Feature Set 7: OpenStreetMap (building footprints, road networks, and bridges)
                for var in osm_vars:
                    osm_name = 'osm_{}'.format(var.lower())
                    osm_path = '{}/{}/OSM_{}_{}_Pad44.tif'.format(folder_osm, zone, var, zone)
                    osm_array = geotiff_to_array(osm_path)
                    if np.any(osm_array==no_data_value): osm_array = fill_array_nodata(osm_array, no_data_value)
                    if osm_array.shape != diff_array.shape: print('ERROR! OSM {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, osm_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, osm_array, pad, dim_out)
                    patch_vectors[osm_name] = feature_patch_array_unpadded.flatten()
                    if osm_name not in feature_names: feature_names.append(osm_name)
                
                # Feature Set 8: Global Canopy Heights (using the cubicspline resampling method)
                gch_name = 'gch'
                gch_path = '{}/{}/GCH_{}_cubicspline_Bounded_Pad44.tif'.format(folder_gch, zone, zone)
                gch_array = geotiff_to_array(gch_path)
                if np.any(gch_array==no_data_value): gch_array = fill_array_nodata(gch_array, no_data_value)
                if gch_array.shape != diff_array.shape: print('ERROR! GCH feature raster does not match target raster')
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, gch_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                feature_patch_arrays.append(feature_patch_array)
                # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                feature_patch_array_unpadded = get_target_patch_array(i, j, gch_array, pad, dim_out)
                patch_vectors[gch_name] = feature_patch_array_unpadded.flatten()
                if gch_name not in feature_names: feature_names.append(gch_name)
                
                # Feature Set 9: Global Forest Cover (using the cubicspline resampling method)
                gfc_name = 'gfc'
                gfc_path = '{}/{}/GFC_treecover2000_{}_cubicspline_Bounded_Pad44.tif'.format(folder_gfc, zone, zone)
                gfc_array = geotiff_to_array(gfc_path)
                if np.any(gfc_array==no_data_value): gfc_array = fill_array_nodata(gfc_array, no_data_value)
                if gfc_array.shape != diff_array.shape: print('ERROR! GFC feature raster does not match target raster')
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, gfc_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                feature_patch_arrays.append(feature_patch_array)
                # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                feature_patch_array_unpadded = get_target_patch_array(i, j, gfc_array, pad, dim_out)
                patch_vectors[gfc_name] = feature_patch_array_unpadded.flatten()
                if gfc_name not in feature_names: feature_names.append(gfc_name)
                
                # Feature Set 10: VIIRS Night-Time Light (using the cubicspline resampling method)
                viirs_name = 'ntl_viirs'
                viirs_path = '{}/{}/NTL_VIIRS_{}_cubicspline_Pad44.tif'.format(folder_viirs, zone, zone)
                viirs_array = geotiff_to_array(viirs_path)
                if np.any(viirs_array==no_data_value): viirs_array = fill_array_nodata(viirs_array, no_data_value)
                if viirs_array.shape != diff_array.shape: print('ERROR! VIIRS NTL feature raster does not match target raster')
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, viirs_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                feature_patch_arrays.append(feature_patch_array)
                # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                feature_patch_array_unpadded = get_target_patch_array(i, j, viirs_array, pad, dim_out)
                patch_vectors[viirs_name] = feature_patch_array_unpadded.flatten()
                if viirs_name not in feature_names: feature_names.append(viirs_name)
                
                # Feature Set 11: DMSP Night-Time Light (using the cubicspline resampling method)
                for var in dmsp_vars:
                    dmsp_name = 'dmsp_{}'.format(var.lower())
                    dmsp_path = '{}/{}/NTL_DMSP_{}_{}_cubicspline_{}Pad44.tif'.format(folder_dmsp, zone, var, zone, 'Bounded_' if var=='pct_lights' else '')
                    dmsp_array = geotiff_to_array(dmsp_path)
                    if np.any(dmsp_array==no_data_value): dmsp_array = fill_array_nodata(dmsp_array, no_data_value)
                    if dmsp_array.shape != diff_array.shape: print('ERROR! DMSP NTL {} feature raster does not match target raster'.format(var.upper()))
                    # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                    feature_patch_array = get_feature_patch_array(i, j, dmsp_array, dim_in, dim_out)
                    if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                    feature_patch_arrays.append(feature_patch_array)
                    # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                    feature_patch_array_unpadded = get_target_patch_array(i, j, dmsp_array, pad, dim_out)
                    patch_vectors[dmsp_name] = feature_patch_array_unpadded.flatten()
                    if dmsp_name not in feature_names: feature_names.append(dmsp_name)
                
                # Land Cover classes (LCDB v5.0): 1D dataset only
                lc_name = 'lcdb'
                lc_path = '{}/LCDB_GroupID_{}_Pad44.tif'.format(folder_lcdb, zone)
                lc_array = geotiff_to_array(lc_path)
                if lc_array.shape != diff_array.shape: print('ERROR! LCDB raster does not match target raster')
                # 1D data: Extract unpadded data array, flatten it, and add it to the patch-wise dictionary of vector data
                feature_patch_array_unpadded = get_target_patch_array(i, j, lc_array, pad, dim_out)
                patch_vectors[lc_name] = feature_patch_array_unpadded.flatten()
                
                
                # After all feature datasets have been processed, consolidate the 1D & 2D data extracted for that patch
                
                # 1D data: Append patch dataframe to the overall dataframe
                patch_vectors_dropna = patch_vectors.replace(-9999.0, np.nan).dropna(axis=0, how='any', inplace=False)
                all_patch_vectors = all_patch_vectors.append(patch_vectors_dropna)
                
                # 2D data: Append 'feature_patch_arrays' list to the overall 'feature_patch_array_stack' list
                feature_patches_list.append(np.array(feature_patch_arrays))


# 1b. Delete redundant variables & save useful processing information

# Delete leftover arrays & partial dataframes, to free up space
del diff_array, srtm_array, aster_array, aw3d30_array, ls7_array, ls8_array, gsw_array, osm_array, gch_array, gfc_array, viirs_array, dmsp_array, feature_patch_array, feature_patch_arrays, feature_patch_array_unpadded, target_patch_array
del patch_vectors, patch_vectors_dropna

# Add a new column to the 1D results dataframe, defining the number of rows (pixels) available for each unique 'idx_valid' ID (for each patch's dim_out x dim_out target extent)
all_patch_vectors = pd.merge(all_patch_vectors, all_patch_vectors.groupby('idx_valid').size().reset_index(name='n_pixels'), on='idx_valid', how='left').fillna({'n_pixels':0})

# Write list of feature names to a pickle file in each result folder
pickle.dump(feature_names, open('{}/feature_names_ordered_all.p'.format(folder_input_1D), 'wb'))
pickle.dump(feature_names, open('{}/feature_names_ordered_all.p'.format(folder_input_2D), 'wb'))

# Save dictionaries describing target & feature patches to pickle objects, in case they're needed later
pickle.dump(target_patch_dict, open('{}/patches/target_patch_dict.p'.format(folder_logs), 'wb'))
pickle.dump(feature_patch_dict, open('{}/patches/feature_patch_dict.p'.format(folder_logs), 'wb'))

# Convert the target patch dictionary to a SHP for manual inspection, and save original dictionaries for later reference
target_patch_df = pd.DataFrame(target_patch_dict).set_index('idx_all', drop=False)
target_patch_df = pd.merge(target_patch_df, all_patch_vectors.groupby('idx_valid').size().reset_index(name='n_pixels'), on='idx_valid', how='left').fillna({'n_pixels':0})
target_patch_df.to_csv('{}/patches/target_patch_df.csv'.format(folder_logs), index=False)
target_patch_df['geometry'] = target_patch_df['geometry'].apply(wkt.loads)
target_patch_gdf = gpd.GeoDataFrame(target_patch_df, geometry='geometry', crs=diff_props['proj'])
target_patch_gdf.to_file('{}/patches/SHP/patches_target.shp'.format(folder_logs))

# Convert the feature patch dictionary to a SHP for manual inspection
feature_patch_df = pd.DataFrame(feature_patch_dict).set_index('idx_all', drop=False)
feature_patch_df = pd.merge(feature_patch_df, all_patch_vectors.groupby('idx_valid').size().reset_index(name='n_pixels'), on='idx_valid', how='left').fillna({'n_pixels':0})
feature_patch_df.to_csv('{}/patches/feature_patch_df.csv'.format(folder_logs), index=False)
feature_patch_df['geometry'] = feature_patch_df['geometry'].apply(wkt.loads)
feature_patch_gdf = gpd.GeoDataFrame(feature_patch_df, geometry='geometry', crs=diff_props['proj'])
feature_patch_gdf.to_file('{}/patches/SHP/patches_features.shp'.format(folder_logs))


# 1c. Save 1D results (dataframe to CSV)

# Write vector dataframe to a CSV file (as a backup)
all_patch_vectors.to_csv('{}/Input1D_All.csv'.format(folder_input_1D), index=False)

# Delete 1D dataframe, to free up enough memory to be able to save 2D results
del all_patch_vectors


# 1d. Save 2D results (ndarrays to numpy files)

# Note: in keras-tensorflow, default ordering is 'channels_last': (batch, height, width, channels)

# Target patches: Stack all target patches in one array, adjust dimensions to match keras input convention, & save to numpy file
target_patches_array = np.array(target_patches_list, dtype=np.float32)
del target_patches_list
target_patches_array.shape       # (16619, 12, 12) = (patches, patch_height, patch_width)
target_patches_array = np.expand_dims(target_patches_array, axis=-1)   # Add extra dimension right at the end
target_patches_array.shape       # (16619, 12, 12, 1) = (patches, patch_height, patch_width, channel)
np.save('{}/Input2D_Target_All.npy'.format(folder_input_2D), target_patches_array)
del target_patches_array

# Feature patches: Stack all feature patches in one array, check its dimensions & save to numpy file
feature_patches_array = np.array(feature_patches_list, dtype=np.float32)
del feature_patches_list
feature_patches_array.shape      # (16619, 73, 100, 100) = (patches, features, patch_height, patch_width)
feature_patches_array = np.moveaxis(feature_patches_array, 1, -1)
feature_patches_array.shape      # (16619, 100, 100, 73) = (patches, patch_height, patch_width, features)
np.save('{}/Input2D_Features_All.npy'.format(folder_input_2D), feature_patches_array)
del feature_patches_array


###############################################################################
# 2. Define testing zones based on manual inspection of patch distribution    #
###############################################################################

# For a fair comparison of the different ML models (whether pixel- or image-based), the dev & test sets should contain only pixels within full/intact image patches

# Read processed 1D input data back into memory as a dataframe
df = pd.read_csv('{}/Input1D_All.csv'.format(folder_input_1D))

# Assuming we want to assign 5% of all available data to testing using only FULL patches, express the total amount of data available (grid cells/pixels) as FULL patch equivalents
n_patches_full_equivalent = len(df.index)/(dim_out * dim_out)    # 13,944.2
0.05 * n_patches_full_equivalent                                 # 697.2 - total number of full patches to assign for testing
(0.05 * n_patches_full_equivalent)/3.                            # 232.4 - number of full patches to assign to each testing zone (assuming there are three of them)

# Manually inspect (in GIS software) the target patch SHP exported above, to select parts of zones to assign as testing sites, prioritising areas with high flood exposure & only using FULL patches
con_a = (df['zone']=='MRL18_WPE') & (df['i']>=3) & (df['i']<=23) & (df['j']>=19) & (df['j']<=29) & (df['n_pixels']==dim_out**2)
con_b = (df['zone']=='MRL18_WVL') & (df['i']>=4) & (df['i']<=28) & (df['j']>=24) & (df['j']<=37) & (df['n_pixels']==dim_out**2)
con_c = (df['zone']=='TSM16_ATG') & (df['i']>=29) & (df['i']<=41) & (df['j']>=22) & (df['j']<=42) & (df['n_pixels']==dim_out**2)

# Check the number of valid PATCHES assigned to each testing zone
len(df[con_a]['idx_valid'].unique().tolist())  # 231 patches
len(df[con_b]['idx_valid'].unique().tolist())  # 233 patches
len(df[con_c]['idx_valid'].unique().tolist())  # 233 patches

# Check fraction (of all pixels, expressed as full-patch equivalents) that this test set selection corresponds to
sum([len(df[con]['idx_valid'].unique().tolist()) for con in [con_a, con_b, con_c]])/n_patches_full_equivalent       # 0.04998493499637693

# Get lists of various subsets of patch IDs
patches_all = df['idx_valid'].unique().tolist()                                    # Index of ALL patches (i.e. patches containing at least 1 valid cell/pixel)
patches_full = df[df['n_pixels']==dim_out**2]['idx_valid'].unique().tolist()       # Index of FULL patches (i.e. comprising only valid cells/pixels)
patches_test = df[con_a|con_b|con_c]['idx_valid'].unique().tolist()                # Index of patches satisfying either of three test zone conditions defined above (only FULL patches considered)

# Check fraction of data assigned to testing set: number of FULL patches divided by all data available (expressed in FULL patch equivalents)
len(patches_test)/n_patches_full_equivalent            # 0.04998493499637693

# Filter the list of 'full' patches to remove those already assigned to testing, leaving only those to be used for the train/dev sets
patches_full_traindev = [patch for patch in patches_full if patch not in patches_test]

# Randomly split the 'full' train/dev patch IDs into train & dev sets, with 5% of all data (expressed as FULL patch equivalents) assigned to dev (and choosing only from the 'full' patches)
n_dev = int(np.round(0.05 * n_patches_full_equivalent))     # 697
np.random.seed(seed=1)
patches_dev = np.random.choice(patches_full_traindev, size=n_dev, replace=False).tolist()
len(patches_dev)        # 697
patches_train = [patch for patch in patches_all if patch not in patches_dev and patch not in patches_test]
len(patches_train)      # 14058

# Save these list of indices of valid patches, for later reference
pickle.dump(patches_all, open('{}/patches/patches_index_all.p'.format(folder_logs), 'wb'))
pickle.dump(patches_full, open('{}/patches/patches_index_full.p'.format(folder_logs), 'wb'))
pickle.dump(patches_train, open('{}/patches/patches_index_train.p'.format(folder_logs), 'wb'))
pickle.dump(patches_dev, open('{}/patches/patches_index_dev.p'.format(folder_logs), 'wb'))
pickle.dump(patches_test, open('{}/patches/patches_index_test.p'.format(folder_logs), 'wb'))


###############################################################################
# 3. Split available data into train, dev & test sets (1D & 2D model inputs)  #
###############################################################################

# Define function to update dataframe rows with train/dev/test allocation, based on lists of patch IDs developed above
def allocate_available_data(row, patches_train, patches_dev, patches_test):
    if row['idx_valid'] in patches_train:
        return 'train'
    elif row['idx_valid'] in patches_dev:
        return 'dev'
    elif row['idx_valid'] in patches_test:
        return 'test'
    else:
        return 'None'

# Reload the lists of patch indices ('index_valid') allocated to each modelling set
with open('{}/patches/patches_index_all.p'.format(folder_logs), 'rb') as f: patches_all = pickle.load(f)
with open('{}/patches/patches_index_train.p'.format(folder_logs), 'rb') as f: patches_train = pickle.load(f)
with open('{}/patches/patches_index_dev.p'.format(folder_logs), 'rb') as f: patches_dev = pickle.load(f)
with open('{}/patches/patches_index_test.p'.format(folder_logs), 'rb') as f: patches_test = pickle.load(f)

# If it's not in memory already, read processed 1D input data as a dataframe
df = pd.read_csv('{}/Input1D_All.csv'.format(folder_input_1D))


# 4a. Split 1D data into train, dev & test sets

# Update vector dataframe with train/dev/test information
df['usage'] = df.apply(lambda row: allocate_available_data(row, patches_train, patches_dev, patches_test), axis=1)
pixels_train_n = len(df[df['usage']=='train'].index)   # 1,807,229
pixels_dev_n = len(df[df['usage']=='dev'].index)       #   100,368
pixels_test_n = len(df[df['usage']=='test'].index)     #   100,368
print('\nPixel-wise data allocation is training ({:.2%}), dev ({:.2%}) & test ({:.2%})'.format(pixels_train_n/len(df.index), pixels_dev_n/len(df.index), pixels_test_n/len(df.index)))
# Pixel-wise data allocation is training (90.00%), dev (5.00%) & test (5.00%)

# Write separate vector dataframes to CSV files for train, dev & test data
df[df['usage']=='train'].to_csv('{}/Input1D_Train.csv'.format(folder_input_1D), index=False)
df[df['usage']=='dev'].to_csv('{}/Input1D_Dev.csv'.format(folder_input_1D), index=False)
df[df['usage']=='test'].to_csv('{}/Input1D_Test.csv'.format(folder_input_1D), index=False)


# 4b. Generate SHPs showing the spatial distribution of patches into train, dev & test sets

# Update target patch dataframe & export as a SHP
target_patch_df = pd.read_csv('{}/patches/target_patch_df.csv'.format(folder_logs))
target_patch_df['usage'] = target_patch_df.apply(lambda row: allocate_available_data(row, patches_train, patches_dev, patches_test), axis=1)
target_patch_df['geometry'] = target_patch_df['geometry'].apply(wkt.loads)
target_patch_gdf = gpd.GeoDataFrame(target_patch_df, geometry='geometry', crs=diff_props['proj'])
target_patch_gdf.to_file('{}/patches/SHP/patches_target_split.shp'.format(folder_logs))

# Update feature patch dataframe & export as a SHP
feature_patch_df = pd.read_csv('{}/patches/feature_patch_df.csv'.format(folder_logs))
feature_patch_df['usage'] = feature_patch_df.apply(lambda row: allocate_available_data(row, patches_train, patches_dev, patches_test), axis=1)
feature_patch_df['geometry'] = feature_patch_df['geometry'].apply(wkt.loads)
feature_patch_gdf = gpd.GeoDataFrame(feature_patch_df, geometry='geometry', crs=diff_props['proj'])
feature_patch_gdf.to_file('{}/patches/SHP/patches_features_split.shp'.format(folder_logs))


# 4c. For 2D data, split the TARGET patch arrays into train, dev & test sets

# Load the numpy array of all target patch data
target_patches_array = np.load('{}/Input2D_Target_All.npy'.format(folder_input_2D))

# Split ndarray of target patch data up into separate train, dev & test ndarrays
target_patches_array_train = target_patches_array[patches_train,:,:,:]
target_patches_array_dev = target_patches_array[patches_dev,:,:,:]
target_patches_array_test = target_patches_array[patches_test,:,:,:]

# Check the shape of each new array, to ensure no information has been lost
target_patches_array_train.shape    # (14058, 12, 12, 1)
target_patches_array_dev.shape      # (697,   12, 12, 1)
target_patches_array_test.shape     # (697,   12, 12, 1)

# Check that the total number of patches used for modelling (train, dev & test) matches the original list of valid patches
sum([array.shape[0] for array in [target_patches_array_train, target_patches_array_dev, target_patches_array_test]])  # 15452
len(patches_all)                                                                                                      # 15452

# Save target arrays to .npy files
np.save('{}/Input2D_Target_Train.npy'.format(folder_input_2D), target_patches_array_train)
np.save('{}/Input2D_Target_Dev.npy'.format(folder_input_2D), target_patches_array_dev)
np.save('{}/Input2D_Target_Test.npy'.format(folder_input_2D), target_patches_array_test)


# 4d. For 2D data, split the FEATURE patch arrays into train, dev & test sets

# Load the numpy array of all feature patch data
feature_patches_array = np.load('{}/Input2D_Features_All.npy'.format(folder_input_2D))

# Split ndarray of feature patch data up into separate train, dev & test ndarrays
feature_patches_array_train = feature_patches_array[patches_train,:,:,:]
feature_patches_array_dev = feature_patches_array[patches_dev,:,:,:]
feature_patches_array_test = feature_patches_array[patches_test,:,:,:]

# Check the shape of each new array, to ensure no information has been lost
feature_patches_array_train.shape    # (14058, 100, 100, 73)
feature_patches_array_dev.shape      # (697,   100, 100, 73)
feature_patches_array_test.shape     # (697,   100, 100, 73)

# Save feature arrays to .npy files
np.save('{}/Input2D_Features_Train.npy'.format(folder_input_2D), feature_patches_array_train)
np.save('{}/Input2D_Features_Dev.npy'.format(folder_input_2D), feature_patches_array_dev)
np.save('{}/Input2D_Features_Test.npy'.format(folder_input_2D), feature_patches_array_test)


###############################################################################
# 5. Compare data distributions between the train, validation & test datasets #
###############################################################################

# Read processed 1D input data back into memory as dataframes
df_train = pd.read_csv('{}/Input1D_Train.csv'.format(folder_input_1D))
df_dev = pd.read_csv('{}/Input1D_Dev.csv'.format(folder_input_1D))
df_test = pd.read_csv('{}/Input1D_Test.csv'.format(folder_input_1D))

# Get the unique list of zones included in the test dataset
zones_test = df_test['zone'].unique().tolist()


# 5a. Compare land cover distributions each each dataset

# Set up a dictionary to contain SRTM-LiDAR difference values corresponding to each Manaaki Whenua landclass type present in that LiDAR zone coverage
diff_by_landcover = {1:{'label':'Artificial\nsurfaces', 'data':[], 'colour':(78/255, 78/255, 78/255)},
                     2:{'label':'Bare/lightly-\nvegetated\nsurfaces', 'data':[], 'colour':(255/255, 235/255, 190/255)},
                     3:{'label':'Water\nbodies', 'data':[], 'colour':(0/255, 197/255, 255/255)},
                     4:{'label':'Cropland', 'data':[], 'colour':(255/255, 170/255, 0/255)},
                     5:{'label':'Grassland,\nSedgeland\n& Marshland', 'data':[], 'colour':(255/255, 255/255, 115/255)},
                     6:{'label':'Scrub &\nShrubland', 'data':[], 'colour':(137/255, 205/255, 102/255)},
                     7:{'label':'Forest', 'data':[], 'colour':(38/255, 115/255, 0/255)},
                     8:{'label':'Other', 'data':[], 'colour':'#FF0000'}}

# Establish list of colours & labels to be used for land cover figures
lc_colours = [diff_by_landcover[i]['colour'] for i in range(1,8)]
lc_labels = [diff_by_landcover[i]['label'] for i in range(1,8)]

# Visualise: Distribution of land cover types within each dataset
n_axes = 2 + len(zones_test)
fig, axes = plt.subplots(nrows=n_axes, sharex=True, figsize=(2*n_axes, 8))
# Add training data to top row
lc_train_counts = [len(df_train[df_train['lcdb']==i].index) for i in range(1,8)]
lc_train_freq = [count/sum(lc_train_counts) for count in lc_train_counts]
axes[0].bar(x=range(1,8), height=lc_train_freq, color=lc_colours, alpha=0.7)
[axes[0].annotate('{:.1f}%'.format(freq*100), xy=(i, freq), fontsize=8, ha='center', va='bottom') for i, freq in zip(range(1,8), lc_train_freq)]
axes[0].annotate('Training set', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9, ha='left', va='top', fontweight='bold', color='dimgrey')
# Add dev data to second row
lc_dev_counts = [len(df_dev[df_dev['lcdb']==i].index) for i in range(1,8)]
lc_dev_freq = [count/sum(lc_dev_counts) for count in lc_dev_counts]
axes[1].bar(x=range(1,8), height=lc_dev_freq, color=lc_colours, alpha=0.7)
[axes[1].annotate('{:.1f}%'.format(freq*100), xy=(i, freq), fontsize=8, ha='center', va='bottom') for i, freq in zip(range(1,8), lc_dev_freq)]
axes[1].annotate('Validation set', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9, ha='left', va='top', fontweight='bold', color='dimgrey')
# Add test data to remaining axes, one zone at a time
for j, zone in enumerate(zones_test):
    lc_test_counts = [len(df_test[(df_test['lcdb']==i)&(df_test['zone']==zone)].index) for i in range(1,8)]
    lc_test_freq = [count/sum(lc_test_counts) for count in lc_test_counts]
    axes[j+2].bar(x=range(1,8), height=lc_test_freq, color=lc_colours, alpha=0.7)
    [axes[j+2].annotate('{:.1f}%'.format(freq*100), xy=(i, freq), fontsize=8, ha='center', va='bottom') for i, freq in zip(range(1,8), lc_test_freq)]
    axes[j+2].annotate('Testing set ({})'.format(zone), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9, ha='left', va='top', fontweight='bold', color='dimgrey')
# Apply axes-wide updates
for k in range(n_axes):
    axes[k].set_xticks(range(1,8))
    axes[k].set_ylabel('Frequency')
    [axes[k].spines[edge].set_visible(False) for edge in ['top','right']]
# Add figure labels, etc
axes[n_axes-1].set_xticklabels(lc_labels, fontsize=7)
fig.tight_layout()
fig.suptitle('Distribution of Land Cover by Dataset', fontsize=10, fontweight='bold')
plt.subplots_adjust(top=0.95)
fig.savefig('{}/All/Distributions/LCDB/landcover_by_split.png'.format(folder_fig), dpi=300)
plt.close()

# Visualise: Distribution of SRTM elevations within each dataset
n_axes = 2 + len(zones_test)
fig, axes = plt.subplots(nrows=n_axes, sharex=True, figsize=(2*n_axes, 8))
# Top: Training data
axes[0].hist(df_train['srtm_z'].values, color='blue', linewidth=1.5, alpha=0.5, bins=50, density=True, histtype='step')
axes[0].annotate('Training set', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=9, ha='center', va='top', fontweight='bold', color='dimgrey')
axes[0].set_ylabel('Frequency')
[axes[0].spines[edge].set_visible(False) for edge in ['top','right']]
# Middle: Validation data
axes[1].hist(df_dev['srtm_z'].values, color='green', linewidth=1.5, alpha=0.5, bins=50, density=True, histtype='step')
axes[1].annotate('Validation set', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=9, ha='center', va='top', fontweight='bold', color='dimgrey')
axes[1].set_ylabel('Frequency')
[axes[1].spines[edge].set_visible(False) for edge in ['top','right']]
# Rest: Testing data (by each zone)
for i, zone in enumerate(zones_test):
    axes[2+i].hist(df_test[df_test['zone']==zone]['srtm_z'].values, color='firebrick', label=zone, linewidth=1.5, alpha=0.5, bins=50, density=True, histtype='step')
    axes[2+i].annotate('Testing set ({})'.format(zone), xy=(0.5, 0.9), xycoords='axes fraction', fontsize=9, ha='center', va='top', fontweight='bold', color='dimgrey')
    axes[2+i].set_ylabel('Frequency')
    [axes[2+i].spines[edge].set_visible(False) for edge in ['top','right']]
# Add figure labels, etc
axes[n_axes-1].set_xlabel('SRTM Elevations [m]')
fig.tight_layout()
fig.suptitle('Distribution of SRTM Elevations by Dataset', fontsize=10, fontweight='bold')
plt.subplots_adjust(top=0.95)
fig.savefig('{}/All/Distributions/SRTM/srtm_z_by_split.png'.format(folder_fig), dpi=300)
plt.close()

# Visualise: Distribution of SRTM slopes within each dataset
n_axes = 2 + len(zones_test)
fig, axes = plt.subplots(nrows=n_axes, sharex=True, figsize=(2*n_axes, 8))
# Top: Training data
axes[0].hist(df_train['srtm_slope'].values, color='blue', linewidth=1.5, alpha=0.5, bins=50, density=True, histtype='step')
axes[0].annotate('Training set', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=9, ha='center', va='top', fontweight='bold', color='dimgrey')
axes[0].set_ylabel('Frequency')
[axes[0].spines[edge].set_visible(False) for edge in ['top','right']]
# Middle: Validation data
axes[1].hist(df_dev['srtm_slope'].values, color='green', linewidth=1.5, alpha=0.5, bins=50, density=True, histtype='step')
axes[1].annotate('Validation set', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=9, ha='center', va='top', fontweight='bold', color='dimgrey')
axes[1].set_ylabel('Frequency')
[axes[1].spines[edge].set_visible(False) for edge in ['top','right']]
# Rest: Testing data (by each zone)
for i, zone in enumerate(zones_test):
    axes[2+i].hist(df_test[df_test['zone']==zone]['srtm_slope'].values, color='firebrick', label=zone, linewidth=1.5, alpha=0.5, bins=50, density=True, histtype='step')
    axes[2+i].annotate('Testing set ({})'.format(zone), xy=(0.5, 0.9), xycoords='axes fraction', fontsize=9, ha='center', va='top', fontweight='bold', color='dimgrey')
    axes[2+i].set_ylabel('Frequency')
    [axes[2+i].spines[edge].set_visible(False) for edge in ['top','right']]
# Add figure labels, etc
axes[n_axes-1].set_xlabel('SRTM Slopes [%]')
fig.tight_layout()
fig.suptitle('Distribution of SRTM Slopes by Dataset', fontsize=10, fontweight='bold')
plt.subplots_adjust(top=0.95)
fig.savefig('{}/All/Distributions/SRTM/srtm_slope_by_split.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 6. Process intact feature & target test datasets for each zone (1D vectors) #
###############################################################################

# Loop through each zone, preparing 1D inputs
for zone in zones:
    
    print('\nProcessing 1D vector data for {} zone:'.format(zone))
    
    # Initialise dataframe to hold feature & target data for input to Random Forest & Densenet models
    df = pd.DataFrame()
    
    # Target: SRTM-DTM DIFF (using resampled 30m DTM based on median of 1m cells)
    diff_path = '{}/{}/SRTM_{}_Median_Diff.tif'.format(folder_srtm, zone, zone)
    diff_array = geotiff_to_array(diff_path)
    df['diff'] = diff_array.flatten()
    
    # Feature Set 1: SRTM data & derivatives
    for var in srtm_vars:
        srtm_name = 'srtm_{}'.format(var.lower())
        srtm_path = '{}/{}/SRTM_{}_{}.tif'.format(folder_srtm, zone, zone, var)
        srtm_array = geotiff_to_array(srtm_path)
        if np.any(srtm_array==no_data_value): srtm_array = fill_array_nodata(srtm_array, no_data_value)
        if srtm_array.shape != diff_array.shape: print('ERROR! SRTM {} feature raster does not match target raster'.format(var.upper()))
        df[srtm_name] = srtm_array.flatten()
    
    # Feature Set 2: ASTER data & derivatives
    for var in aster_vars:
        aster_name = 'aster_{}'.format(var.lower())
        aster_path = '{}/{}/ASTER_{}_{}.tif'.format(folder_aster, zone, zone, var.capitalize())
        aster_array = geotiff_to_array(aster_path)
        if np.any(aster_array==no_data_value): aster_array = fill_array_nodata(aster_array, no_data_value)
        if aster_array.shape != diff_array.shape: print('ERROR! ASTER {} feature raster does not match target raster'.format(var.upper()))
        df[aster_name] = aster_array.flatten()
    
    # Feature Set 3: AW3D30 data & derivatives
    for var in aw3d30_vars:
        aw3d30_name = 'aw3d30_{}'.format(var.lower())
        aw3d30_path = '{}/{}/AW3D30_{}_{}.tif'.format(folder_aw3d30, zone, zone, var.capitalize())
        aw3d30_array = geotiff_to_array(aw3d30_path)
        if np.any(aw3d30_array==no_data_value): aw3d30 = fill_array_nodata(aw3d30_array, no_data_value)
        if aw3d30_array.shape != diff_array.shape: print('ERROR! AW3D30 {} feature raster does not match target raster'.format(var.upper()))
        df[aw3d30_name] = aw3d30_array.flatten()
    
    # Feature Set 4: Landsat 7 - nine (9) bands (incl both Band 6 options) & twelve (12) derivatives available - all based on median composite rasters (gap-filled using lower-quality composite, resampled using cubicspline)
    for var in ls7_vars:
        ls7_name = 'ls7_{}'.format(var.lower())
        ls7_path = '{}/{}/LS7_{}_{}_{}_Bounded.tif'.format(folder_ls7, zone, var, query_window, zone)
        ls7_array = geotiff_to_array(ls7_path)
        if np.any(ls7_array==no_data_value): ls7_array = fill_array_nodata(ls7_array, no_data_value)
        if ls7_array.shape != diff_array.shape: print('ERROR! LS7 {} feature raster does not match target raster'.format(var.upper()))
        df[ls7_name] = ls7_array.flatten()
    
    # Feature Set 5: Landsat 8 - eleven (11) bands & twelve (12) derivatives available - all based on median composite rasters (gap-filled using lower-quality composite, resampled using cubicspline)
    for var in ls8_vars:
        ls8_name = 'ls8_{}'.format(var.lower())
        ls8_path = '{}/{}/LS8_{}_{}_{}_Bounded.tif'.format(folder_ls8, zone, var, query_window, zone)
        ls8_array = geotiff_to_array(ls8_path)
        if np.any(ls8_array==no_data_value): ls8_array = fill_array_nodata(ls8_array, no_data_value)
        if ls8_array.shape != diff_array.shape: print('ERROR! LS8 {} feature raster does not match target raster'.format(var.upper()))
        df[ls8_name] = ls8_array.flatten()
    
    # Feature Set 6: Global Surface Water - only the 'occurrence' raster seems to be relevant (using cubicspline for resampling, as above)
    for var in gsw_vars:
        gsw_name = 'gsw_{}'.format(var.lower())
        gsw_path = '{}/{}/GSW_{}_{}_cubicspline_Bounded.tif'.format(folder_gsw, zone, var, zone)
        gsw_array = geotiff_to_array(gsw_path)
        if np.any(gsw_array==no_data_value): gsw_array = fill_array_nodata(gsw_array, no_data_value)
        if gsw_array.shape != diff_array.shape: print('ERROR! GSW {} feature raster does not match target raster'.format(var.upper()))
        df[gsw_name] = gsw_array.flatten()
    
    # Feature Set 7: OpenStreetMap (building footprints, road networks, and bridges)
    for var in osm_vars:
        osm_name = 'osm_{}'.format(var.lower())
        osm_path = '{}/{}/OSM_{}_{}.tif'.format(folder_osm, zone, var, zone)
        osm_array = geotiff_to_array(osm_path)
        if np.any(osm_array==no_data_value): osm_array = fill_array_nodata(osm_array, no_data_value)
        if osm_array.shape != diff_array.shape: print('ERROR! OSM {} feature raster does not match target raster'.format(var.upper()))
        df[osm_name] = osm_array.flatten()
    
    # Feature Set 8: Global Canopy Heights (using the cubicspline resampling method)
    gch_name = 'gch'
    gch_path = '{}/{}/GCH_{}_cubicspline_Bounded.tif'.format(folder_gch, zone, zone)
    gch_array = geotiff_to_array(gch_path)
    if np.any(gch_array==no_data_value): gch_array = fill_array_nodata(gch_array, no_data_value)
    if gch_array.shape != diff_array.shape: print('ERROR! GCH feature raster does not match target raster')
    df[gch_name] = gch_array.flatten()
    
    # Feature Set 9: Global Forest Cover (using the cubicspline resampling method)
    gfc_name = 'gfc'
    gfc_path = '{}/{}/GFC_treecover2000_{}_cubicspline_Bounded.tif'.format(folder_gfc, zone, zone)
    gfc_array = geotiff_to_array(gfc_path)
    if np.any(gfc_array==no_data_value): gfc_array = fill_array_nodata(gfc_array, no_data_value)
    if gfc_array.shape != diff_array.shape: print('ERROR! GFC feature raster does not match target raster')
    df[gfc_name] = gfc_array.flatten()
    
    # Feature Set 10: VIIRS Night-Time Light (using the cubicspline resampling method)
    viirs_name = 'ntl_viirs'
    viirs_path = '{}/{}/NTL_VIIRS_{}_cubicspline.tif'.format(folder_viirs, zone, zone)
    viirs_array = geotiff_to_array(viirs_path)
    if np.any(viirs_array==no_data_value): viirs_array = fill_array_nodata(viirs_array, no_data_value)
    if viirs_array.shape != diff_array.shape: print('ERROR! VIIRS NTL feature raster does not match target raster')
    df[viirs_name] = viirs_array.flatten()
    
    # Feature Set 11: DMSP Night-Time Light (using the cubicspline resampling method)
    for var in dmsp_vars:
        dmsp_name = 'dmsp_{}'.format(var.lower())
        dmsp_path = '{}/{}/NTL_DMSP_{}_{}_cubicspline{}.tif'.format(folder_dmsp, zone, var, zone, '_Bounded' if var=='pct_lights' else '')
        dmsp_array = geotiff_to_array(dmsp_path)
        if np.any(dmsp_array==no_data_value): dmsp_array = fill_array_nodata(dmsp_array, no_data_value)
        if dmsp_array.shape != diff_array.shape: print('ERROR! DMSP NTL {} feature raster does not match target raster'.format(var.upper()))
        df[dmsp_name] = dmsp_array.flatten()
    
    # Land Cover classes (LCDB v5.0): 1D dataset only
    lc_name = 'lcdb'
    lc_path = '{}/LCDB_GroupID_{}.tif'.format(folder_lcdb, zone)
    lc_array = geotiff_to_array(lc_path)
    if lc_array.shape != diff_array.shape: print('ERROR! LCDB raster does not match target raster')
    df[lc_name] = lc_array.flatten()
    
    # Save 1D data for that zone to a CSV
    df.to_csv('{}/Input1D_ByZone_{}.csv'.format(folder_input_1D, zone), index=False)


###############################################################################
# 7. Process intact feature & target test datasets for each zone (2D arrays)  #
###############################################################################

# Loop through each zone, preparing 2D inputs
#  - Note: TSM17_GLB is too big to hold in memory - not necessary anyway, as test zones are in MRL18_WPE, MRL18_WVL & TSM16_ATG
for zone in [z for z in zones if z != 'TSM17_GLB']:
    
    print('\nProcessing 2D array data for {} zone:'.format(zone))
    
    # Initialise dictionaries to track the target & feature patch generation for current zone
    zone_target_patch_dict = {'zone':[], 'idx_all':[], 'i':[], 'j':[], 'geometry':[]}
    zone_feature_patch_dict = {'zone':[], 'idx_all':[], 'i':[], 'j':[], 'geometry':[]}
    
    # Initialise lists to hold arrays of all extracted patch data
    zone_target_patches_list = []
    zone_feature_patches_list = []
    
    # Initialise tracker of patch index numbers
    idx_all = 0
    
    # Target: SRTM-DTM DIFF (using resampled 30m DTM based on median of 1m cells)
    diff_path = '{}/{}/SRTM_{}_Median_Diff_Pad44.tif'.format(folder_srtm, zone, zone)
    diff_array = geotiff_to_array(diff_path)
    diff_props = get_geotiff_props(diff_path)
    
    # Calculate the number of patches available in the x & y dimensions (for that particular zone)
    n_patches_x = np.ceil((diff_props['width']-2*pad)/dim_out).astype(np.int16)
    n_patches_y = np.ceil((diff_props['height']-2*pad)/dim_out).astype(np.int16)
    
    # Starting at the top & moving down:
    for j in range(n_patches_y):
        
        print('{}/{}...'.format(j+1, n_patches_y))
        
        # Starting at the left & moving right:
        for i in range(n_patches_x):
            
            # Get the geometry of the target patch (smaller)
            zone_target_patch_wkt = get_target_patch_geom(i, j, diff_props, pad, dim_out)
            zone_target_patch_dict['geometry'].append(zone_target_patch_wkt)
            zone_target_patch_dict['zone'].append(zone)
            zone_target_patch_dict['idx_all'].append(idx_all)
            zone_target_patch_dict['i'].append(i)
            zone_target_patch_dict['j'].append(j)
                    
            # Get the geometry of the feature patch (larger)
            zone_feature_patch_wkt = get_feature_patch_geom(i, j, diff_props, pad, dim_in, dim_out)
            zone_feature_patch_dict['geometry'].append(zone_feature_patch_wkt)
            zone_feature_patch_dict['zone'].append(zone)
            zone_feature_patch_dict['idx_all'].append(idx_all)
            zone_feature_patch_dict['i'].append(i)
            zone_feature_patch_dict['j'].append(j)
            
            # Get the target data for that patch
            zone_target_patch_array = get_target_patch_array(i, j, diff_array, pad, dim_out)
            
            # Pad zone_target_patch_array with no_data values if necessary
            if zone_target_patch_array.shape != (dim_out, dim_out): zone_target_patch_array = pad_array(zone_target_patch_array, dim_out, no_data_value)
            
            # Append array of target data available for this patch to the list
            zone_target_patches_list.append(zone_target_patch_array)
            
            # Increment patch index count
            idx_all += 1
                            
            # 2D data: Initialise a list to hold all feature arrays available for this particular patch
            zone_feature_patch_arrays = []
            
            # Feature Set 1: SRTM data & derivatives
            for var in srtm_vars:
                srtm_name = 'srtm_{}'.format(var.lower())
                srtm_path = '{}/{}/SRTM_{}_{}_Pad44.tif'.format(folder_srtm, zone, zone, var)
                srtm_array = geotiff_to_array(srtm_path)
                if np.any(srtm_array==no_data_value): srtm_array = fill_array_nodata(srtm_array, no_data_value)
                if srtm_array.shape != diff_array.shape: print('ERROR! SRTM {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, srtm_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            # Feature Set 2: ASTER data & derivatives
            for var in aster_vars:
                aster_name = 'aster_{}'.format(var.lower())
                aster_path = '{}/{}/ASTER_{}_{}_Pad44.tif'.format(folder_aster, zone, zone, var.capitalize())
                aster_array = geotiff_to_array(aster_path)
                if np.any(aster_array==no_data_value): aster_array = fill_array_nodata(aster_array, no_data_value)
                if aster_array.shape != diff_array.shape: print('ERROR! ASTER {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, aster_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            # Feature Set 3: AW3D30 data & derivatives
            for var in aw3d30_vars:
                aw3d30_name = 'aw3d30_{}'.format(var.lower())
                aw3d30_path = '{}/{}/AW3D30_{}_{}_Pad44.tif'.format(folder_aw3d30, zone, zone, var.capitalize())
                aw3d30_array = geotiff_to_array(aw3d30_path)
                if np.any(aw3d30_array==no_data_value): aw3d30 = fill_array_nodata(aw3d30_array, no_data_value)
                if aw3d30_array.shape != diff_array.shape: print('ERROR! AW3D30 {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, aw3d30_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            # Feature Set 4: Landsat 7 - nine (9) bands (incl both Band 6 options) & twelve (12) derivatives available - all based on median composite rasters (gap-filled using lower-quality composite, resampled using cubicspline)
            for var in ls7_vars:
                ls7_name = 'ls7_{}'.format(var.lower())
                ls7_path = '{}/{}/LS7_{}_{}_{}_Pad44_Bounded.tif'.format(folder_ls7, zone, var, query_window, zone)
                ls7_array = geotiff_to_array(ls7_path)
                if np.any(ls7_array==no_data_value): ls7_array = fill_array_nodata(ls7_array, no_data_value)
                if ls7_array.shape != diff_array.shape: print('ERROR! LS7 {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, ls7_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            # Feature Set 5: Landsat 8 - eleven (11) bands & twelve (12) derivatives available - all based on median composite rasters (gap-filled using lower-quality composite, resampled using cubicspline)
            for var in ls8_vars:
                ls8_name = 'ls8_{}'.format(var.lower())
                ls8_path = '{}/{}/LS8_{}_{}_{}_Pad44_Bounded.tif'.format(folder_ls8, zone, var, query_window, zone)
                ls8_array = geotiff_to_array(ls8_path)
                if np.any(ls8_array==no_data_value): ls8_array = fill_array_nodata(ls8_array, no_data_value)
                if ls8_array.shape != diff_array.shape: print('ERROR! LS8 {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, ls8_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            # Feature Set 6: Global Surface Water - only the 'occurrence' raster seems to be relevant (using cubicspline for resampling, as above)
            for var in gsw_vars:
                gsw_name = 'gsw_{}'.format(var.lower())
                gsw_path = '{}/{}/GSW_{}_{}_cubicspline_Bounded_Pad44.tif'.format(folder_gsw, zone, var, zone)
                gsw_array = geotiff_to_array(gsw_path)
                if np.any(gsw_array==no_data_value): gsw_array = fill_array_nodata(gsw_array, no_data_value)
                if gsw_array.shape != diff_array.shape: print('ERROR! GSW {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, gsw_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            # Feature Set 7: OpenStreetMap (building footprints, road networks, and bridges)
            for var in osm_vars:
                osm_name = 'osm_{}'.format(var.lower())
                osm_path = '{}/{}/OSM_{}_{}_Pad44.tif'.format(folder_osm, zone, var, zone)
                osm_array = geotiff_to_array(osm_path)
                if np.any(osm_array==no_data_value): osm_array = fill_array_nodata(osm_array, no_data_value)
                if osm_array.shape != diff_array.shape: print('ERROR! OSM {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, osm_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            # Feature Set 8: Global Canopy Heights (using the cubicspline resampling method)
            gch_name = 'gch'
            gch_path = '{}/{}/GCH_{}_cubicspline_Bounded_Pad44.tif'.format(folder_gch, zone, zone)
            gch_array = geotiff_to_array(gch_path)
            if np.any(gch_array==no_data_value): gch_array = fill_array_nodata(gch_array, no_data_value)
            if gch_array.shape != diff_array.shape: print('ERROR! GCH feature raster does not match target raster')
            # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
            feature_patch_array = get_feature_patch_array(i, j, gch_array, dim_in, dim_out)
            if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
            zone_feature_patch_arrays.append(feature_patch_array)
            
            # Feature Set 9: Global Forest Cover (using the cubicspline resampling method)
            gfc_name = 'gfc'
            gfc_path = '{}/{}/GFC_treecover2000_{}_cubicspline_Bounded_Pad44.tif'.format(folder_gfc, zone, zone)
            gfc_array = geotiff_to_array(gfc_path)
            if np.any(gfc_array==no_data_value): gfc_array = fill_array_nodata(gfc_array, no_data_value)
            if gfc_array.shape != diff_array.shape: print('ERROR! GFC feature raster does not match target raster')
            # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
            feature_patch_array = get_feature_patch_array(i, j, gfc_array, dim_in, dim_out)
            if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
            zone_feature_patch_arrays.append(feature_patch_array)
            
            # Feature Set 10: VIIRS Night-Time Light (using the cubicspline resampling method)
            viirs_name = 'ntl_viirs'
            viirs_path = '{}/{}/NTL_VIIRS_{}_cubicspline_Pad44.tif'.format(folder_viirs, zone, zone)
            viirs_array = geotiff_to_array(viirs_path)
            if np.any(viirs_array==no_data_value): viirs_array = fill_array_nodata(viirs_array, no_data_value)
            if viirs_array.shape != diff_array.shape: print('ERROR! VIIRS NTL feature raster does not match target raster')
            # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
            feature_patch_array = get_feature_patch_array(i, j, viirs_array, dim_in, dim_out)
            if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
            zone_feature_patch_arrays.append(feature_patch_array)
            
            # Feature Set 11: DMSP Night-Time Light (using the cubicspline resampling method)
            for var in dmsp_vars:
                dmsp_name = 'dmsp_{}'.format(var.lower())
                dmsp_path = '{}/{}/NTL_DMSP_{}_{}_cubicspline_{}Pad44.tif'.format(folder_dmsp, zone, var, zone, 'Bounded_' if var=='pct_lights' else '')
                dmsp_array = geotiff_to_array(dmsp_path)
                if np.any(dmsp_array==no_data_value): dmsp_array = fill_array_nodata(dmsp_array, no_data_value)
                if dmsp_array.shape != diff_array.shape: print('ERROR! DMSP NTL {} feature raster does not match target raster'.format(var.upper()))
                # 2D data: Extract padded data array corresponding to the current patch & append it to the patch list
                feature_patch_array = get_feature_patch_array(i, j, dmsp_array, dim_in, dim_out)
                if feature_patch_array.shape != (dim_in, dim_in): feature_patch_array = pad_array(feature_patch_array, dim_in, no_data_value)
                zone_feature_patch_arrays.append(feature_patch_array)
                
            
            # 2D data: Append 'zone_feature_patch_arrays' list to the overall 'zone_feature_patches_list' list
            zone_feature_patches_list.append(np.array(zone_feature_patch_arrays))
    
    # Save dictionaries describing target & feature patches to pickle objects, in case they're needed later
    print('All patches generated')
    pickle.dump(zone_target_patch_dict, open('{}/patches/target_patch_dict_ByZone_{}.p'.format(folder_logs, zone), 'wb'))
    pickle.dump(zone_feature_patch_dict, open('{}/patches/feature_patch_dict_ByZone_{}.p'.format(folder_logs, zone), 'wb'))
    
    # Update zone target patch dataframe; save as CSV & SHP (for reference when rebuilding intact prediction array)
    zone_target_patch_df = pd.DataFrame(zone_target_patch_dict).set_index('idx_all', drop=True)
    zone_target_patch_df['geometry'] = zone_target_patch_df['geometry'].apply(wkt.loads)
    zone_target_patch_df.to_csv('{}/patches/target_patch_df_ByZone_{}.csv'.format(folder_logs, zone))
    zone_target_patch_gdf = gpd.GeoDataFrame(zone_target_patch_df, geometry='geometry', crs=diff_props['proj'])
    zone_target_patch_gdf.to_file('{}/patches/SHP/target_patch_ByZone_{}.shp'.format(folder_logs, zone))
    del zone_target_patch_dict, zone_target_patch_df, zone_target_patch_gdf
    
    # Update feature patch dataframe; save as CSV & SHP (for reference when rebuilding intact prediction array)
    zone_feature_patch_df = pd.DataFrame(zone_feature_patch_dict).set_index('idx_all', drop=True)
    zone_feature_patch_df['geometry'] = zone_feature_patch_df['geometry'].apply(wkt.loads)
    zone_feature_patch_df.to_csv('{}/patches/feature_patch_df_ByZone_{}.csv'.format(folder_logs, zone))
    zone_feature_patch_gdf = gpd.GeoDataFrame(zone_feature_patch_df, geometry='geometry', crs=diff_props['proj'])
    zone_feature_patch_gdf.to_file('{}/patches/SHP/feature_patch_ByZone_{}.shp'.format(folder_logs, zone))
    del zone_feature_patch_dict, zone_feature_patch_df, zone_feature_patch_gdf
    
    # Process & save 2D array stacks: note - in keras-tensorflow, default ordering is 'channels_last': (batch, height, width, channels)
    
    # Target patches: Stack all target patches in one array, adjust dimensions to match keras input convention, & save to numpy file
    zone_target_stack = np.array(zone_target_patches_list, dtype=np.float32)
    del zone_target_patches_list
    zone_target_stack = np.expand_dims(zone_target_stack, axis=-1)             # Add extra dimension right at the end
    np.save('{}/Input2D_Target_ByZone_{}.npy'.format(folder_input_2D, zone), zone_target_stack)
    del zone_target_stack
    
    # Feature patches: Stack all feature patches in one array, check its dimensions & save to numpy file
    zone_feature_stack = np.array(zone_feature_patches_list, dtype=np.float32)
    del zone_feature_patches_list
    zone_feature_stack = np.moveaxis(zone_feature_stack, 1, -1)                # Rearrange array axes to match 'channels_last' convention
    np.save('{}/Input2D_Features_ByZone_{}.npy'.format(folder_input_2D, zone), zone_feature_stack)
    del zone_feature_stack