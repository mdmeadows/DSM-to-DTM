# Process: Landsat 7 multi-spectral imagery

# Import required packages
import os, sys, subprocess
import requests
import gzip
import pandas as pd
import shutil
import datetime
import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, get_geotiff_props, get_geotiff_projection, get_geotiff_nodatavalue, geotiff_to_array, array_to_geotiff, ls_subfolder_valid, ls7_spectral_index, create_bounded_geotiff

# List paths to GDAL scripts
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'
gdal_merge = 'C:/Anaconda3/envs/geo/Scripts/gdal_merge.py'
gdal_buildvrt = 'C:/Anaconda3/envs/geo/Library/bin/gdalbuildvrt.exe'

# Define path to relevant folders
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'
folder_dtm = 'E:/mdm123/D/data/DTM/proc/'
folder_ls7 = 'E:/mdm123/D/data/Landsat7/'
folder_archive = 'V:/mdm123/Landsat7/'
folder_fig = 'E:/mdm123/D/figures/'

# Define relevant parameters of SRTM data collection & search window desired
survey_start = datetime.date(2000, 2, 11)
survey_end = datetime.date(2000, 2, 22)
survey_window = survey_end - survey_start

# Define the range of survey windows to be tested (the duration of the period for which LS7 data should be processed, centred on the SRTM survey period)
query_windows = [60, 90, 120, 150, 180]   # Number of days

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define dictionary to hold information relating to each zone covered by the Marlborough (2018) survey
dtm_dict = {'MRL18_WPE':{'label':'Wairau Plains East (Marlborough 2018)', 'year':'2018', 'wrs_prs':['073089']},
            'MRL18_WVL':{'label':'Wairau Valley (Marlborough 2018)', 'year':'2018', 'wrs_prs':['073089','074089']},
            'MRL18_WKW':{'label':'Picton - Waikawa (Marlborough 2018)', 'year':'2018', 'wrs_prs':['073089']},
            'MRL18_FGA':{'label':'Flaxbourne, Grassmere & Lower Awatere (Marlborough 2018)', 'year':'2018', 'wrs_prs':['073089']},
            'TSM17_STA':{'label':'St Arnaud (Tasman 2017)', 'year':'2017', 'wrs_prs':['073089','074089']},
            'TSM17_LDM':{'label':'Lee Dam (Tasman 2017)', 'year':'2017', 'wrs_prs':['073089','074089']},
            'TSM17_GLB':{'label':'Golden Bay & Farewell Spit (Tasman 2017)', 'year':'2017', 'wrs_prs':['074088']},
            'TSM16_ATG':{'label':'Abel Tasman & Golden Bay (Tasman 2016)', 'year':'2016', 'wrs_prs':['074088','074089']}}

# Check full list of WRS path-row tiles needed
wrs_prs_all = set([wrs_pr for wrs_prs in [dtm_dict[zone]['wrs_prs'] for zone in zones] for wrs_pr in wrs_prs])     # ('073089', '074088', '074089')

# Define the number of cells of padding to add along each raster boundary
pad = 44

# Define the no_data value to be used for GeoTIFF creation
no_data_value = -9999


###############################################################################
# 1. Archive to Public folder any scenes for which cloud cover (land) > 85%   #
###############################################################################

# Define a threshold for the maximum cloud coverage (over land) to be allowed
cloud_cover_max = 85

# Download the up-to-date Landsat 7 metadata from USGS
ls7_metadata_url = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_ETM_C1.csv.gz'
ls7_metadata_read = requests.get(ls7_metadata_url)
ls7_metadata_path = '{}metadata/{}'.format(folder_ls7, ls7_metadata_url.split('/')[-1])
with open(ls7_metadata_path, 'wb') as ls7_metadata_write:
    ls7_metadata_write.write(ls7_metadata_read.content)

# Extract the compressed file (.gz) & read the CSV inside into a pandas dataframe
with gzip.open(ls7_metadata_path) as ls7_metadata_csv:
    ls7_metadata = pd.read_csv(ls7_metadata_csv)

# Initialise a count of valid & invalid scenes (wrt cloud cover over land)
valid_scenes = 0
invalid_scenes = 0

# Loop through each subfolder in the Level-2 folder, checking its cloud cover over land
print('Archiving scenes for which cloud cover (over land) > {}%:'.format(cloud_cover_max))
for local_subfolder in os.listdir('{}raw/Level2'.format(folder_ls7)):
    
    # Process further if path is to a folder (rather than a file)
    if os.path.isdir('{}raw/Level2/{}'.format(folder_ls7, local_subfolder)):
        
        # Check the cloud cover (over land) associated with that scene
        cloud_cover = ls7_metadata.loc[ls7_metadata['LANDSAT_PRODUCT_ID']==local_subfolder, 'CLOUD_COVER_LAND'].values[0]
        
        # Update scene counts based on whether cloud cover exceeds threshold or not
        if cloud_cover > cloud_cover_max:
            
            print(' - {} ({:.1f}%):'.format(local_subfolder, cloud_cover), end=' ')
            
            # Update count of invalid scenes
            invalid_scenes += 1
            
            # Level-2 subfolder: move from Local to Public archive folder
            l2_local = '{}raw/Level2/{}'.format(folder_ls7, local_subfolder)
            l2_archive = '{}raw/Level2/{}'.format(folder_archive, local_subfolder)
            shutil.move(l2_local, l2_archive)
            print('Level-2', end=' ')
            
            # Level-1: move the corresponding subfolder from Local to Public archive folder too
            l1_local = '{}raw/Level1/{}'.format(folder_ls7, local_subfolder)
            l1_archive = '{}raw/Level1/{}'.format(folder_archive, local_subfolder)
            shutil.move(l1_local, l1_archive)
            print('Level-1')
            
        else:
            valid_scenes += 1

print('\n{} scenes retained, with {} scenes rejected (cloud cover over land > {}%) & archived'.format(valid_scenes, invalid_scenes, cloud_cover_max))


###############################################################################
# 2. Archive redundant files to Public folder (to save space on local drive)  #
###############################################################################

# Bands 1-5 & 7 are available as Level-2 products (surface reflectance) so aren't required in Level-1 folders
for local_subfolder in os.listdir('{}raw/Level1'.format(folder_ls7)):
    
    # Process further if path is to a folder (rather than a file)
    if os.path.isdir('{}raw/Level1/{}'.format(folder_ls7, local_subfolder)):
        
        # If a corresponding subfolder doesn't exist in the Public folder, create one
        archive_path = '{}raw/Level1/{}'.format(folder_archive, local_subfolder)
        if not os.path.exists(archive_path): os.mkdir(archive_path)
        
        # Move rasters corresponding to Bands 1-5 & 7
        for f in os.listdir('{}raw/Level1/{}'.format(folder_ls7, local_subfolder)):
            if f.lower().endswith('b1.tif') or f.lower().endswith('b2.tif') or f.lower().endswith('b3.tif') or f.lower().endswith('b4.tif') or f.lower().endswith('b5.tif') or f.lower().endswith('b7.tif'):
                shutil.move('{}raw/Level1/{}/{}'.format(folder_ls7, local_subfolder, f), '{}/{}'.format(archive_path, f))


###############################################################################
# 3. Develop cloud-free composites (original grids) for all temporal windows  #
###############################################################################

# Define lists of the bands in each collection
l2_bands = ['1', '2', '3', '4', '5', '7']
l1_bands = ['6_VCID_1', '6_VCID_2', '8']

# Define the QA pixel values considered acceptable (relevant to SR Bands 1-5 & 7 of Level 2 products). See https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_SurfaceReflectance-LEDAPS_ProductGuide-v2.pdf
pqa_acceptable_values_hq = [66, 130, 68, 132]                                  # High quality filter: only entries for "Clear" & "Water"
pqa_acceptable_values_lq = pqa_acceptable_values_hq + [80,96,160,224]          # Low quality filter: also allow low/medium/high-confidence cloud codes without shadow or saturation (required to fill all gaps)

# Define the BQA pixel values considered acceptable (relevant to Bands 6 & 8 of Level 1 products). See https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con
bqa_acceptable_values_hq = [672, 676, 680, 684]                                # High quality filter: only entries for "Clear"
bqa_acceptable_values_lq = bqa_acceptable_values_hq + [704,708,712,716,752]    # Low quality filter: also allow medium/high cloud confidence codes (without saturation or shadows)

# Loop through each of the DTM survey zones
for zone in zones:
    
    print('\nProcessing Landsat 7 imagery for {}...'.format(dtm_dict[zone]['label']))
    
    # Create necessary result folder if it doesn't yet exist
    folder_original_zone = '{}proc/cloudfree/Original/{}/'.format(folder_ls7, zone)
    if not os.path.exists(folder_original_zone): os.mkdir(folder_original_zone)
    
    # Define path to the padded GeoTIFF describing that zone's SRTM coverage
    srtm_tif = '{}proc/{}/SRTM_{}_Z_Pad44.tif'.format(folder_srtm, zone, zone)
    srtm_proj = get_geotiff_projection(srtm_tif)
    
    # Define which Ls7 WRS path-rows are relevant for that zone
    zone_wrs_prs = dtm_dict[zone]['wrs_prs']
    
    # Loop through the list of possible query windows
    for query_window in query_windows:
        
        print(' - Using a {} day search window'.format(query_window))
        
        # Define the period over which S2 data should be sought (centred on each survey's collection period)
        query_window = datetime.timedelta(days=query_window)
        
        # Determine search window for tiles, by buffering the start & end dates retrieved above
        query_start = survey_start - (query_window - survey_window)/2
        query_end = survey_end + (query_window - survey_window)/2
    
        # Loop through all bands, setting parameters as appropriate (based on whether band is from Level-1 or Level-2 collection)
        for band in l2_bands + l1_bands:
            
            # Define collection level & folder containing raw data for processing & lists of QA codes for processing
            level = 2 if band in l2_bands else 1
            folder = '{}raw/Level{}/'.format(folder_ls7, level)
            print('   - Band {} (Level-{} data):'.format(band, level), end=' ')
            
            # Assign other raster processing parameters based on collection level
            if level == 2:
                # Level-2 collection
                band_prefix = 'sr_band'
                qa_suffix = 'pixel_qa'
                qa_acceptable_values_hq, qa_acceptable_values_lq = pqa_acceptable_values_hq, pqa_acceptable_values_lq
                valid_from = 0.
                valid_to = 10000.
            elif level == 1:
                # Level-1 collection
                band_prefix = 'B'
                qa_suffix = 'BQA'
                qa_acceptable_values_hq, qa_acceptable_values_lq = bqa_acceptable_values_hq, bqa_acceptable_values_lq
                valid_from = 0.
                valid_to = 255.
            
            # Get list of rasters available in selected folder
            subfolders = os.listdir(folder)
            
            # Loop through subfolders, clipping each based on extent info extracted above, masking based on QA raster, and adding to a list of masked arrays
            
            # Initialise lists to hold masked rasters relating to that band
            band_masked_arrays_hq = []
            band_masked_arrays_lq = []
            
            # Initialise a count of the number of valid layers available
            n_valid_layers = 0
            
            # Loop through each available subfolder, extracting relevant data if it's within search window
            for subfolder in subfolders:
                
                # Process data if capture date is within search window
                if ls_subfolder_valid(subfolder, zone_wrs_prs, query_start, query_end):
                    
                    # Update the count of valid layers
                    n_valid_layers += 1
                    
                    # Define paths to the band raster & its associated QA raster
                    band_path = '{}{}/{}_{}{}.tif'.format(folder, subfolder, subfolder, band_prefix, band)
                    qa_path = '{}{}/{}_{}.tif'.format(folder, subfolder, subfolder, qa_suffix)
                    
                    # Get the projection properties of the band raster
                    band_proj = get_geotiff_projection(band_path)
                    band_props = get_geotiff_props(band_path)
                    
                    # Project the padded GeoTIFF describing that zone's SRTM coverage into the band's CRS
                    srtm_projected = '{}proc/{}/SRTM_{}_Z_Pad44_Projected.tif'.format(folder_srtm, zone, zone)
                    if os.path.exists(srtm_projected): os.remove(srtm_projected)
                    project_command = [gdal_warp, '-s_srs', srtm_proj, '-t_srs', band_proj, srtm_tif, srtm_projected]
                    project_result = subprocess.run(project_command, stdout=subprocess.PIPE)
                    if project_result.returncode != 0:
                        print('\nProcess failed, with error message: {}\n'.format(project_result.stdout))
                        break
                    
                    # Get the projected GeoTIFF's extent (in the CRS used by the input band)
                    srtm_projected_props = get_geotiff_props(srtm_projected)
                    os.remove(srtm_projected)
                    AOI_x_min = srtm_projected_props['x_min']
                    AOI_x_max = srtm_projected_props['x_max']
                    AOI_y_min = srtm_projected_props['y_min']
                    AOI_y_max = srtm_projected_props['y_max']
                    
                    # Snap these extents to match the band raster's grid, adding a buffer of 50 cells along each edge to be safe
                    buffer = 50
                    clip_x_min = band_props['x_min'] + ceil((AOI_x_min - band_props['x_min'])/band_props['res_x'])*band_props['res_x'] - buffer*band_props['res_x']
                    clip_x_max = band_props['x_max'] + floor((AOI_x_max - band_props['x_max'])/band_props['res_x'])*band_props['res_x'] + buffer*band_props['res_x']
                    clip_y_min = band_props['y_min'] + floor((AOI_y_min - band_props['y_min'])/-band_props['res_y'])*-band_props['res_y'] - buffer*-band_props['res_y']
                    clip_y_max = band_props['y_max'] + ceil((AOI_y_max - band_props['y_max'])/-band_props['res_y'])*-band_props['res_y'] + buffer*-band_props['res_y']
                    
                    # Clip the band raster such that its extent matches AOI but its grid alignment matches original band raster
                    band_clip = '{}{}/{}_{}{}_{}.tif'.format(folder, subfolder, subfolder, band_prefix, band, zone)
                    warp_band_command = [gdal_warp, '-overwrite', band_path, band_clip, '-s_srs', band_props['proj'], '-t_srs', band_props['proj'], '-tr', str(band_props['res_x']), str(-band_props['res_y']), '-te', str(clip_x_min), str(clip_y_min), str(clip_x_max), str(clip_y_max), '-te_srs', band_props['proj'], '-r', 'near', '-dstnodata', '-9999']
                    subprocess.call(warp_band_command)
                    
                    # Clip the QA raster too, to be used as the mask
                    qa_clip = '{}{}/{}_{}_{}.tif'.format(folder, subfolder, subfolder, qa_suffix, zone)
                    warp_qa_command = [gdal_warp, '-overwrite', qa_path, qa_clip, '-s_srs', band_props['proj'], '-t_srs', band_props['proj'], '-tr', str(band_props['res_x']), str(-band_props['res_y']), '-te', str(clip_x_min), str(clip_y_min), str(clip_x_max), str(clip_y_max), '-te_srs', band_props['proj'], '-r', 'near', '-dstnodata', '-9999']
                    subprocess.call(warp_qa_command)
                    
                    # Read in arrays of the clipped band raster & the clipped QA raster (to use as a mask)
                    band_array = geotiff_to_array(band_clip)
                    qa_array = geotiff_to_array(qa_clip)
                    
                    # Check the no_data_value used by the input band raster
                    band_nodatavalue = get_geotiff_nodatavalue(band_clip)
                    
                    # Mask the band array wherever the band_nodatavalue is present
                    band_masked_array = np.ma.masked_where(band_array==band_nodatavalue, band_array)
                    
                    # Restrict to valid range & rescale
                    # Replace values below lower bound with no_data_value
                    band_masked_array[band_masked_array < valid_from] = no_data_value
                    # Replace values above upper bound with no_data_value
                    band_masked_array[band_masked_array > valid_to] = no_data_value
                    # Re-mask, using the newly applied no_data_value
                    band_masked_array = np.ma.masked_where(band_masked_array==no_data_value, band_masked_array)
                    # Rescale values by appropriate factor
                    band_masked_array = band_masked_array / valid_to
                    
                    # Mask the band array using the HQ version of the QA array
                    band_masked_array_hq = np.ma.masked_where(~np.isin(qa_array, qa_acceptable_values_hq), band_masked_array)
                    band_masked_arrays_hq.append(band_masked_array_hq)
                    del band_masked_array_hq
                    
                    # Set up a second masked band array, using the LQ version of the QA array (for gap-filling later)
                    band_masked_array_lq = np.ma.masked_where(~np.isin(qa_array, qa_acceptable_values_lq), band_masked_array)
                    band_masked_arrays_lq.append(band_masked_array_lq)
                    del band_masked_array_lq
                    
                    # Clean up arrays no longer needed
                    del band_array, qa_array, band_masked_array
            
            print('{} images available'.format(n_valid_layers), end=' ')
            
            # Calculate median of the HQ array list representing data for the selected band
            band_masked_arrays_hq_stack = np.ma.array(band_masked_arrays_hq)
            band_masked_median_hq = np.ma.median(band_masked_arrays_hq_stack, axis=0, overwrite_input=True)
            del band_masked_arrays_hq, band_masked_arrays_hq_stack
            
            # Calculate median of the LQ array list representing data for the selected band
            band_masked_arrays_lq_stack = np.ma.array(band_masked_arrays_lq)
            band_masked_median_lq = np.ma.median(band_masked_arrays_lq_stack, axis=0)
            del band_masked_arrays_lq, band_masked_arrays_lq_stack
            
            # Save results in which the LQ results are used to gap-fill the HQ results
            # Retrieve the mask derived from the HQ filter
            mask_hq = np.ma.getmask(band_masked_median_hq)
            # Make copies of the HQ median results
            band_masked_median_gapfill = np.ma.copy(band_masked_median_hq)
            # Try to fill in masked values using results from the LQ results, where possible
            band_masked_median_gapfill[mask_hq] = band_masked_median_lq[mask_hq]
            del band_masked_median_hq, band_masked_median_lq, mask_hq
            # Fill remaining gaps with the same no_data value used previously
            band_masked_median_gapfill = band_masked_median_gapfill.filled(fill_value=no_data_value)
            
            # Update the dictionary describing band raster properties
            band_props['x_min'] = clip_x_min
            band_props['x_max'] = clip_x_max
            band_props['y_min'] = clip_y_min
            band_props['y_max'] = clip_y_max
            band_props['width'] = int((clip_x_max - clip_x_min)/band_props['res_x'])
            band_props['height'] = int((clip_y_max - clip_y_min)/-band_props['res_y'])
            
            # Save these new arrays to TIF files
            tif_median_path_gapfill = '{}LS7_L{}_B{}_{}d_GapFill_Median_{}.tif'.format(folder_original_zone, level, band, str(query_window.days).zfill(3), zone)
            array_to_geotiff(band_masked_median_gapfill, tif_median_path_gapfill, no_data_value, band_props)
            del band_masked_median_gapfill, band_props
            print('DONE')


###############################################################################
# 4. Calculate spectral index products, using original composite rasters      #
###############################################################################

# Define the list of spectral index products to be generated
spectral_index_products = ['NDVI','EVI','AVI','SAVI','MSAVI','SI','BSI','NDMI','MNDWI','AWEInsh','AWEIsh','NDBI']

# Loop through all available survey zones
for zone in zones:
    
    print('\nCalculating Landsat 7 spectral indices for {}...'.format(dtm_dict[zone]['label']))
    folder_ls7_zone = '{}proc/cloudfree/Original/{}'.format(folder_ls7, zone)
    
    # Loop through all considered query windows
    for query_window in query_windows:
        
        print(' - Indices based on {}d window:'.format(query_window), end=' ')
        
        # Define paths to relevant band rasters
        band_1_path = '{}/LS7_L2_B1_{}d_GapFill_Median_{}.tif'.format(folder_ls7_zone, str(query_window).zfill(3), zone)
        band_2_path = '{}/LS7_L2_B2_{}d_GapFill_Median_{}.tif'.format(folder_ls7_zone, str(query_window).zfill(3), zone)
        band_3_path = '{}/LS7_L2_B3_{}d_GapFill_Median_{}.tif'.format(folder_ls7_zone, str(query_window).zfill(3), zone)
        band_4_path = '{}/LS7_L2_B4_{}d_GapFill_Median_{}.tif'.format(folder_ls7_zone, str(query_window).zfill(3), zone)
        band_5_path = '{}/LS7_L2_B5_{}d_GapFill_Median_{}.tif'.format(folder_ls7_zone, str(query_window).zfill(3), zone)
        band_7_path = '{}/LS7_L2_B7_{}d_GapFill_Median_{}.tif'.format(folder_ls7_zone, str(query_window).zfill(3), zone)
        
        # Read data arrays for each into memory
        b1, b2, b3, b4, b5, b7 = [geotiff_to_array(band_path) for band_path in [band_1_path, band_2_path, band_3_path, band_4_path, band_5_path, band_7_path]]
        b_props = get_geotiff_props(band_1_path)
        
        # Loop through all spectral index products
        for product in spectral_index_products:
            # Define output path
            product_path = '{}/LS7_{}_{}d_{}.tif'.format(folder_ls7_zone, product, str(query_window).zfill(3), zone)
            # Use Landsat 7 function to calculate the spectral index (including gap-filling)
            with np.errstate(divide='ignore'):  # Divide by zero issues are handled already - suppressing warnings
                ls7_spectral_index(b1, b2, b3, b4, b5, b7, product, b_props, product_path, no_data_value)
            print(product, end=' ')
        
        # Build a VRT (Virtual Dataset) containing the R,G,B bands - for visualisation as a True Colour Image (TCI)
        vrt_path = '{}/LS7_TCI_{}d_{}.vrt'.format(folder_ls7_zone, str(query_window).zfill(3), zone)
        vrt_command = [gdal_buildvrt, '-separate', vrt_path] + ['{}/LS7_L2_B{}_{}d_GapFill_Median_{}.tif'.format(folder_ls7_zone, band, str(query_window).zfill(3), zone) for band in ['3','2','1']]
        vrt_result = subprocess.run(vrt_command, stdout=subprocess.PIPE)
        if vrt_result.returncode != 0:
            print(vrt_result.stdout)
            break
        print('TCI')


###############################################################################
# 5. Resample cloud-free composites (all time windows) to match SRTM grids    #
###############################################################################

# The 'cubicspline' resampling method was found to provide the best results based on visual comparisons
resampling = 'cubicspline'

# Loop through all zones, resampling all available Landsat 7 imagery to arrays coincident with the SRTM arrays
for zone in zones:
    
    print('\nProcessing Landsat 7 data for {}...'.format(dtm_dict[zone]['label']))
    
    # 5a. Read the appropriate SRTM DSM raster into memory & retrieve its properties
    print(' - Analysing zonal SRTM raster to align grids...')
    srtm_filename = '{}proc/{}/SRTM_{}_Z.tif'.format(folder_srtm, zone, zone)
    srtm_proj, srtm_res_x, srtm_res_y, srtm_x_min, srtm_x_max, srtm_y_min, srtm_y_max, srtm_width, srtm_height = extract_projection_info(srtm_filename)
    
    # Define a new bounding box, including the padding required for the 2D convnet data pre-processing
    pad_x_min = srtm_x_min - pad*srtm_res_x
    pad_x_max = srtm_x_max + pad*srtm_res_x
    pad_y_min = srtm_y_min - pad*-srtm_res_y
    pad_y_max = srtm_y_max + pad*-srtm_res_y
    pad_width = srtm_width + 2*pad
    pad_height = srtm_height + 2*pad
    
    # Create new folder for resampled rasters, if it doesn't exist already
    folder_ls7_zone = '{}proc/cloudfree/Resampled/{}'.format(folder_ls7, zone)
    if not os.path.exists(folder_ls7_zone):
        os.makedirs(folder_ls7_zone)
    
    # 5b. Resample each band, setting parameters as appropriate (based on whether band is from Level-1 or Level-2 collection)
    for band in l2_bands + l1_bands:
        
        # Define collection level
        level = 2 if band in l2_bands else 1
        print(' - Band {} (Level-{} data):'.format(band, level), end=' ')
        
        # Loop through the processed results, for the various time windows trialled
        for query_window in query_windows:
            
            # Open band raster and extract its coordinate reference system (CRS)
            ls7_tile_path = '{}proc/cloudfree/Original/{}/LS7_L{}_B{}_{}d_GapFill_Median_{}.tif'.format(folder_ls7, zone, level, band, str(query_window).zfill(3), zone)
            ls7_proj = get_geotiff_projection(ls7_tile_path)
            
            # Warp band raster to WGS84, aligning with padded SRTM raster
            ls7_resample_pad = '{}/LS7_B{}_{}d_{}_Pad44.tif'.format(folder_ls7_zone, band, str(query_window).zfill(3), zone)
            warp_command_pad = [gdal_warp, '-overwrite', ls7_tile_path, ls7_resample_pad, '-s_srs', ls7_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', srtm_proj, '-r', resampling, '-dstnodata', str(no_data_value)]
            warp_result_pad = subprocess.run(warp_command_pad, stdout=subprocess.PIPE)
            if warp_result_pad.returncode != 0:
                print(warp_result_pad.stdout)
                break
            
            # Now clip that padded raster to the unpadded extent (i.e. same as SRTM/DTM zone raster)
            ls7_resample = '{}/LS7_B{}_{}d_{}.tif'.format(folder_ls7_zone, band, str(query_window).zfill(3), zone)
            warp_command = [gdal_warp, '-overwrite', ls7_resample_pad, ls7_resample, '-s_srs', srtm_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', srtm_proj, '-r', 'near', '-dstnodata', str(no_data_value)]
            warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
            if warp_result.returncode != 0:
                print(warp_result.stdout)
                break
            
        print('DONE')
        
    # 5c. Resample each index product
    for product in spectral_index_products:
        
        print(' - {} spectral index:'.format(product), end=' ')
        
        # Loop through the processed results, for the three time windows trialled
        for query_window in query_windows:
            
            # Open product raster and extract its coordinate reference system (CRS)
            ls7_tile_path = '{}proc/cloudfree/Original/{}/LS7_{}_{}d_{}.tif'.format(folder_ls7, zone, product, str(query_window).zfill(3), zone)
            ls7_proj = get_geotiff_projection(ls7_tile_path)
            
            # Warp band raster to WGS84, aligning with padded SRTM raster
            ls7_resample_pad = '{}/LS7_{}_{}d_{}_Pad44.tif'.format(folder_ls7_zone, product, str(query_window).zfill(3), zone)
            warp_command_pad = [gdal_warp, '-overwrite', ls7_tile_path, ls7_resample_pad, '-s_srs', ls7_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', srtm_proj, '-r', resampling, '-dstnodata', str(no_data_value)]
            warp_result_pad = subprocess.run(warp_command_pad, stdout=subprocess.PIPE)
            if warp_result_pad.returncode != 0:
                print(warp_result_pad.stdout)
                break
            
            # Now clip that padded raster to the unpadded extent (i.e. same as SRTM/DTM zone raster)
            ls7_resample = '{}/LS7_{}_{}d_{}.tif'.format(folder_ls7_zone, product, str(query_window).zfill(3), zone)
            warp_command = [gdal_warp, '-overwrite', ls7_resample_pad, ls7_resample, '-s_srs', srtm_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', srtm_proj, '-r', 'near', '-dstnodata', str(no_data_value)]
            warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
            if warp_result.returncode != 0:
                print(warp_result.stdout)
                break
            
        print('DONE')


###############################################################################
# 6. Compare band/product value distribution across all available zones       #
###############################################################################

# Loop through all available resampled rasters (bands & products)
for raster in ['B{}'.format(b) for b in l2_bands + l1_bands] + spectral_index_products:
    
    print('\nProcessing {} raster...'.format(raster), end=' ')
    
    # Loop through all available temporary search windows
    for query_window in query_windows:
        
        # Initialise a figure to show that raster's data distribution for all zones
        fig, axes = plt.subplots(nrows=len(zones), sharex=True, figsize=(9,15))
        
        # Loop through all available zones
        for i, zone in enumerate(zones):
            
            # Define path to appropriate raster & retrieve its values
            raster_path = '{}proc/cloudfree/Resampled/{}/LS7_{}_{}d_{}.tif'.format(folder_ls7, zone, raster, str(query_window).zfill(3), zone)
            raster_array = geotiff_to_array(raster_path)
            raster_values = raster_array[raster_array != no_data_value].flatten()
            
            # Show that zone's data distribution using a histogram & label figure with zone name
            axes[i].hist(raster_values, bins=100)
            axes[i].annotate(zone, xy=(0.99, 0.96), xycoords='axes fraction', ha='right', va='top')
            
            # Show extreme values using dashed red lines
            axes[i].axvline(x=min(raster_values), color='red', linestyle='dashed', alpha=0.3)
            axes[i].axvline(x=max(raster_values), color='red', linestyle='dashed', alpha=0.3)
        
        # Finalise figure
        fig.suptitle('Distribution of {} values across all zones ({} day window)'.format(raster, query_window), fontweight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.96)
        fig.savefig('{}All/Distributions/Landsat7/{}_{}d.png'.format(folder_fig, raster, str(query_window).zfill(3)), dpi=300)
        plt.close()
        
        print('{}d'.format(query_window), end=' ')


###############################################################################
# 7. Bound bands & spectral indices, to truncate artefacts from resampling    #
###############################################################################

# Define a dictionary of bounds for each spectral index product
bounds_dict = {'NDVI':(-1., 1.),       # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-difference-vegetation-index
               'EVI':(-1., 1.),        # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-enhanced-vegetation-index
               'AVI':(-1., 1.),
               'SAVI':(-1., 1.),       # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-soil-adjusted-vegetation-index
               'MSAVI':(-1., 1.),      # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-modified-soil-adjusted-vegetation-index
               'SI':(0., 1.),          # Source: based on equation used
               'BSI':(-1., 1.),        # Source: based on equation used
               'NDMI':(-1., 1.),       # Source: https://www.usgs.gov/land-resources/nli/landsat/normalized-difference-moisture-index
               'MNDWI':(-1., 1.),      # Source: Xu 2006
               'AWEInsh':(-7., 4.),    # Source: based on equation used
               'AWEIsh':(-3.25, 3.5),  # Source: based on equation used
               'NDBI':(-1., 1.)}       # Source: based on equation used

# Add bounds for all bands too
for band in l2_bands + l1_bands:
    bounds_dict['B{}'.format(band)] = (0., 1.)

# Loop through all bands & spectral index products
for raster in ['B{}'.format(b) for b in l2_bands + l1_bands] + spectral_index_products:
    
    print('\nBounding {} raster...'.format(raster), end=' ')
    
    # Retrieve the bounds defined for that raster (if any)
    bounds = bounds_dict[raster]
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    
    # Loop through all available temporary search windows
    for query_window in query_windows:
        
        # Loop through all available zones
        for i, zone in enumerate(zones):
            
            # Loop through both versions of each raster (padded & unpadded)
            for padding in ['', '_Pad44']:
                
                # Define path to that raster GeoTIFF
                raster_unbounded = '{}proc/cloudfree/Resampled/{}/LS7_{}_{}d_{}{}.tif'.format(folder_ls7, zone, raster, str(query_window).zfill(3), zone, padding)
                raster_bounded = '{}proc/cloudfree/Resampled/{}/LS7_{}_{}d_{}{}_Bounded.tif'.format(folder_ls7, zone, raster, str(query_window).zfill(3), zone, padding)
                
                # Create a bounded GeoTIFF, using the helper function defined
                with np.errstate(invalid='ignore'):
                    create_bounded_geotiff(raster_unbounded, raster_bounded, lower_bound, upper_bound, no_data_value)
        
        print('{}d'.format(query_window), end=' ')


###############################################################################
# 8. Compare BOUNDED band/product value distribution for all available zones  #
###############################################################################

# Loop through all available resampled rasters (bands & products)
for raster in ['B{}'.format(b) for b in l2_bands + l1_bands] + spectral_index_products:
    
    print('\nProcessing {} raster...'.format(raster), end=' ')
    
    # Loop through all available temporary search windows
    for query_window in query_windows:
        
        # Initialise a figure to show that raster's data distribution for all zones
        fig, axes = plt.subplots(nrows=len(zones), sharex=True, figsize=(9,15))
        
        # Loop through all available zones
        for i, zone in enumerate(zones):
            
            # Define path to appropriate raster & retrieve its values
            raster_path = '{}proc/cloudfree/Resampled/{}/LS7_{}_{}d_{}_Bounded.tif'.format(folder_ls7, zone, raster, str(query_window).zfill(3), zone)
            raster_array = geotiff_to_array(raster_path)
            raster_values = raster_array[raster_array != no_data_value].flatten()
            
            # Show that zone's data distribution using a histogram & label figure with zone name
            axes[i].hist(raster_values, bins=100, color='green', alpha=0.5)
            axes[i].annotate(zone, xy=(0.99, 0.96), xycoords='axes fraction', ha='right', va='top')
            
            # Show extreme values using dashed red lines
            axes[i].axvline(x=min(raster_values), color='red', linestyle='dashed', alpha=0.3)
            axes[i].axvline(x=max(raster_values), color='red', linestyle='dashed', alpha=0.3)
        
        # Finalise figure
        fig.suptitle('Distribution of {} values (bounded) across all zones ({} day window)'.format(raster, query_window), fontweight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.96)
        fig.savefig('{}All/Distributions/Landsat7/{}_{}d_Bounded.png'.format(folder_fig, raster, str(query_window).zfill(3)), dpi=300)
        plt.close()
        
        print('{}d'.format(query_window), end=' ')