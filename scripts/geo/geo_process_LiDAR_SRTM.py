# Process: DTM (LiDAR) & DSM (SRTM) topography data

# Import required packages
import sys, os, subprocess
import geopandas as gpd
from osgeo import osr
import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, geotiff_to_array

# List paths to GDAL scripts
gdal_merge = 'C:/Anaconda3/envs/geo/Scripts/gdal_merge.py'
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_dem = 'C:/Anaconda3/envs/geo/Library/bin/gdaldem.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'
gdal_rasterise = 'C:/Anaconda3/envs/geo/Library/bin/gdal_rasterize.exe'
gdal_polygonise = 'C:/Anaconda3/envs/geo/Scripts/gdal_polygonize.py'
ogr2ogr = 'C:/Anaconda3/envs/geo/Library/bin/ogr2ogr.exe'

# Define path to SRTM, LiDAR DTM & output figure folder
folder_geoid = 'E:/mdm123/D/data/Geoid'
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'
folder_dtm = 'E:/mdm123/D/data/DTM/'
folder_fig = 'E:/mdm123/D/figures/'

# Define the number of cells of padding to add along each raster boundary
pad = 44


###############################################################################
# 1. Process SRTM DSM data - merge tiles & generate derivative rasters        #
###############################################################################

# 1a. Merge SRTM tiles covering Area of Interest

# Get full list of image tiles actually available
tiles_available = [entry.name for entry in os.scandir('{}raw/'.format(folder_srtm)) if entry.name.endswith('.tif')]

# Build gdal_merge command
no_data_in = '-32767'
no_data_out = '-9999'
output_format = 'GTiff'
output_type = 'Int16'
srtm_tif = '{}proc/SRTM_Z.tif'.format(folder_srtm)
merge_command = ['python', gdal_merge, '-n', no_data_in, '-a_nodata', no_data_out, '-ot', output_type, '-of', output_format, '-o', srtm_tif] + ['{}raw/{}'.format(folder_srtm, tile) for tile in tiles_available]
merge_result = subprocess.run(merge_command, stdout=subprocess.PIPE)
if merge_result.returncode != 0: print(merge_result.stdout)


# 1b. Generate slope raster based on the SRTM DEM

# Extract projection & resolution info from the SRTM raster
srtm_proj, srtm_res_x, srtm_res_y, srtm_x_min, srtm_x_max, srtm_y_min, srtm_y_max, srtm_width, srtm_height = extract_projection_info(srtm_tif)

# Project SRTM raster to NZTM2000 (EPSG:2193) so that both horizontal & vertical units are metres
nztm2000_proj = 'EPSG:2193'
srtm_tif_NZTM2000 = '{}proc/SRTM_Z_NZTM2000.tif'.format(folder_srtm)
project_command = [gdal_warp, '-overwrite', '-s_srs', srtm_proj, '-t_srs', nztm2000_proj, '-dstnodata', '-9999', '-r', 'bilinear', srtm_tif, srtm_tif_NZTM2000]
project_result = subprocess.run(project_command, stdout=subprocess.PIPE)
if project_result.returncode != 0: print(project_result.stdout)

# Generate a slope raster (using the NZTM2000 version of the SRTM raster)
slope_NZTM2000 = '{}proc/SRTM_Slope_NZTM2000.tif'.format(folder_srtm)
slope_command = [gdal_dem, 'slope', srtm_tif_NZTM2000, slope_NZTM2000, '-compute_edges', '-alg', 'ZevenbergenThorne']
slope_result = subprocess.run(slope_command, stdout=subprocess.PIPE)
if slope_result.returncode != 0: print(slope_result.stdout)

# Re-project the slope raster to WGS84, ensuring it aligns with the original SRTM raster
slope_WGS84 = '{}proc/SRTM_Slope.tif'.format(folder_srtm)
reproject_command = [gdal_warp, '-overwrite', '-s_srs', nztm2000_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', srtm_proj, '-r', 'bilinear', '-dstnodata', '-9999', slope_NZTM2000, slope_WGS84]
reproject_result = subprocess.run(reproject_command, stdout=subprocess.PIPE)
if reproject_result.returncode != 0: print(reproject_result.stdout)

# Tidy up leftover variables, to avoid confusion with any variable names reused later
del project_command, project_result, slope_command, slope_result, reproject_command, reproject_result

# Remove any rasters no longer necessary (to preserve space), after confirming that they processed as expected
os.remove(srtm_tif_NZTM2000)
os.remove(slope_NZTM2000)


# 1c. Generate other topographical index rasters based on the SRTM DEM

# Generate an aspect raster (using the original, WGS84 version of the SRTM raster)
aspect_tif = '{}proc/SRTM_Aspect.tif'.format(folder_srtm)
aspect_command = [gdal_dem, 'aspect', srtm_tif, aspect_tif, '-zero_for_flat', '-alg', 'ZevenbergenThorne', '-compute_edges']
aspect_result = subprocess.run(aspect_command, stdout=subprocess.PIPE)
if aspect_result.returncode != 0: print(aspect_result.stdout)

# Generate a Terrain Ruggedness Index raster (using the original, WGS84 version of the SRTM raster)
TRI_tif = '{}proc/SRTM_TRI.tif'.format(folder_srtm)
TRI_command = [gdal_dem, 'TRI', srtm_tif, TRI_tif, '-compute_edges']
TRI_result = subprocess.run(TRI_command, stdout=subprocess.PIPE)
if TRI_result.returncode != 0: print(TRI_result.stdout)

# Generate a Topographic Position Index raster (using the original, WGS84 version of the SRTM raster)
TPI_tif = '{}proc/SRTM_TPI.tif'.format(folder_srtm)
TPI_command = [gdal_dem, 'TPI', srtm_tif, TPI_tif, '-compute_edges']
TPI_result = subprocess.run(TPI_command, stdout=subprocess.PIPE)
if TPI_result.returncode != 0: print(TPI_result.stdout)

# Generate a roughness raster (using the original, WGS84 version of the SRTM raster)
roughness_tif = '{}proc/SRTM_Roughness.tif'.format(folder_srtm)
roughness_command = [gdal_dem, 'roughness', srtm_tif, roughness_tif, '-compute_edges']
roughness_result = subprocess.run(roughness_command, stdout=subprocess.PIPE)
if roughness_result.returncode != 0: print(roughness_result.stdout)

# Tidy up leftover variables, to avoid confusion with any variable names reused later
del aspect_command, aspect_result, TRI_command, TRI_result, TPI_command, TPI_result, roughness_command, roughness_result


###############################################################################
# 2. Process each LiDAR DTM - resample & align with its clipped SRTM raster   #
###############################################################################

# Manually add a field (e.g. in QGIS) to the tile index SHP associated with each LiDAR survey, defining the 'Zone' to which each tile belongs
#  - MRL18_WPE: Marlborough (2018) - Wairau Plains East
#  - MRL18_WVL: Marlborough (2018) - Wairau Valley
#  - MRL18_WKW: Marlborough (2018) - Picton - Waikawa
#  - MRL18_FGA: Marlborough (2018) - Flaxbourne, Grassmere & Lower Awatere
#  - TSM17_STA: Tasman (2017) - St Arnaud
#  - TSM17_LDM: Tasman (2017) - Lee Dam
#  - TSM17_GLB: Tasman (2017) - Golden Bay (including Farewell Spit)
#  - TSM16_ATG: Tasman (2016) - Abel Tasman & Golden Bay

# 2a. Determine which survey zone each LiDAR DTM tile should be associated with

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define dictionary to hold information relating to each zone covered by the Marlborough (2018) survey
dtm_dict = {'MRL18_WPE':{'label':'Wairau Plains East (Marlborough 2018)', 'year':'2018', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'marlborough-lidar-index-tiles-2018.shp'},
            'MRL18_WVL':{'label':'Wairau Valley (Marlborough 2018)', 'year':'2018', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'marlborough-lidar-index-tiles-2018.shp'},
            'MRL18_WKW':{'label':'Picton - Waikawa (Marlborough 2018)', 'year':'2018', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'marlborough-lidar-index-tiles-2018.shp'},
            'MRL18_FGA':{'label':'Flaxbourne, Grassmere & Lower Awatere (Marlborough 2018)', 'year':'2018', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'marlborough-lidar-index-tiles-2018.shp'},
            'TSM17_STA':{'label':'St Arnaud (Tasman 2017)', 'year':'2017', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'tasman-golden-bay-lidar-index-tiles-2017.shp'},
            'TSM17_LDM':{'label':'Lee Dam (Tasman 2017)', 'year':'2017', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'tasman-golden-bay-lidar-index-tiles-2017.shp'},
            'TSM17_GLB':{'label':'Golden Bay & Farewell Spit (Tasman 2017)', 'year':'2017', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'tasman-golden-bay-lidar-index-tiles-2017.shp'},
            'TSM16_ATG':{'label':'Abel Tasman & Golden Bay (Tasman 2016)', 'year':'2016', 'tiles_expected':[], 'tiles_available':[], 'tiles_shp':'tasman-abel-tasman-and-golden-bay-lidar-index-tiles-2016.shp'}}

# Loop through all survey zones available
for zone in zones:
    # Get code for survey which that zone was part of
    survey = zone.split('_')[0]
    # Get full list of image tiles actually available for that survey overall
    tiles_available = [entry.name for entry in os.scandir('{}raw/{}/'.format(folder_dtm, survey)) if entry.name.endswith('.tif')]
    dtm_dict[zone]['tiles_available'] = tiles_available
    # Read index SHP for that survey into a gpd dataframe
    tiles_shp = dtm_dict[zone]['tiles_shp']
    zone_gdf = gpd.read_file('{}index/{}/{}'.format(folder_dtm, survey, tiles_shp))
    # Update DTM dictionary with lists of tile names
    dtm_dict[zone]['tiles_expected'] = zone_gdf[zone_gdf['Zone']==zone]['TileName'].tolist()


# 2b. Read the SRTM grid into memory - to be used as a reference for all LiDAR DTM reprojection & resampling

# Read the merged SRTM raster into memory & retrieve its properties
srtm_proj, srtm_res_x, srtm_res_y, srtm_x_min, srtm_x_max, srtm_y_min, srtm_y_max, srtm_width, srtm_height = extract_projection_info(srtm_tif)

# Define a coordinate transformer from NZTM2000 to WGS84
NZTM2000 = osr.SpatialReference()
NZTM2000.ImportFromEPSG(2193)
NZTM2000_proj4_string = NZTM2000.ExportToProj4()  # '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS84 = osr.SpatialReference()
WGS84.ImportFromEPSG(4326)
WGS84_proj4_string = WGS84.ExportToProj4()        # '+proj=longlat +datum=WGS84 +no_defs'
NZTM2000_to_WGS84 = osr.CoordinateTransformation(NZTM2000, WGS84)


# 2c. Loop through all available LiDAR DTM zones, warping (resample/reproject) them to SRTM-aligned grids
for zone in zones:
    
    print('\nProcessing LiDAR DTM data for {}...'.format(dtm_dict[zone]['label']))
    
    # Get survey code & year
    survey = zone.split('_')[0]
    year = dtm_dict[zone]['year']
    
    # Create zone folder for DTM if it doesn't yet exist
    folder_dtm_zone = '{}proc/{}/'.format(folder_dtm, zone)
    if not os.path.exists(folder_dtm_zone):
        os.mkdir(folder_dtm_zone)
    
    # Create zone folder for text files (if it doesn't yet exist)
    if not os.path.exists('{}TXT/'.format(folder_dtm_zone)):
        os.mkdir('{}TXT/'.format(folder_dtm_zone))
    
    # Create zone folder for shapefiles (if it doesn't yet exist)
    if not os.path.exists('{}SHP/'.format(folder_dtm_zone)):
        os.mkdir('{}SHP/'.format(folder_dtm_zone))
    
    # Create zone folder for DSM if it doesn't yet exist
    folder_srtm_zone = '{}proc/{}/'.format(folder_srtm, zone)
    if not os.path.exists(folder_srtm_zone):
        os.mkdir(folder_srtm_zone)
        
    print(' - Merging available LiDAR DTM tiles...', end=' ')
    # Retrieve the list of expected tiles, from the dictionary developed above
    tiles_expected_for_zone = dtm_dict[zone]['tiles_expected']
    # For the TSM16 survey only, manually filter out tiles which overlap with TSM17
    if survey == 'TSM16': tiles_expected_for_zone = [tiff for tiff in tiles_expected_for_zone if tiff not in ['BN24_1000_3521','BN24_1000_3522','BN24_1000_3523','BN24_1000_3423','BN24_1000_3424']]
    tiles_available = dtm_dict[zone]['tiles_available']
    # Convert this to a list of actual tif files
    tiffs_expected_for_zone = ['DEM_{}_{}_{}.tif'.format(tile[:4], year, tile[5:]) for tile in tiles_expected_for_zone]
    # Filter this to exclude any tiffs not actually available in the folder
    tiffs_available_for_zone = [tiff for tiff in tiffs_expected_for_zone if tiff in tiles_available]
    # Write list of available tiles to a helper text file
    tile_list_path = '{}TXT/DTM_{}_tiles.txt'.format(folder_dtm_zone, zone)
    with open(tile_list_path, 'w') as tile_list:
        tile_list.write('\n'.join('"{}raw/{}/{}"'.format(folder_dtm, survey, tiff) for tiff in tiffs_available_for_zone))
    # Build gdal_merge command
    no_data_in = '-9999'
    no_data_out = '-9999'
    output_format = 'GTiff'
    output_type = 'Float32'
    merge_tif = '{}DTM_{}_Merge.tif'.format(folder_dtm_zone, zone)
    if os.path.exists(merge_tif): os.remove(merge_tif)
    merge_command = ['python', gdal_merge, '-n', no_data_in, '-a_nodata', no_data_out, '-ot', output_type, '-of', output_format, '-o', merge_tif, '--optfile', tile_list_path]
    merge_result = subprocess.run(merge_command, stdout=subprocess.PIPE)
    if merge_result.returncode != 0:
        print(merge_result.stdout)
        break
    print('DONE')
    del merge_command, merge_result
    
    # Read the merged LiDAR raster into memory & retrieve its properties
    print(' - Analysing merged LiDAR DTM raster to align grids...', end=' ')
    dtm_filename = '{}DTM_{}_Merge.tif'.format(folder_dtm_zone, zone)
    dtm_data = geotiff_to_array(dtm_filename)
    dtm_proj, dtm_res_x, dtm_res_y, dtm_x_min, dtm_x_max, dtm_y_min, dtm_y_max, dtm_width, dtm_height = extract_projection_info(dtm_filename)
    
    # Adjust this DTM bounding box to ensure only rows/columns with valid data are included
    dtm_cols_with_data = np.where(dtm_data.max(axis=0)!=-9999)[0]   # Get list of columns which have at least one valid grid cell
    dtm_rows_with_data = np.where(dtm_data.max(axis=1)!=-9999)[0]   # Get list of rows which have at least one valid grid cell
    dtm_x_min_adj = min(dtm_cols_with_data) * dtm_res_x
    dtm_x_max_adj = (dtm_width - max(dtm_cols_with_data)) * dtm_res_x
    dtm_y_max_adj = min(dtm_rows_with_data) * dtm_res_y
    dtm_y_min_adj = (dtm_height - max(dtm_rows_with_data)) * dtm_res_y
    
    # Transform bounding box coordinates into WGS84 coordinates
    dtm_y_min_WGS84, dtm_x_min_WGS84 = NZTM2000_to_WGS84.TransformPoint(dtm_y_min - dtm_y_min_adj, dtm_x_min + dtm_x_min_adj)[:2]
    dtm_y_max_WGS84, dtm_x_max_WGS84 = NZTM2000_to_WGS84.TransformPoint(dtm_y_max + dtm_y_max_adj, dtm_x_max - dtm_x_max_adj)[:2]
    print('DONE')
    
    # Adjust these bounding box coordinates, to 'snap' to interior SRTM cell edges
    print(' - Snapping to SRTM DSM grid resolution & alignment...', end=' ')
    dst_x_min_WGS84 = (ceil((dtm_x_min_WGS84 - srtm_x_min)/srtm_res_x) * srtm_res_x) + srtm_x_min
    dst_x_max_WGS84 = (floor((dtm_x_max_WGS84 - srtm_x_min)/srtm_res_x) * srtm_res_x) + srtm_x_min
    dst_y_min_WGS84 = (ceil((dtm_y_min_WGS84 - srtm_y_min)/srtm_res_y) * srtm_res_y) + srtm_y_min - srtm_res_y
    dst_y_max_WGS84 = (floor((dtm_y_max_WGS84 - srtm_y_min)/srtm_res_y) * srtm_res_y) + srtm_y_min + srtm_res_y
    print('DONE')
    
    # Develop description of DTM spatial reference, including vertical datum
    geoid_NZVD2016 = '{}/proj-datumgrid-nz-20191203/nzgeoid2016.gtx'.format(folder_geoid)
    dtm_srs = osr.SpatialReference()
    dtm_srs.ImportFromWkt(dtm_proj)
    dtm_srs_string = dtm_srs.ExportToProj4() + ' +vunits=m +geoidgrids={}'.format(geoid_NZVD2016)
    
    # Develop description of DSM spatial reference, including assumed vertical datum
    geoid_EGM96 = '{}/EGM96/egm96_15.gtx'.format(folder_geoid)
    srtm_srs = osr.SpatialReference()
    srtm_srs.ImportFromWkt(srtm_proj)
    srtm_srs_string = srtm_srs.ExportToProj4() + ' +vunits=m +geoidgrids={}'.format(geoid_EGM96)
    
    # Warp the LiDAR DTM to WGS84 coordinate reference system, taking the MEDIAN value and using the same raster resolution & alignment as the DSM grid
    print(' - Resampling LiDAR DTM to match SRTM DSM...', end=' ')
    dst_filename_median = '{}DTM_{}_30m_Median.tif'.format(folder_dtm_zone, zone)
    warp_median_command = [gdal_warp, '-overwrite', dtm_filename, dst_filename_median, '-s_srs', dtm_srs_string, '-t_srs', srtm_srs_string, '-to', 'ERROR_ON_MISSING_VERT_SHIFT=YES', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84), '-te_srs', srtm_srs_string, '-r', 'med']
    warp_median_result = subprocess.run(warp_median_command, stdout=subprocess.PIPE)
    if warp_median_result.returncode != 0:
        print(warp_median_result.stdout)
        break
    print('DONE')
    del warp_median_command, warp_median_result
    
    # Generate an additional version in which raster is padded along each edge by the number of cells defined in the 'pad' variable
    print(' - Padding resampled LiDAR DTM with {} pixels along each edge...'.format(pad), end=' ')
    pad_x_min = dst_x_min_WGS84 - pad*srtm_res_x
    pad_x_max = dst_x_max_WGS84 + pad*srtm_res_x
    pad_y_min = dst_y_min_WGS84 - pad*-srtm_res_y
    pad_y_max = dst_y_max_WGS84 + pad*-srtm_res_y
    dst_filename_median_pad = '{}DTM_{}_30m_Median_Pad44.tif'.format(folder_dtm_zone, zone)
    pad_median_command = [gdal_warp, '-overwrite', dst_filename_median, dst_filename_median_pad, '-s_srs', srtm_srs_string, '-t_srs', srtm_srs_string, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', srtm_srs_string, '-r', 'near']
    pad_median_result = subprocess.run(pad_median_command, stdout=subprocess.PIPE)
    if pad_median_result.returncode != 0:
        print(pad_median_result.stdout)
        break
    print('DONE')
    del pad_median_command, pad_median_result
    
    # Generate a second warped (unpadded) LiDAR DTM, applying only a horizontal transformation (to WGS84) but NOT any vertical transformation, just to check
    print(' - Alternative resampling (no vertical transformation), for comparison...', end=' ')
    # Run that command for MEDIAN raster
    dst_filename_median_noZ = '{}DTM_{}_30m_Median_noZ.tif'.format(folder_dtm_zone, zone)
    warp_command_median_noZ = [gdal_warp, '-overwrite', dtm_filename, dst_filename_median_noZ, '-s_srs', 'EPSG:2193', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84), '-te_srs', 'EPSG:4326', '-r', 'med']
    warp_result_median_noZ = subprocess.run(warp_command_median_noZ, stdout=subprocess.PIPE)
    if warp_result_median_noZ.returncode != 0:
        print(warp_result_median_noZ.stdout)
        break
    print('DONE')
    del warp_command_median_noZ, warp_result_median_noZ
    
    # Calculate the vertical difference between the alternative transformations (with & without vertical transformation)
    print(' - Calculating impact of attempted vertical transformation...', end=' ')
    # Run command for MEDIAN raster
    dst_filename_median_Z_diff = '{}DTM_{}_30m_Median_noZ_diff.tif'.format(folder_dtm_zone, zone)
    if os.path.exists(dst_filename_median_Z_diff): os.remove(dst_filename_median_Z_diff)
    median_Z_diff_command = ['python', gdal_calc, '-A', dst_filename_median, '-B', dst_filename_median_noZ, '--outfile={}'.format(dst_filename_median_Z_diff), '--calc=A-B', '--NoDataValue=-9999']
    median_Z_diff_result = subprocess.run(median_Z_diff_command, stdout=subprocess.PIPE)
    if median_Z_diff_result.returncode != 0:
        print(median_Z_diff_result.stdout)
        break
    print('DONE')
    del median_Z_diff_command, median_Z_diff_result
    
    # Generate a SHP showing the boundary of the full-resolution LiDAR DTM
    print(' - Generating extent SHP for resampled LiDAR DTM...', end=' ')
    # Generate a temporary raster defining data coverage
    dtm_extent_raster = '{}SHP/DTM_{}_Extent.tif'.format(folder_dtm_zone, zone)
    if os.path.exists(dtm_extent_raster): os.remove(dtm_extent_raster)
    coverage_command = ['python', gdal_calc, '-A', dst_filename_median, '--outfile={}'.format(dtm_extent_raster), '--calc=A!=-9999', '--NoDataValue=-9999']
    coverage_result = subprocess.run(coverage_command, stdout=subprocess.PIPE)
    if coverage_result.returncode != 0:
        print(coverage_result.stdout)
        break
    # Convert this to a SHP
    dtm_extent_shp = '{}SHP/DTM_{}_Extent.shp'.format(folder_dtm_zone, zone)
    # If this SHP exists already, delete it (and all associated files), to ensure the latest version is stored
    if os.path.exists(dtm_extent_shp):
        [os.remove('{}SHP/{}'.format(folder_dtm_zone, dtm_extent_file)) for dtm_extent_file in os.listdir('{}SHP'.format(folder_dtm_zone)) if not dtm_extent_file.endswith('.tif')]
    # Convert raster to a polygon SHP
    polygonise_command = ['python', gdal_polygonise, dtm_extent_raster, dtm_extent_shp, '-b', '1', '{}_Extent'.format(zone), '-overwrite']
    polygonise_result = subprocess.run(polygonise_command, stdout=subprocess.PIPE)
    if polygonise_result.returncode != 0:
        print(polygonise_result.stdout)
        break
    print('DONE')
    del coverage_command, coverage_result, polygonise_command, polygonise_result
    
    # Clip SRTM & derivatives to the same extent as the new, resampled DTM raster (without any padding)
    print(' - Clipping SRTM DSM & derivatives to same extent as resampled LiDAR DTM...', end=' ')
    for derivative in ['Z', 'Slope', 'Aspect', 'Roughness', 'TPI', 'TRI']:
        srtm_derivative = '{}proc/SRTM_{}.tif'.format(folder_srtm, derivative)
        srtm_clip = '{}proc/{}/SRTM_{}_{}.tif'.format(folder_srtm, zone, zone, derivative)
        clip_command = [gdal_warp, '-overwrite', '-te', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84), srtm_derivative, srtm_clip]
        clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
        if clip_result.returncode != 0:
            print(clip_result.stdout)
            break
    print('DONE')
    del clip_command, clip_result
    
    # Clip SRTM & derivatives to the same extent as the padded version of the resampled DTM raster
    print(' - Clipping SRTM DSM & derivatives to same extent as padded LiDAR DTM...', end=' ')
    for derivative in ['Z', 'Slope', 'Aspect', 'Roughness', 'TPI', 'TRI']:
        srtm_derivative = '{}proc/SRTM_{}.tif'.format(folder_srtm, derivative)
        srtm_clip_pad = '{}proc/{}/SRTM_{}_{}_Pad44.tif'.format(folder_srtm, zone, zone, derivative)
        clip_command_pad = [gdal_warp, '-overwrite', '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), srtm_derivative, srtm_clip_pad]
        clip_result_pad = subprocess.run(clip_command_pad, stdout=subprocess.PIPE)
        if clip_result_pad.returncode != 0:
            print(clip_result_pad.stdout)
            break
    print('DONE')
    del clip_command_pad, clip_result_pad
    
    # Calculate the difference between the SRTM and the LiDAR DTM (i.e. SRTM - LiDAR) - without padding
    print(' - Calculating difference between SRTM & resampled (median) DTM...', end=' ')
    srtm_z_clip = '{}proc/{}/SRTM_{}_Z.tif'.format(folder_srtm, zone, zone)
    srtm_diff_median = '{}proc/{}/SRTM_{}_Median_Diff.tif'.format(folder_srtm, zone, zone)
    if os.path.exists(srtm_diff_median): os.remove(srtm_diff_median)
    diff_median_command = ['python', gdal_calc, '-A', srtm_z_clip, '-B', dst_filename_median, '--outfile={}'.format(srtm_diff_median), '--calc=A-B', '--NoDataValue=-9999']
    diff_median_result = subprocess.run(diff_median_command, stdout=subprocess.PIPE)
    if diff_median_result.returncode != 0:
        print('\nProcess failed, with error message: {}\n'.format(diff_median_result.stdout))
        break
    print('DONE')
    del diff_median_command, diff_median_result
    
    # Calculate the difference between the SRTM and the LiDAR DTM (i.e. SRTM - LiDAR) - WITH padding
    print(' - Calculating difference between padded SRTM & resampled (median) DTM...', end=' ')
    srtm_z_clip_pad = '{}proc/{}/SRTM_{}_Z_Pad44.tif'.format(folder_srtm, zone, zone)
    srtm_diff_median_pad = '{}proc/{}/SRTM_{}_Median_Diff_Pad44.tif'.format(folder_srtm, zone, zone)
    if os.path.exists(srtm_diff_median_pad): os.remove(srtm_diff_median_pad)
    diff_median_command_pad = ['python', gdal_calc, '-A', srtm_z_clip_pad, '-B', dst_filename_median_pad, '--outfile={}'.format(srtm_diff_median_pad), '--calc=A-B', '--NoDataValue=-9999']
    diff_median_result_pad = subprocess.run(diff_median_command_pad, stdout=subprocess.PIPE)
    if diff_median_result_pad.returncode != 0:
        print('\nProcess failed, with error message: {}\n'.format(diff_median_result_pad.stdout))
        break
    print('DONE')
    del diff_median_command_pad, diff_median_result_pad
    
    # Clip the Manaaki Whenua land cover SHP to the extent covered by the current zone
    print(' - Clipping Manaaki Whenua land cover SHP to zone extent...', end=' ')
    LCDB_shp_full = 'C:/Users/mdm123/D/data/LRIS/lris-lcdb-v50/proc/LCDB_v50_WGS84.shp'
    LCDB_shp_clip = 'C:/Users/mdm123/D/data/LRIS/lris-lcdb-v50/proc/LCDB_v50_WGS84_{}.shp'.format(zone)
    ogr2ogr_command = [ogr2ogr, '-clipsrc', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84), LCDB_shp_clip, LCDB_shp_full]
    ogr2ogr_result = subprocess.run(ogr2ogr_command, stdout=subprocess.PIPE)
    if ogr2ogr_result.returncode != 0:
        print('\nProcess failed, with error message: {}\n'.format(ogr2ogr_result.stdout))
        break
    print('DONE')
    
    # Rasterise the Manaaki Whenua land cover SHP to a raster (aligning it with the others, in terms of resolution & extent)
    print(' - Converting clipped land cover SHP to raster...', end=' ')
    LCDB_raster = 'C:/Users/mdm123/D/data/LRIS/lris-lcdb-v50/proc/LCDB_GroupID_{}.tif'.format(zone)
    if os.path.exists(LCDB_raster): os.remove(LCDB_raster)
    rasterise_command = [gdal_rasterise, '-a', 'GrpID_2018', '-l', LCDB_shp_clip.split('/')[-1][:-4], LCDB_shp_clip, LCDB_raster, '-a_nodata', '-9999', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84)]
    rasterise_result = subprocess.run(rasterise_command, stdout=subprocess.PIPE)
    if rasterise_result.returncode != 0:
        print('\nProcess failed, with error message: {}\n'.format(rasterise_result.stdout))
        break
    print('DONE')
    
    # Set up a dictionary to contain SRTM-LiDAR difference values corresponding to each Manaaki Whenua landclass type present in that LiDAR zone coverage
    diff_by_landcover = {1:{'label':'Artificial\nsurfaces', 'data':[], 'colour':(78/255, 78/255, 78/255)},
                         2:{'label':'Bare/lightly-\nvegetated\nsurfaces', 'data':[], 'colour':(255/255, 235/255, 190/255)},
                         3:{'label':'Water\nbodies', 'data':[], 'colour':(0/255, 197/255, 255/255)},
                         4:{'label':'Cropland', 'data':[], 'colour':(255/255, 170/255, 0/255)},
                         5:{'label':'Grassland,\nSedgeland\n& Marshland', 'data':[], 'colour':(255/255, 255/255, 115/255)},
                         6:{'label':'Scrub &\nShrubland', 'data':[], 'colour':(137/255, 205/255, 102/255)},
                         7:{'label':'Forest', 'data':[], 'colour':(38/255, 115/255, 0/255)},
                         8:{'label':'Other', 'data':[], 'colour':'#FF0000'}}
    
    # Read the SRTM-LiDAR diff raster (MEDIAN) & resampled Manaaki Whenua LCDB raster into memory as numpy arrays
    diff_array = geotiff_to_array(srtm_diff_median)
    diff_array = np.ma.masked_equal(diff_array, -9999)
    LCDB_array = geotiff_to_array(LCDB_raster)
    LCDB_array = np.ma.masked_equal(LCDB_array, -9999)
    print(' - SRTM-LiDAR diff & LCDB land cover rasters match' if diff_array.shape==LCDB_array.shape else ' - SRTM-LiDAR diff & LCDB land cover rasters DO NOT MATCH')
    
    # Loop through each potential land cover class (as defined in proc_LCDB.py and above) and populate the dictionary
    print(' - Extracting SRTM-LiDAR diff values for LCDB class:', end=' ')
    for i in range(1,9):
        # Assign SRTM-LiDAR diff values for which LCDB class is i, to the appropriate dictionary entry
        ivalues = diff_array[LCDB_array==i].tolist()
        diff_by_landcover[i]['data'] = [iv for iv in ivalues if iv != None]
        print(i, end=' ')
    
    # Create new folder for figures, if it doesn't exist already
    folder_fig_zone = '{}{}'.format(folder_fig, zone)
    if not os.path.exists(folder_fig_zone):
        os.makedirs(folder_fig_zone)
    
    # Generate horizontal boxplots (including outliers) showing DIFF values for each land class present in this LiDAR zone
    print('\n - Drawing boxplots (including outliers)...', end=' ')
    boxplot_data = [diff_by_landcover[i]['data'] for i in range(1,9)]
    fig, axes = plt.subplots(figsize=(10,6))
    axes.set_title('Boxplots of SRTM-LiDAR vertical differences in {}'.format(dtm_dict[zone]['label']))
    bps = axes.boxplot(boxplot_data, vert=False, patch_artist=True, labels=[diff_by_landcover[i]['label'] for i in range(1,9)])
    for i, patch in enumerate(bps['boxes']):
        patch.set_facecolor(diff_by_landcover[i+1]['colour'])
    axes.set_xlabel('Vertical difference between SRTM DSM and LiDAR DTM [m]')
    # Add annotations noting number of samples available for each land cover class
    n_labels_y = [0.099, 0.204, 0.308, 0.43, 0.52, 0.625, 0.745, 0.852]
    for i in range(1,9):
        n = len(diff_by_landcover[i]['data'])
        axes.annotate('(n={:,})'.format(n), xy=(0.167, n_labels_y[i-1]), xycoords='figure fraction', fontsize='small', color='grey', ha='right')
    plt.grid(which='major', axis='x', alpha=0.5)
    plt.tight_layout()
    fig.savefig('{}{}/SRTM_DTM_DIFF_Boxplots_WithOutliers.png'.format(folder_fig, zone), dpi=300)
    plt.close()
    print('DONE')
    
    # Generate horizontal boxplots (NOT including outliers) showing DIFF values for each land class present in this LiDAR zone
    print(' - Drawing boxplots (not including outliers)...', end=' ')
    fig, axes = plt.subplots(figsize=(10,6))
    axes.set_title('Boxplots of SRTM-LiDAR vertical differences in {}'.format(dtm_dict[zone]['label']))
    bps = axes.boxplot(boxplot_data, vert=False, patch_artist=True, labels=[diff_by_landcover[i]['label'] for i in range(1,9)], showfliers=False)
    for i, patch in enumerate(bps['boxes']):
        patch.set_facecolor(diff_by_landcover[i+1]['colour'])
    axes.set_xlabel('Vertical difference between SRTM DSM and LiDAR DTM [m]')
    # Add annotations noting number of samples available for each land cover class
    n_labels_y = [0.099, 0.204, 0.308, 0.43, 0.52, 0.625, 0.745, 0.852]
    for i in range(1,9):
        n = len(diff_by_landcover[i]['data'])
        axes.annotate('(n={:,})'.format(n), xy=(0.167, n_labels_y[i-1]), xycoords='figure fraction', fontsize='small', color='grey', ha='right')
    plt.grid(which='major', axis='x', alpha=0.5)
    plt.tight_layout()
    fig.savefig('{}{}/SRTM_DTM_DIFF_Boxplots_WithoutOutliers.png'.format(folder_fig, zone), dpi=150)
    plt.close()
    print('DONE')
    
    # Prepare data for violin plots (which don't handle missing data well)
    violin_data = []
    violin_classes = []
    violin_n = [len(data) for data in boxplot_data]
    for i, data in enumerate(boxplot_data):
        # Only append datasets & labels if data are available for that class
        if data != []:
            violin_data.append(data)
            violin_classes.append(i+1)
    
    # Generate violin plots showing DIFF values for each land class present in this LiDAR zone (default colour)
    print(' - Drawing violin plots (default colours)...', end=' ')
    fig, axes = plt.subplots(figsize=(10,6))
    axes.set_title('SRTM-LiDAR Vertical Difference in {}'.format(dtm_dict[zone]['label']))
    axes.violinplot(violin_data, violin_classes, vert=False)
    axes.set_yticks(range(1,8))
    axes.set_yticklabels([diff_by_landcover[i]['label'] for i in range(1,8)])
    #axes.set_yticklabels([diff_by_landcover[i]['label'] for i in violin_classes])
    axes.set_xlabel('Vertical difference between SRTM DSM and LiDAR DTM [m]')
    # Add annotations noting number of samples available for each land cover class
    n_labels_y = [0.115, 0.232, 0.350, 0.481, 0.584, 0.705, 0.837]
    for i in range(1,8):
        axes.annotate('(n={:,})'.format(violin_n[i-1]), xy=(0.167, n_labels_y[i-1]), xycoords='figure fraction', fontsize='small', color='grey', ha='right')
    plt.grid(which='major', axis='x', alpha=0.5)
    plt.tight_layout()
    fig.savefig('{}{}/SRTM_DTM_DIFF_Violin.png'.format(folder_fig, zone), dpi=150)
    plt.close()
    print('DONE')
    
    # Generate violin plots showing DIFF values for each land class present in this LiDAR zone (custom colours)
    print(' - Drawing violin plots (custom colours)...', end=' ')
    fig, axes = plt.subplots(figsize=(10,6))
    axes.set_title('SRTM-LiDAR Vertical Difference in {}'.format(dtm_dict[zone]['label']))
    vls = axes.violinplot(violin_data, violin_classes, vert=False, showmedians=True)
    for i, vl in enumerate(vls['bodies']):
        vl.set_facecolor(diff_by_landcover[i+1]['colour'])
        vl.set_alpha(1)
    for param in ['cbars','cmins','cmaxes','cmedians']:
        vls[param].set_color('dimgrey')
        vls[param].set_linewidth(0.75)
    axes.set_yticks(range(1,8))
    axes.set_yticklabels([diff_by_landcover[i]['label'] for i in range(1,8)])
    axes.set_xlabel('Vertical difference between SRTM DSM and LiDAR DTM [m]')
    # Add annotations noting number of samples available for each land cover class
    n_labels_y = [0.115, 0.232, 0.350, 0.481, 0.584, 0.705, 0.837]
    for i in range(1,8):
        axes.annotate('(n={:,})'.format(violin_n[i-1]), xy=(0.167, n_labels_y[i-1]), xycoords='figure fraction', fontsize='small', color='grey', ha='right')
    plt.grid(which='major', axis='x', alpha=0.5)
    plt.tight_layout()
    fig.savefig('{}{}/SRTM_DTM_DIFF_Violin_Colours.png'.format(folder_fig, zone), dpi=150)
    plt.close()
    print('DONE')
    
    # Generate violin plots with fixed x-axis ranges, showing DIFF values for each land class present in this LiDAR zone (custom colours)
    print(' - Drawing violin plots (custom colours) with fixed x-axis range...', end=' ')
    fig, axes = plt.subplots(figsize=(10,6))
    axes.set_title('SRTM-LiDAR Vertical Difference in {}'.format(dtm_dict[zone]['label']))
    vls = axes.violinplot(violin_data, violin_classes, vert=False, showmedians=True)
    for i, vl in enumerate(vls['bodies']):
        vl.set_facecolor(diff_by_landcover[i+1]['colour'])
        vl.set_alpha(1)
    for param in ['cbars','cmins','cmaxes','cmedians']:
        vls[param].set_color('dimgrey')
        vls[param].set_linewidth(0.75)
    axes.set_yticks(range(1,8))
    axes.set_yticklabels([diff_by_landcover[i]['label'] for i in range(1,8)])
    axes.set_xlabel('Vertical difference between SRTM DSM and LiDAR DTM [m]')
    axes.set_xlim((-15,30))
    # Add annotations noting number of samples available for each land cover class
    n_labels_y = [0.115, 0.232, 0.350, 0.481, 0.584, 0.705, 0.837]
    for i in range(1,8):
        axes.annotate('(n={:,})'.format(violin_n[i-1]), xy=(0.167, n_labels_y[i-1]), xycoords='figure fraction', fontsize='small', color='grey', ha='right')
    plt.grid(which='major', axis='x', alpha=0.5)
    plt.tight_layout()
    fig.savefig('{}{}/SRTM_DTM_DIFF_Violin_Colours_FixedX.png'.format(folder_fig, zone), dpi=150)
    plt.close()
    print('DONE')