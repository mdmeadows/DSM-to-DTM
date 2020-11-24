# Process: ASTER GDEM (DSM) topography data

# Import required packages
import os, sys, subprocess

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, get_geotiff_projection

# List paths to GDAL scripts
gdal_merge = 'C:/Anaconda3/envs/geo/Scripts/gdal_merge.py'
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_dem = 'C:/Anaconda3/envs/geo/Library/bin/gdaldem.exe'

# Define path to ASTER, SRTM & output figure folder
folder_aster = 'E:/mdm123/D/data/DSM/ASTER/'
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'
folder_fig = 'E:/mdm123/D/figures/'

# Define the number of cells of padding to add along each raster boundary
pad = 44


###############################################################################
# 1. Process ASTER DSM data - merge tiles & generate derivative rasters       #
###############################################################################

# 1a. Merge ASTER tiles covering Area of Interest

# Get full list of image tiles actually available
tiles_available = [entry.name for entry in os.scandir('{}raw/'.format(folder_aster)) if entry.name.endswith('_dem.tif')]

# Build gdal_merge command
no_data_out = '-9999'
output_format = 'GTiff'
output_type = 'Int16'
aster_tif = '{}proc/ASTER_Z.tif'.format(folder_aster)
if os.path.exists(aster_tif): os.remove(aster_tif)
merge_command = ['python', gdal_merge, '-a_nodata', no_data_out, '-ot', output_type, '-of', output_format, '-o', aster_tif] + ['{}raw/{}'.format(folder_aster, tile) for tile in tiles_available]
merge_result = subprocess.run(merge_command, stdout=subprocess.PIPE)
if merge_result.returncode != 0: print(merge_result.stdout)


# 1b. Generate slope raster based on the ASTER DEM

# Extract projection & resolution info from the ASTER raster
aster_proj, aster_res_x, aster_res_y, aster_x_min, aster_x_max, aster_y_min, aster_y_max, aster_width, aster_height = extract_projection_info(aster_tif)

# Project ASTER raster to NZTM2000 (EPSG:2193) so that both horizontal & vertical units are metres (for slope calculation)
nztm2000_proj = 'EPSG:2193'
aster_tif_NZTM2000 = '{}proc/ASTER_Z_NZTM2000.tif'.format(folder_aster)
project_command = [gdal_warp, '-overwrite', '-ot', 'Float32', '-s_srs', aster_proj, '-t_srs', nztm2000_proj, '-dstnodata', '-9999', '-r', 'bilinear', aster_tif, aster_tif_NZTM2000]
project_result = subprocess.run(project_command, stdout=subprocess.PIPE)
if project_result.returncode != 0: print(project_result.stdout)

# Generate a slope raster (using the NZTM2000 version of the ASTER raster)
slope_NZTM2000 = '{}proc/ASTER_Slope_NZTM2000.tif'.format(folder_aster)
if os.path.exists(slope_NZTM2000): os.remove(slope_NZTM2000)
slope_command = [gdal_dem, 'slope', aster_tif_NZTM2000, slope_NZTM2000, '-compute_edges', '-alg', 'ZevenbergenThorne']
slope_result = subprocess.run(slope_command, stdout=subprocess.PIPE)
if slope_result.returncode != 0: print(slope_result.stdout)

# Tidy up leftover variables, to avoid confusion with any variable names reused later
del project_command, project_result, slope_command, slope_result


# 1c. Generate other topographical index rasters based on the ASTER DEM

# Generate an aspect raster (using the original, WGS84 version of the ASTER raster)
aspect_tif = '{}proc/ASTER_Aspect.tif'.format(folder_aster)
if os.path.exists(aspect_tif): os.remove(aspect_tif)
aspect_command = [gdal_dem, 'aspect', aster_tif, aspect_tif, '-zero_for_flat', '-alg', 'ZevenbergenThorne', '-compute_edges']
aspect_result = subprocess.run(aspect_command, stdout=subprocess.PIPE)
if aspect_result.returncode != 0: print(aspect_result.stdout)

# Generate a Terrain Ruggedness Index raster (using the original, WGS84 version of the ASTER raster)
TRI_tif = '{}proc/ASTER_TRI.tif'.format(folder_aster)
if os.path.exists(TRI_tif): os.remove(TRI_tif)
TRI_command = [gdal_dem, 'TRI', aster_tif, TRI_tif, '-compute_edges']
TRI_result = subprocess.run(TRI_command, stdout=subprocess.PIPE)
if TRI_result.returncode != 0: print(TRI_result.stdout)

# Generate a Topographic Position Index raster (using the original, WGS84 version of the ASTER raster)
TPI_tif = '{}proc/ASTER_TPI.tif'.format(folder_aster)
if os.path.exists(TPI_tif): os.remove(TPI_tif)
TPI_command = [gdal_dem, 'TPI', aster_tif, TPI_tif, '-compute_edges']
TPI_result = subprocess.run(TPI_command, stdout=subprocess.PIPE)
if TPI_result.returncode != 0: print(TPI_result.stdout)

# Generate a roughness raster (using the original, WGS84 version of the ASTER raster)
roughness_tif = '{}proc/ASTER_Roughness.tif'.format(folder_aster)
if os.path.exists(roughness_tif): os.remove(roughness_tif)
roughness_command = [gdal_dem, 'roughness', aster_tif, roughness_tif, '-compute_edges']
roughness_result = subprocess.run(roughness_command, stdout=subprocess.PIPE)
if roughness_result.returncode != 0: print(roughness_result.stdout)

# Tidy up leftover variables, to avoid confusion with any variable names reused later
del aspect_command, aspect_result, TRI_command, TRI_result, TPI_command, TPI_result, roughness_command, roughness_result


###############################################################################
# 2. Resample ASTER rasters for each available DTM zone, matching SRTM grids  #
###############################################################################

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define dictionary to hold information relating to each zone covered by the Marlborough (2018) survey
dtm_dict = {'MRL18_WPE':{'label':'Wairau Plains East (Marlborough 2018)', 'year':'2018'},
            'MRL18_WVL':{'label':'Wairau Valley (Marlborough 2018)', 'year':'2018'},
            'MRL18_WKW':{'label':'Picton - Waikawa (Marlborough 2018)', 'year':'2018'},
            'MRL18_FGA':{'label':'Flaxbourne, Grassmere & Lower Awatere (Marlborough 2018)', 'year':'2018'},
            'TSM17_STA':{'label':'St Arnaud (Tasman 2017)', 'year':'2017'},
            'TSM17_LDM':{'label':'Lee Dam (Tasman 2017)', 'year':'2017'},
            'TSM17_GLB':{'label':'Golden Bay & Farewell Spit (Tasman 2017)', 'year':'2017'},
            'TSM16_ATG':{'label':'Abel Tasman & Golden Bay (Tasman 2016)', 'year':'2016'}}

# Loop through all survey zones, warping each ASTER derivative to align with the corresponding SRTM grid
for zone in zones:
    
    print('\nProcessing ASTER DSM data for {}...'.format(dtm_dict[zone]['label']))
    
    # Create zone folder, if it doesn't already exist
    if not os.path.exists('{}proc/{}/'.format(folder_aster, zone)):
        os.mkdir('{}proc/{}/'.format(folder_aster, zone))
    
    # Read the appropriate SRTM DSM raster into memory & retrieve its properties
    print(' - Analysing zonal SRTM raster to align grids...')
    srtm_filename = '{}proc/{}/SRTM_{}_Z.tif'.format(folder_srtm, zone, zone)
    srtm_proj, srtm_res_x, srtm_res_y, srtm_x_min, srtm_x_max, srtm_y_min, srtm_y_max, srtm_width, srtm_height = extract_projection_info(srtm_filename)
    
    # Resample & clip each ASTER product so that it matches up with that zone's SRTM grid - for both padded & unpadded rasters
    print(' - Warping ASTER derivatives:', end=' ')
    for derivative in ['Z', 'Slope', 'Aspect', 'TRI', 'TPI', 'Roughness']:
        
        # Open the full raster corresponding to that derivative & extract its coordinate reference system (CRS)
        src_filename = '{}proc/ASTER_Slope_NZTM2000.tif'.format(folder_aster) if derivative=='Slope' else '{}proc/ASTER_{}.tif'.format(folder_aster, derivative)
        src_proj = get_geotiff_projection(src_filename)
        
        # Warp the ASTER derivative, padding extra data along each edge, by the number of pixels defined in the "pad" variable
        pad_x_min = srtm_x_min - pad*srtm_res_x
        pad_x_max = srtm_x_max + pad*srtm_res_x
        pad_y_min = srtm_y_min - pad*-srtm_res_y
        pad_y_max = srtm_y_max + pad*-srtm_res_y
        dst_filename_pad = '{}proc/{}/ASTER_{}_{}_Pad44.tif'.format(folder_aster, zone, zone, derivative)
        warp_command_pad = [gdal_warp, '-overwrite', '-ot', 'Float32', src_filename, dst_filename_pad, '-s_srs', src_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', srtm_proj, '-r', 'bilinear', '-dstnodata', '-9999']
        warp_result_pad = subprocess.run(warp_command_pad, stdout=subprocess.PIPE)
        if warp_result_pad.returncode != 0:
            print(warp_result_pad.stdout)
            break
        
        # Now clip the warped ASTER derivative to get the "unpadded" version
        dst_filename = '{}proc/{}/ASTER_{}_{}.tif'.format(folder_aster, zone, zone, derivative)
        warp_command = [gdal_warp, '-overwrite', dst_filename_pad, dst_filename, '-s_srs', srtm_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', srtm_proj, '-r', 'near', '-dstnodata', '-9999']
        warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
        if warp_result.returncode != 0:
            print(warp_result.stdout)
            break
        
        print(derivative, end=' ')
        
    print('DONE')

# Remove any rasters no longer necessary (to preserve space), after confirming that they processed as expected
os.remove(aster_tif_NZTM2000)
os.remove(slope_NZTM2000)