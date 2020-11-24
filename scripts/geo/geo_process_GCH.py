# Process: Global Canopy Heights (GCH) dataset developed by Simard et al (2011): https://landscape.jpl.nasa.gov/

# Import required packages
import os, sys, subprocess

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, get_geotiff_projection, create_bounded_geotiff

# List paths to GDAL scripts
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'

# Define paths to GCH & SRTM folders
folder_gch = 'E:/mdm123/D/data/GCH/'
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'

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

# Define the number of cells of padding to add along each raster boundary
pad = 44


###############################################################################
# 1. Manually download the GCH datasets from the JPL website (links below)    #
###############################################################################

# Global Canopy Height datasets - Simard et al 2011 (https://landscape.jpl.nasa.gov/)
#  - Canopy height: https://landscape.jpl.nasa.gov/resources/Simard_Pinto_3DGlobalVeg_JGR.tif.gz
#  - Legend: https://landscape.jpl.nasa.gov/images/legend_3DGlobalVeg.png
#  - Error map: https://landscape.jpl.nasa.gov/resources/Global_l3c_error_map.tif.gz

# All rasters downloaded to: E:/mdm123/D/data/GCH/raw


###############################################################################
# 2. Resample each GCH raster to match SRTM resolution/alignment              #
###############################################################################

# Loop through all survey zones, warping each GCH raster to align with the corresponding SRTM grid
for zone in zones:
    
    print('\n\nProcessing GCH data for {}...'.format(dtm_dict[zone]['label']))
    
    # Read the appropriate SRTM DSM raster into memory & retrieve its properties
    print(' - Analysing zonal SRTM raster to align grids...', end=' ')
    srtm_filename = '{}proc/{}/SRTM_{}_Z.tif'.format(folder_srtm, zone, zone)
    srtm_proj, srtm_res_x, srtm_res_y, srtm_x_min, srtm_x_max, srtm_y_min, srtm_y_max, srtm_width, srtm_height = extract_projection_info(srtm_filename)
    
    # Define a new bounding box, including the padding required for the 2D convnet data pre-processing
    pad_x_min = srtm_x_min - pad*srtm_res_x
    pad_x_max = srtm_x_max + pad*srtm_res_x
    pad_y_min = srtm_y_min - pad*-srtm_res_y
    pad_y_max = srtm_y_max + pad*-srtm_res_y
    pad_width = srtm_width + 2*pad
    pad_height = srtm_height + 2*pad
    
    
    # Resample GCH canopy height data (using cubic splines for resampling)
    
    # Create new folder for resampled rasters, if it doesn't exist already
    folder_gch_zone = '{}proc/{}/'.format(folder_gch, zone)
    if not os.path.exists(folder_gch_zone):
        os.makedirs(folder_gch_zone)
    
    # Find the raster file corresponding to that dataset
    gch_path = '{}raw/Simard_Pinto_3DGlobalVeg_L3C.tif'.format(folder_gch)
    
    # Open the GFC raster file for that dataset and extract its coordinate reference system (CRS)
    gch_proj = get_geotiff_projection(gch_path)
    
    # Upsample 1km canopy height to 30m grid (to match PADDED SRTM raster) using the selected resampling method
    gch_upsample = '{}proc/{}/GCH_{}_cubicspline_Pad44.tif'.format(folder_gch, zone, zone)
    warp_command = [gdal_warp, '-overwrite', gch_path, gch_upsample, '-ot', 'Float32', '-s_srs', gch_proj, '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', 'EPSG:4326', '-r', 'cubicspline', '-dstnodata', '-9999']
    warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
    if warp_result.returncode != 0:
        print(warp_result.stdout)
        break
    
    # Clip the generated raster to the UNPADDED SRTM extent too
    gch_clip = '{}proc/{}/GCH_{}_cubicspline.tif'.format(folder_gch, zone, zone)
    clip_command = [gdal_warp, '-overwrite', gch_upsample, gch_clip, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
    clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
    if clip_result.returncode != 0:
        print(clip_result.stdout)
        break
    
    # For both outputs, also generated BOUNDED versions (low: 0), since negative tree heights don't make any sense
    gch_upsample_bounded = '{}proc/{}/GCH_{}_cubicspline_Bounded_Pad44.tif'.format(folder_gch, zone, zone)
    gch_clip_bounded = '{}proc/{}/GCH_{}_cubicspline_Bounded.tif'.format(folder_gch, zone, zone)
    create_bounded_geotiff(gch_upsample, gch_upsample_bounded, 0., -9999, -9999)
    create_bounded_geotiff(gch_clip, gch_clip_bounded, 0., -9999, -9999)