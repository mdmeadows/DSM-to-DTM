# Process: MERIT DEM developed by Yamazaki et al. (2017)

# Import required packages
import os, sys, subprocess

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, get_geotiff_projection

# List paths to GDAL scripts
gdal_merge = 'C:/Anaconda3/envs/geo/Scripts/gdal_merge.py'
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_dem = 'C:/Anaconda3/envs/geo/Library/bin/gdaldem.exe'

# Define path to MERIT, SRTM & output figure folder
folder_merit = 'E:/mdm123/D/data/DSM/MERIT/'
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'
folder_fig = 'E:/mdm123/D/figures/'

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
# 1. Resample MERIT DEM raster to match the SRTM grid for each DTM zone       #
###############################################################################

# Loop through all survey zones, warping MERIT data to align with the corresponding SRTM grid
for zone in zones:
    
    print('\nProcessing MERIT DSM data for {}...'.format(dtm_dict[zone]['label']))
    
    # Create zone folder, if it doesn't already exist
    if not os.path.exists('{}proc/{}/'.format(folder_merit, zone)):
        os.mkdir('{}proc/{}/'.format(folder_merit, zone))
    
    # Read the appropriate SRTM DSM raster into memory & retrieve its properties
    print(' - Analysing zonal SRTM raster to align grids...')
    srtm_filename = '{}proc/{}/SRTM_{}_Z.tif'.format(folder_srtm, zone, zone)
    srtm_proj, srtm_res_x, srtm_res_y, srtm_x_min, srtm_x_max, srtm_y_min, srtm_y_max, srtm_width, srtm_height = extract_projection_info(srtm_filename)
    
    # Resample & clip MERIT DEM so that it matches up with that zone's SRTM grid
    print(' - Warping MERIT DEM')
    
    # Open the full MERIT DEM raster & extract its coordinate reference system (CRS)
    src_filename = '{}raw/dem_tif_s60e150/s45e170_dem.tif'.format(folder_merit)
    src_proj = get_geotiff_projection(src_filename)
    
    # Warp the MERIT DEM, padding extra data along each edge, by the number of pixels defined in the "pad" variable
    pad_x_min = srtm_x_min - pad*srtm_res_x
    pad_x_max = srtm_x_max + pad*srtm_res_x
    pad_y_min = srtm_y_min - pad*-srtm_res_y
    pad_y_max = srtm_y_max + pad*-srtm_res_y
    dst_filename_pad = '{}proc/{}/MERIT_{}_Pad44.tif'.format(folder_merit, zone, zone)
    warp_command_pad = [gdal_warp, '-overwrite', '-ot', 'Float32', src_filename, dst_filename_pad, '-s_srs', src_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', srtm_proj, '-r', 'bilinear', '-dstnodata', '-9999']
    warp_result_pad = subprocess.run(warp_command_pad, stdout=subprocess.PIPE)
    if warp_result_pad.returncode != 0:
        print(warp_result_pad.stdout)
        break
    
    # Now clip the warped MERIT DEM to get the "unpadded" version
    dst_filename = '{}proc/{}/MERIT_{}.tif'.format(folder_merit, zone, zone)
    warp_command = [gdal_warp, '-overwrite', dst_filename_pad, dst_filename, '-s_srs', srtm_proj, '-t_srs', srtm_proj, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', srtm_proj, '-r', 'near', '-dstnodata', '-9999']
    warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
    if warp_result.returncode != 0:
        print(warp_result.stdout)
        break