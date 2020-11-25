# Process: Night-Time Light (NTL) datasets available
#  - 2000: DMSP - https://ngdc.noaa.gov/eog/dmsp/downloadV4composites.html
#  - 2018: VIIRS - https://ngdc.noaa.gov/eog/viirs/download_dnb_composites.html

# Import required packages
import os, sys, subprocess

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, get_geotiff_projection, create_bounded_geotiff

# List paths to GDAL scripts
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'

# Define paths to NTL & LiDAR folders
folder_ntl = 'E:/mdm123/D/data/NTL/'
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
# 1. Manually download the NTL datasets from the NOAA website (links below)   #
###############################################################################

# Download manually from NOAA website
#  - DMSP data: https://ngdc.noaa.gov/eog/dmsp/downloadV4composites.html
#  - VIIRS data: https://ngdc.noaa.gov/eog/viirs/download_dnb_composites.html


###############################################################################
# 2. Resample each NTL raster to match SRTM resolution/alignment              #
###############################################################################

# Define list of the layers available for DMSP
dmsp_layers = ['avg_vis', 'avg_vis_stable', 'pct_lights', 'avg_lights_pct']

# Define dictionary specifying each of the DMSP NTL layers available
dmsp_dict = {'avg_vis':{'folder':'F152000.v4', 'suffix':'_web.avg_vis'},
             'avg_vis_stable':{'folder':'F152000.v4', 'suffix':'_web.stable_lights.avg_vis'},
             'pct_lights':{'folder':'F152000.v4b.avg_lights_x_pct', 'suffix':'.pct_lights'},
             'avg_lights_pct':{'folder':'F152000.v4b.avg_lights_x_pct', 'suffix':'.avg_lights_x_pct'}}

# Define list of resampling methods to try
resampling_options = ['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos']


# Loop through all survey zones, warping each NTL raster to align with the corresponding SRTM grid
for zone in zones:
    
    print('\nProcessing NTL data for {}...'.format(dtm_dict[zone]['label']))
    
    # 2a. Read desired properties from SRTM raster covering that zone
    
    # Read the appropriate SRTM DSM raster into memory & retrieve its properties
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
    
    print(' - Processing NTL datasets available...', end=' ')
    
    
    # 2b. Resample VIIRS NTL data, testing a variety of different upsampling approaches
    
    # Create new folder for resampled rasters, if it doesn't exist already
    folder_viirs_zone = '{}/VIIRS/proc/{}/'.format(folder_ntl, zone)
    if not os.path.exists(folder_viirs_zone):
        os.makedirs(folder_viirs_zone)
    
    # Define path to selected VIIRS raster
    viirs_path = '{}VIIRS/raw/SVDNB_npp_20180801-20180831_00N060E_vcmcfg_v10_c201809070900.avg_rade9h.tif'.format(folder_ntl)
    
    # Open the VIIRS raster file and extract its coordinate reference system (CRS)
    viirs_proj = get_geotiff_projection(viirs_path)
    
    # Loop through all resampling options considered, generating an upsampled raster for that VIIRS layer with each
    for resampling in resampling_options:
        
        # Upsample 500m VIIRS NTL raster to 30m grid matching PADDED SRTM raster (using the selected resampling method)
        viirs_upsample = '{}VIIRS/proc/{}/NTL_VIIRS_{}_{}_Pad44.tif'.format(folder_ntl, zone, zone, resampling)
        warp_command = [gdal_warp, '-overwrite', viirs_path, viirs_upsample, '-ot', 'Float32', '-s_srs', viirs_proj, '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', 'EPSG:4326', '-r', resampling, '-dstnodata', '-9999']
        warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
        if warp_result.returncode != 0:
            print(warp_result.stdout)
            break
        
        # Clip the generated raster to the UNPADDED SRTM extent too
        viirs_clip = '{}VIIRS/proc/{}/NTL_VIIRS_{}_{}.tif'.format(folder_ntl, zone, zone, resampling)
        clip_command = [gdal_warp, '-overwrite', viirs_upsample, viirs_clip, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
        clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
        if clip_result.returncode != 0:
            print(clip_result.stdout)
            break
    print('VIIRS', end=' ')
    
    
    # 2c. Resample DMSP NTL data, testing a variety of different upsampling approaches
    
    # Create new folder for resampled rasters, if it doesn't exist already
    folder_dmsp_zone = '{}/DMSP/proc/{}/'.format(folder_ntl, zone)
    if not os.path.exists(folder_dmsp_zone):
        os.makedirs(folder_dmsp_zone)
    
    # Loop through all DMSP NTL rasters available
    for layer in dmsp_layers:
        
        # Define path to selected DMSP raster
        dmsp_subfolder = dmsp_dict[layer]['folder']
        dmsp_suffix = dmsp_dict[layer]['suffix']
        dmsp_path = '{}DMSP/raw/{}/F152000.v4b{}.tif'.format(folder_ntl, dmsp_subfolder, dmsp_suffix)
        
        # Open the DMSP raster file and extract its coordinate reference system (CRS)
        dmsp_proj = get_geotiff_projection(dmsp_path)
        
        # Loop through all resampling options considered, generating an upsampled raster for that DMSP layer with each
        for resampling in resampling_options:
            
            # Upsample DMSP NTL raster to 30m grid matching PADDED SRTM raster (using the selected resampling method)
            dmsp_upsample = '{}DMSP/proc/{}/NTL_DMSP_{}_{}_{}_Pad44.tif'.format(folder_ntl, zone, layer, zone, resampling)
            warp_command = [gdal_warp, '-overwrite', dmsp_path, dmsp_upsample, '-ot', 'Float32', '-s_srs', dmsp_proj, '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', 'EPSG:4326', '-r', resampling, '-dstnodata', '-9999']
            warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
            if warp_result.returncode != 0:
                print(warp_result.stdout)
                break
            
            # Clip the generated raster to the UNPADDED SRTM extent too
            dmsp_clip = '{}DMSP/proc/{}/NTL_DMSP_{}_{}_{}.tif'.format(folder_ntl, zone, layer, zone, resampling)
            clip_command = [gdal_warp, '-overwrite', dmsp_upsample, dmsp_clip, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
            clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
            if clip_result.returncode != 0:
                print(clip_result.stdout)
                break
            
            # For "pct_lights", generated BOUNDED versions (0-100), since "pct_lights" is percentage
            if layer == 'pct_lights':
                dmsp_upsample_bounded = '{}DMSP/proc/{}/NTL_DMSP_{}_{}_{}_Bounded_Pad44.tif'.format(folder_ntl, zone, layer, zone, resampling)
                dmsp_clip_bounded = '{}DMSP/proc/{}/NTL_DMSP_{}_{}_{}_Bounded.tif'.format(folder_ntl, zone, layer, zone, resampling)
                create_bounded_geotiff(dmsp_upsample, dmsp_upsample_bounded, 0., 100., -9999)
                create_bounded_geotiff(dmsp_clip, dmsp_clip_bounded, 0., 100., -9999)
    print('DMSP')