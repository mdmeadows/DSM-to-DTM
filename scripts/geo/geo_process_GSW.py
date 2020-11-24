# Process: Global Surface Water (GSW) dataset developed by Pekel et al. (2016): https://global-surface-water.appspot.com/download

# Import required packages
import os, sys, urllib.request, subprocess

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, get_geotiff_projection, create_bounded_geotiff

# List paths to GDAL scripts
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'

# Define paths to GSW & LiDAR folders
folder_gsw = 'E:/mdm123/D/data/GSW/'
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define dictionary to hold information relating to each zone covered by the Marlborough (2018) survey
dtm_dict = {'MRL18_WPE':{'label':'Wairau Plains East (Marlborough 2018)', 'year':'2018', 'loc_string':'170E_40S'},
            'MRL18_WVL':{'label':'Wairau Valley (Marlborough 2018)', 'year':'2018', 'loc_string':'170E_40S'},
            'MRL18_WKW':{'label':'Picton - Waikawa (Marlborough 2018)', 'year':'2018', 'loc_string':'170E_40S'},
            'MRL18_FGA':{'label':'Flaxbourne, Grassmere & Lower Awatere (Marlborough 2018)', 'year':'2018', 'loc_string':'170E_40S'},
            'TSM17_STA':{'label':'St Arnaud (Tasman 2017)', 'year':'2017', 'loc_string':'170E_40S'},
            'TSM17_LDM':{'label':'Lee Dam (Tasman 2017)', 'year':'2017', 'loc_string':'170E_40S'},
            'TSM17_GLB':{'label':'Golden Bay & Farewell Spit (Tasman 2017)', 'year':'2017', 'loc_string':'170E_40S'},
            'TSM16_ATG':{'label':'Abel Tasman & Golden Bay (Tasman 2016)', 'year':'2016', 'loc_string':'170E_40S'}}

# Define the number of cells of padding to add along each raster boundary
pad = 44


###############################################################################
# 1. Download the GSW datasets from the website listed below                  #
###############################################################################

# Define the datasets available
datasets = ['occurrence', 'change', 'seasonality', 'recurrence', 'transitions', 'extent']

# Define lng and lat of target tile: based on https://global-surface-water.appspot.com/download
lng = '170E'
lat = '40S'

# Loop through all datasets and download corresponding file to the folder defined above
for dataset in datasets:
    filename = '{}_{}_{}_v1_1.tif'.format(dataset, lng, lat)
    # Check if file is already available there
    if os.path.exists(folder_gsw + 'raw/' + filename):
        print(folder_gsw + filename + ' already exists - skipping')
    else:
        url = 'http://storage.googleapis.com/global-surface-water/downloads2/{}/{}'.format(dataset, filename)
        code = urllib.request.urlopen(url).getcode()
        if (code != 404):
            print('Downloading {} tile...'.format(dataset))
            urllib.request.urlretrieve(url, folder_gsw + 'raw/' + filename)
        else:
            print(url + ' not found')


###############################################################################
# 2. Resample to match LiDAR extent & SRTM resolution/alignment               #
###############################################################################

# Define appropriate processing parameters for each available product
dataset_dict = {'occurrence':{'resample_method':'cubicspline', 'resample_type':'Float32'},
                'change':{'resample_method':'cubicspline', 'resample_type':'Float32'},
                'seasonality':{'resample_method':'cubicspline', 'resample_type':'Float32'},
                'recurrence':{'resample_method':'cubicspline', 'resample_type':'Float32'},
                'transitions':{'resample_method':'near', 'resample_type':'Int16'},
                'extent':{'resample_method':'near', 'resample_type':'Int16'}}

# Process one survey zone at a time
for zone in zones:
    
    print('\n\nProcessing GSW data for {}...'.format(dtm_dict[zone]['label']))
    
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
    
    
    # 2b. Resample GSW data available for the six (6) products available
    
    # Create new folder for resampled rasters, if it doesn't exist already
    folder_gsw_zone = '{}proc/{}/'.format(folder_gsw, zone)
    if not os.path.exists(folder_gsw_zone):
        os.makedirs(folder_gsw_zone)
    
    # Loop through the six (6) products available, resampling & reprojecting to match the PADDED SRTM raster
    print(' - Resampling all products:', end=' ')
    for dataset in datasets:
        
        # Find the raster file corresponding to that dataset
        gsw_tile_path = '{}raw/{}_{}_v1_1.tif'.format(folder_gsw, dataset, dtm_dict[zone]['loc_string'])
        
        # Open the GSW raster file for that dataset and extract its coordinate reference system (CRS)
        gsw_proj = get_geotiff_projection(gsw_tile_path)
        
        # Warp the GSW raster to WGS84 coordinate reference system, matching the PADDED SRTM grid (with the appropriate resolution & alignment)
        gsw_resample_path = '{}proc/{}/GSW_{}_{}_Pad44.tif'.format(folder_gsw, zone, dataset, zone)
        gsw_resample_method = dataset_dict[dataset]['resample_method']
        gsw_resample_type = dataset_dict[dataset]['resample_type']
        warp_command = [gdal_warp, '-overwrite', gsw_tile_path, gsw_resample_path, '-ot', gsw_resample_type, '-s_srs', gsw_proj, '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', 'EPSG:4326', '-r', gsw_resample_method, '-dstnodata', '-9999']
        warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
        if warp_result.returncode != 0:
            print(warp_result.stdout)
            break
        
        # Clip the generated raster to the UNPADDED SRTM extent too
        gsw_clip_path = '{}proc/{}/GSW_{}_{}_{}.tif'.format(folder_gsw, zone, dataset, zone, gsw_resample_method)
        clip_command = [gdal_warp, '-overwrite', gsw_resample_path, gsw_clip_path, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
        clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
        if clip_result.returncode != 0:
            print(clip_result.stdout)
            break
        
        # For "occurrence", generated BOUNDED versions (0-100), since "occurrence" is percentage
        if dataset == 'occurrence':
            gsw_resample_bounded_path = '{}proc/{}/GSW_{}_{}_{}_Bounded_Pad44.tif'.format(folder_gsw, zone, dataset, zone, gsw_resample_method)
            gsw_clip_bounded_path = '{}proc/{}/GSW_{}_{}_{}_Bounded.tif'.format(folder_gsw, zone, dataset, zone, gsw_resample_method)
            create_bounded_geotiff(gsw_resample_path, gsw_resample_bounded_path, 0., 100., -9999)
            create_bounded_geotiff(gsw_clip_path, gsw_clip_bounded_path, 0., 100., -9999)

        print(dataset, end=' ')