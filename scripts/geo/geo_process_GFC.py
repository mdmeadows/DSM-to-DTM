# Process: Global Forest Cover (GFC) dataset developed by Hansen et al (2013): https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.6.html

# Import required packages
import os, sys, urllib.request, subprocess

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, get_geotiff_projection, create_bounded_geotiff

# List paths to GDAL scripts
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'

# Define paths to GFC & SRTM folders
folder_gfc = 'E:/mdm123/D/data/GFC/'
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define dictionary to hold information relating to each zone covered by the Marlborough (2018) survey
dtm_dict = {'MRL18_WPE':{'label':'Wairau Plains East (Marlborough 2018)', 'year':'2018', 'loc_string':'40S_170E'},
            'MRL18_WVL':{'label':'Wairau Valley (Marlborough 2018)', 'year':'2018', 'loc_string':'40S_170E'},
            'MRL18_WKW':{'label':'Picton - Waikawa (Marlborough 2018)', 'year':'2018', 'loc_string':'40S_170E'},
            'MRL18_FGA':{'label':'Flaxbourne, Grassmere & Lower Awatere (Marlborough 2018)', 'year':'2018', 'loc_string':'40S_170E'},
            'TSM17_STA':{'label':'St Arnaud (Tasman 2017)', 'year':'2017', 'loc_string':'40S_170E'},
            'TSM17_LDM':{'label':'Lee Dam (Tasman 2017)', 'year':'2017', 'loc_string':'40S_170E'},
            'TSM17_GLB':{'label':'Golden Bay & Farewell Spit (Tasman 2017)', 'year':'2017', 'loc_string':'40S_170E'},
            'TSM16_ATG':{'label':'Abel Tasman & Golden Bay (Tasman 2016)', 'year':'2016', 'loc_string':'40S_170E'}}

# Define the number of cells of padding to add along each raster boundary
pad = 44


###############################################################################
# 1. Download the GFC datasets from the Google Cloud repository               #
###############################################################################

# Define the datasets available
datasets = ['treecover2000', 'gain', 'lossyear', 'datamask', 'first', 'last']

# Define lng and lat of target tile: based on https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.6.html
lng = '170E'
lat = '40S'

# Loop through all datasets and download corresponding file to the folder defined above
for dataset in datasets:
    filename = 'Hansen_GFC-2018-v1.6_{}_{}_{}.tif'.format(dataset, lat, lng)
    # Check if file is already available there
    if os.path.exists(folder_gfc + 'raw/' + filename):
        print(folder_gfc + filename + ' already exists - skipping')
    else:
        url = 'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2018-v1.6/{}'.format(filename)
        code = urllib.request.urlopen(url).getcode()
        if (code != 404):
            print('Downloading {} tile...'.format(dataset))
            urllib.request.urlretrieve(url, folder_gfc + 'raw/' + filename)
        else:
            print(url + ' not found')


###############################################################################
# 2. Resample to match LiDAR extent & SRTM resolution/alignment               #
###############################################################################

# Define resampling parameters for each of the available datasets
dataset_dict = {'treecover2000':{'resample_method':'cubicspline', 'resample_type':'Float32'},
                'gain':{'resample_method':'near', 'resample_type':'Int16'},
                'lossyear':{'resample_method':'near', 'resample_type':'Int16'},
                'datamask':{'resample_method':'near', 'resample_type':'Int16'},
                'first':{'resample_method':'cubicspline', 'resample_type':'Float32'},
                'last':{'resample_method':'cubicspline', 'resample_type':'Float32'}}

# Process one LiDAR DTM zone at a time
for zone in zones:
    
    print('\n\nProcessing GFC data for {}...'.format(dtm_dict[zone]['label']))
    
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
    
    
    # 2b. Resample GFC data for the six (6) products available
    
    # Create new folder for resampled rasters, if it doesn't exist already
    folder_gfc_zone = '{}proc/{}/'.format(folder_gfc, zone)
    if not os.path.exists(folder_gfc_zone):
        os.makedirs(folder_gfc_zone)
    
    # Loop through the six (6) products available, resampling & reprojecting to match the PADDED SRTM raster
    print(' - Resampling all products:', end=' ')
    for dataset in datasets:
        
        # Find the raster file corresponding to that dataset
        gfc_tile_path = '{}raw/Hansen_GFC-2018-v1.6_{}_{}.tif'.format(folder_gfc, dataset, dtm_dict[zone]['loc_string'])
        
        # Open the GFC raster file for that dataset and extract its coordinate reference system (CRS)
        gfc_proj = get_geotiff_projection(gfc_tile_path)
        
        # Warp the GFC raster to align with PADDED SRTM (using WGS84 coordinate reference system and appropriate interpolation)
        gfc_resample_path = '{}proc/{}/GFC_{}_{}_Pad44.tif'.format(folder_gfc, zone, dataset, zone)
        gfc_resample_method = dataset_dict[dataset]['resample_method']
        gfc_resample_type = dataset_dict[dataset]['resample_type']
        warp_command = [gdal_warp, '-overwrite', gfc_tile_path, gfc_resample_path, '-ot', gfc_resample_type, '-s_srs', gfc_proj, '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', 'EPSG:4326', '-r', gfc_resample_method, '-dstnodata', '-9999']
        warp_result = subprocess.run(warp_command, stdout=subprocess.PIPE)
        if warp_result.returncode != 0:
                print(warp_result.stdout)
                break
        
        # Clip the generated raster to the UNPADDED SRTM extent too
        gfc_clip_path = '{}proc/{}/GFC_{}_{}.tif'.format(folder_gfc, zone, dataset, zone)
        clip_command = [gdal_warp, '-overwrite', gfc_resample_path, gfc_clip_path, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
        clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
        if clip_result.returncode != 0:
            print(clip_result.stdout)
            break
        
        # For "treecover2000", generate BOUNDED versions (0-100), given that values are %s
        if dataset == 'treecover2000':
            gfc_resample_bounded = '{}proc/{}/GFC_{}_{}_{}_Bounded_Pad44.tif'.format(folder_gfc, zone, dataset, zone, gfc_resample_method)
            gfc_clip_bounded = '{}proc/{}/GFC_{}_{}_{}_Bounded.tif'.format(folder_gfc, zone, dataset, zone, gfc_resample_method)
            create_bounded_geotiff(gfc_resample_path, gfc_resample_bounded, 0., 100., -9999)
            create_bounded_geotiff(gfc_clip_path, gfc_clip_bounded, 0., 100., -9999)
        
        print(dataset, end=' ')