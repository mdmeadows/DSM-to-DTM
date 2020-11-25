# Process: OpenStreetMap (OSM) Road, Building & Bridge features

# Import required packages
import os, sys, subprocess
import geopandas as gpd

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info

# List paths to GDAL scripts
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_rasterise = 'C:/Anaconda3/envs/geo/Library/bin/gdal_rasterize.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'
ogr2ogr = 'C:/Anaconda3/envs/geo/Library/bin/ogr2ogr.exe'

# Define path to LiDAR DTM folder & output figure folder
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'
folder_osm = 'E:/mdm123/D/data/OSM/'

# Define paths to downloaded OSM data (done using the "osm_download_OSM.py" script, in the "osm" virtual environment)
osm_roads_shp = 'E:/mdm123/D/data/OSM/raw/Rds/edges/edges.shp'
osm_buildings_shp = 'E:/mdm123/D/data/OSM/raw/Bld/Bld.shp'
osm_riv_lines_shp = 'E:/mdm123/D/data/OSM/raw/Riv/edges/edges.shp'
osm_bridges_shp = 'E:/mdm123/D/data/OSM/raw/Brd/Brd.shp'

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
# 1. Filter 'Roads' SHP to include only large roads                           #
###############################################################################

# Save a filtered version of the road SHP, containing only those with 'highway' type: motorway, trunk, primary, secondary, tertiary
osm_roads_shp_filter = '{}raw/Rds/Rds_Filter.shp'.format(folder_osm)
highway_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
filter_query = ' or '.join(["highway LIKE '%{}%'".format(highway) for highway in highway_types])
filter_command = [ogr2ogr, osm_roads_shp_filter, osm_roads_shp, '-sql', 'SELECT * FROM edges WHERE {}'.format(filter_query), '-t_srs', 'EPSG:4326', '-overwrite']
subprocess.call(filter_command)


###############################################################################
# 2. Generate SHP representing Bridges - potentially important for hydrology  #
###############################################################################

# Generate a point SHP of bridges: where any highway=* features intersect waterway='river bank' features
# Set up a new folder for Bridges
folder_osm_bridges = '{}raw/Brd/'.format(folder_osm)
if not os.path.exists(folder_osm_bridges):
    os.makedirs(folder_osm_bridges)
# Copy the two required shapefiles there
subprocess.call([ogr2ogr, '{}/Rds_Hwy.shp'.format(folder_osm_bridges), osm_roads_shp_filter])
subprocess.call([ogr2ogr, '{}/Wtr_Riv.shp'.format(folder_osm_bridges), osm_riv_lines_shp])
# Intersect these two line features
rds_hwy = gpd.read_file('{}/Rds_Hwy.shp'.format(folder_osm_bridges))
wtr_riv = gpd.read_file('{}/Wtr_Riv.shp'.format(folder_osm_bridges))
brd_riv = rds_hwy.unary_union.intersection(wtr_riv.unary_union)
# Convert to a geodataframe, adding CRS info and an index
brd_riv_gdf = gpd.GeoDataFrame(brd_riv, columns=['geometry'])
brd_riv_gdf.crs = {'init':'epsg:4326'}
brd_riv_gdf['id']= brd_riv_gdf.index
brd_riv_gdf.to_file(osm_bridges_shp)


###############################################################################
# 3. Loop through all survey zones, preparing rasters for each                #
###############################################################################

# Process one LiDAR DTM zone at a time
for zone in zones:
    
    print('\nProcessing OSM data for {}...'.format(dtm_dict[zone]['label']))
    
    # Create new folder for rasters, if it doesn't exist already
    folder_osm_zone = '{}proc/{}'.format(folder_osm, zone)
    if not os.path.exists(folder_osm_zone):
        os.makedirs(folder_osm_zone)
    
    
    # 3a. Read desired properties from SRTM raster covering that zone
    
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
    
    
    # 3b. Process building footprint data
    
    print(' - Processing Building footprints...', end=' ')
    
    # Rasterise OSM Building Footprint file to 1m resolution initially - PADDED extent
    osm_buildings_tif_temp = '{}/OSM_Bld_{}_1m.tif'.format(folder_osm_zone, zone)
    rasterise_command = [gdal_rasterise, '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-tr', str(srtm_res_x/30.), str(-srtm_res_y/30.), '-burn', '1', '-ot', 'Float32', '-a_nodata', '-9999', '-init', '0', osm_buildings_shp, osm_buildings_tif_temp]
    subprocess.call(rasterise_command)
    
    # Warp 1m OSM Building raster to 30m raster matching the PADDED SRTM, taking the average value (i.e. fraction of raster cell covered by buildings)
    osm_buildings_tif_full = '{}/OSM_Bld_{}_Pad44.tif'.format(folder_osm_zone, zone)
    warp_command = [gdal_warp, '-overwrite', osm_buildings_tif_temp, osm_buildings_tif_full, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', 'EPSG:4326', '-r', 'Average', '-dstnodata', '-9999']
    subprocess.call(warp_command)
    
    # Clip the generated raster to the UNPADDED SRTM extent too
    osm_buildings_tif_clip = '{}/OSM_Bld_{}.tif'.format(folder_osm_zone, zone)
    clip_command = [gdal_warp, '-overwrite', osm_buildings_tif_full, osm_buildings_tif_clip, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
    clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
    if clip_result.returncode != 0:
        print(clip_result.stdout)
        break
    
    # Remove the intermediate raster
    os.remove(osm_buildings_tif_temp)
    print('DONE')
    
    
    # 3c. Process road data
    
    print(' - Processing Roads...', end=' ')
    
    # Rasterise OSM Roads (filtered) file to 10m resolution initially (with 10m used to ensure road widths are reasonably well represented)
    osm_roads_tif_temp = '{}/OSM_Rds_{}_10m.tif'.format(folder_osm_zone, zone)
    rasterise_command = [gdal_rasterise, '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-tr', str(srtm_res_x/3.), str(-srtm_res_y/3.), '-burn', '1', '-ot', 'Float32', '-a_nodata', '-9999', '-init', '0', osm_roads_shp_filter, osm_roads_tif_temp]
    subprocess.call(rasterise_command)
    
    # Warp 10m OSM Roads (filtered) raster to 30m raster matching SRTM, taking the average value (i.e. fraction of raster cell covered by buildings)
    osm_roads_tif_full = '{}/OSM_Rds_{}_Pad44.tif'.format(folder_osm_zone, zone)
    warp_command = [gdal_warp, '-overwrite', osm_roads_tif_temp, osm_roads_tif_full, '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-te_srs', 'EPSG:4326', '-r', 'Average', '-dstnodata', '-9999']
    subprocess.call(warp_command)
    
    # Clip the generated raster to the UNPADDED SRTM extent too
    osm_roads_tif_clip = '{}/OSM_Rds_{}.tif'.format(folder_osm_zone, zone)
    clip_command = [gdal_warp, '-overwrite', osm_roads_tif_full, osm_roads_tif_clip, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
    clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
    if clip_result.returncode != 0:
        print(clip_result.stdout)
        break
    
    # Remove the intermediate raster
    os.remove(osm_roads_tif_temp)
    print('DONE')
    
    
    # 3d. Process bridge data
    
    print(' - Processing Bridges...', end=' ')
    
    # Rasterise OSM Bridge SHP (points) to 30m resolution (to match PADDED SRTM zone)
    osm_bridges_tif_full = '{}/OSM_Brd_{}_Pad44.tif'.format(folder_osm_zone, zone)
    rasterise_command = [gdal_rasterise, '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max), '-tr', str(srtm_res_x), str(-srtm_res_y), '-burn', '1', '-ot', 'Int16', '-a_nodata', '-9999', '-init', '0', osm_bridges_shp, osm_bridges_tif_full]
    subprocess.call(rasterise_command)
    
    # Clip the generated raster to the UNPADDED SRTM extent too
    osm_bridges_tif_clip = '{}/OSM_Brd_{}.tif'.format(folder_osm_zone, zone)
    clip_command = [gdal_warp, '-overwrite', osm_bridges_tif_full, osm_bridges_tif_clip, '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(srtm_x_min), str(srtm_y_min), str(srtm_x_max), str(srtm_y_max), '-te_srs', 'EPSG:4326', '-r', 'near', '-dstnodata', '-9999']
    clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
    if clip_result.returncode != 0:
        print(clip_result.stdout)
        break
    print('DONE')