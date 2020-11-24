# Process: Manaaki Whenua Land Cover Database (LCDB v5.0)

# Import required packages
import sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info

# List paths to GDAL scripts
ogr2ogr = 'C:/Anaconda3/envs/geo/Library/bin/ogr2ogr.exe'
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_rasterise = 'C:/Anaconda3/envs/geo/Library/bin/gdal_rasterize.exe'

# Define paths to SRTM & LCDB folders
folder_srtm = 'E:/mdm123/D/data/DSM/SRTM/'
folder_lcdb = 'E:/mdm123/D/data/LRIS/lris-lcdb-v50'
folder_inputs = 'E:/mdm123/D/ML/inputs/1D'
folder_fig = 'E:/mdm123/D/figures'

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
# 1. Reproject LCDB SHP (from NZTM2000 to WGS84)                              #
###############################################################################

# Reproject the Manaaki Whenua land cover SHP from NZTM2000 to WGS84 (EPSG:4326)
LCDB_shp_NZTM2000 = '{}/raw/lcdb-v50-land-cover-database-version-50-mainland-new-zealand.shp'.format(folder_lcdb)
LCDB_shp_WGS84 = '{}/proc/LCDB_v50_WGS84.shp'.format(folder_lcdb)
WGS84 = 'EPSG:4326'
reproject_command = [ogr2ogr, LCDB_shp_WGS84, LCDB_shp_NZTM2000, '-t_srs', WGS84, '-overwrite']
reproject_result = subprocess.run(reproject_command, stdout=subprocess.PIPE)
if reproject_result.returncode != 0: print(reproject_result.stdout)


###############################################################################
# 2. ArcMap codeblocks for reclassification of sub-classes to main groupings  #
###############################################################################

# Used for Python codeblock of field calculation in ArcMap (based on LCDB v4+ definitions of CLASS & NAME)
def classify_group(c):
    # Artificial surfaces
    if c in [1, 2, 5, 6]:
        return 'Artificial Surfaces'
    # Bare or Lightly-vegetated Surfaces
    elif c in [10, 12, 14, 15, 16]:
        return 'Bare or Lightly-vegetated Surfaces'
    # Water Bodies
    elif c in [20, 21, 22]:
        return 'Water Bodies'
    # Cropland
    elif c in [30, 33]:
        return 'Cropland'
    # Grassland, Sedgeland and Marshland
    elif c in [40, 41, 43, 44, 45, 46, 47]:
        return 'Grassland, Sedgeland and Marshland'
    # Scrub and Shrubland
    elif c in [50, 51, 52, 54, 55, 56, 58, 80, 81]:
        return 'Scrub and Shrubland'
    # Forest
    elif c in [64, 68, 69, 70, 71]:
        return 'Forest'
    # Other
    else:
        return 'Other'

# Used for Python codeblock of field calculation in ArcMap (based on LCDB v4+ definitions of CLASS & NAME)
def classify_ID(c):
    # Artificial surfaces = 1
    if c in [1, 2, 5, 6]:
        return 1
    # Bare or Lightly-vegetated Surfaces = 2
    elif c in [10, 12, 14, 15, 16]:
        return 2
    # Water Bodies = 3
    elif c in [20, 21, 22]:
        return 3
    # Cropland = 4
    elif c in [30, 33]:
        return 4
    # Grassland, Sedgeland and Marshland = 5
    elif c in [40, 41, 43, 44, 45, 46, 47]:
        return 5
    # Scrub and Shrubland = 6
    elif c in [50, 51, 52, 54, 55, 56, 58, 80, 81]:
        return 6
    # Forest = 7
    elif c in [64, 68, 69, 70, 71]:
        return 7
    # Other = 8
    else:
        return 8


###############################################################################
# 3. Resample LCDB rasters to match padded SRTM grids for each zone           #
###############################################################################

# Loop through all available DTM survey zones
for zone in zones:
    
    print('\nProcessing {} zone:'.format(zone))
    
    # Open a template raster (DEM for that zone, with pad=44) & extract its properties
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
    
    # Rasterise the Manaaki Whenua land cover SHP to a raster (aligning it with the others, in terms of resolution & extent)
    print(' - Rasterising land cover SHP to zone GeoTIFF...')
    LCDB_shp = '{}/proc/LCDB_v50_WGS84.shp'.format(folder_lcdb)
    LCDB_tif = '{}/proc/LCDB_GroupID_{}_Pad44.tif'.format(folder_lcdb, zone)
    rasterise_command = [gdal_rasterise, '-a', 'GrpID_2018', '-l', LCDB_shp.split('/')[-1][:-4], LCDB_shp, LCDB_tif, '-a_nodata', '-9999', '-tr', str(srtm_res_x), str(-srtm_res_y), '-te', str(pad_x_min), str(pad_y_min), str(pad_x_max), str(pad_y_max)]
    rasterise_result = subprocess.run(rasterise_command, stdout=subprocess.PIPE)
    if rasterise_result.returncode != 0:
        print('\nProcess failed, with error message: {}\n'.format(rasterise_result.stdout))
        break


###############################################################################
# 4. More processing of LCDB rasters within the geo_process_LiDAR_SRTM script #
###############################################################################

# Further processing & visualisation of the LCDB raster data was done in the "geo_process_LiDAR_SRTM.py" script


###############################################################################
# 5. Generate thumbnail histograms for each zone, to include in LCDB map      #
###############################################################################

# Set up a dictionary of properties for each Manaaki Whenua landclass type present
lcdb_dict = {1:{'label':'Artificial\nsurfaces', 'colour':(78/255, 78/255, 78/255)},
             2:{'label':'Bare or Lightly-\nvegetated Surfaces', 'colour':(255/255, 235/255, 190/255)},
             3:{'label':'Water\nBodies', 'colour':(0/255, 197/255, 255/255)},
             4:{'label':'Cropland', 'colour':(255/255, 170/255, 0/255)},
             5:{'label':'Grassland, Sedgeland\nand Marshland', 'colour':(255/255, 255/255, 115/255)},
             6:{'label':'Scrub and\nShrubland', 'colour':(137/255, 205/255, 102/255)},
             7:{'label':'Forest', 'colour':(38/255, 115/255, 0/255)},
             8:{'label':'Other', 'colour':'red'}}

# Set up list of bin edges & colours
lcdb_bins = np.linspace(0.5, 7.5, num=8)
lcdb_colours = [lcdb_dict[l]['colour'] for l in range(1,8)]

# Loop through each zone, generating a very simple histogram (colours only) of the land cover classes present
for zone in zones:
    
    # Read 1D vector of processed input data for that zone
    df = pd.read_csv('{}/Input1D_ByZone_{}.csv'.format(folder_inputs, zone))
    
    # Get a list of LCDB class codes for all valid pixels
    lcdb_list = df['lcdb'].loc[df['diff'] != -9999].tolist()
    lcdb_list = [l for l in lcdb_list if (not np.isnan(l) and l != None and l != -9999)]
    
    # Generate histogram manually, to ensure all classes covered (even if not present in that zone)
    fig, axes = plt.subplots(figsize=(2,0.8))
    _,_,patches = axes.hist(lcdb_list, bins=lcdb_bins, edgecolor='dimgrey', linewidth=0.3)
    for patch, colour in zip(patches, lcdb_colours):
        patch.set_facecolor(colour)
    # Tidy up figure & save
    [axes.spines[edge].set_visible(False) for edge in ['left','top','right']]
    axes.spines['bottom'].set_color('black')
    axes.spines['bottom'].set_linewidth(0.5)
    axes.yaxis.set_visible(False)
    axes.set_xticklabels([])
    axes.set_xticks([])
    fig.tight_layout()
    fig.savefig('{}/All/Distributions/LCDB/landcover_hist_{}.png'.format(folder_fig, zone), dpi=150, transparent=True, bbox='tight')
    plt.close()