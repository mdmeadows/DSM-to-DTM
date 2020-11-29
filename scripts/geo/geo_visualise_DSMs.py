# Visualise: Compare available DSMs with the LiDAR-derived DTMs, to select most suitable base DSM

# Import required packages
from osgeo import gdal, gdalconst, osr
import numpy as np
import os, subprocess, sys
from math import ceil, floor
import matplotlib.pyplot as plt
gdal.UseExceptions()                                                           # Useful for trouble-shooting

# Import helper functions relevant to this script
sys.path.append('E:/mdm123/D/scripts/geo/')
from geo_helpers import extract_projection_info, geotiff_to_array

# List paths to GDAL scripts
gdal_warp = 'C:/Anaconda3/envs/geo/Library/bin/gdalwarp.exe'
gdal_calc = 'C:/Anaconda3/envs/geo/Scripts/gdal_calc.py'

# Define paths to relevant folders
folder_geoid = 'E:/mdm123/D/data/Geoid'
folder_dsm_general = 'E:/mdm123/D/data/DSM'
folder_dtm = 'E:/mdm123/D/data/DTM/proc'
folder_fig = 'E:/mdm123/D/figures'

# Define lists of available DSMs and survey zones
DSMs = ['SRTM','ASTER','AW3D30']
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define a dictionary of each LiDAR survey zone's properties
zone_dict = {'MRL18_WPE':{'label':'Wairau Plains East (Marlborough)',
                          'colour':'green'},
             'MRL18_WVL':{'label':'Wairau Valley (Marlborough)',
                          'colour':'red'},
             'MRL18_WKW':{'label':'Picton - Waikawa (Marlborough)',
                          'colour':'purple'},
             'MRL18_FGA':{'label':'Flaxbourne, Grassmere & Lower Awatere (Marlborough)',
                          'colour':'blue'},
             'TSM17_STA':{'label':'St Arnaud (Tasman)',
                          'colour':'orange'},
             'TSM17_LDM':{'label':'Lee Dam (Tasman)',
                          'colour':'dimgrey'},
             'TSM17_GLB':{'label':'Golden Bay (Tasman)',
                          'colour':'saddlebrown'},
             'TSM16_ATG':{'label':'Tasman (Tasman)',
                          'colour':'pink'}}

# Define a coordinate transformer from NZTM2000 to WGS84
NZTM2000 = osr.SpatialReference()
NZTM2000.ImportFromEPSG(2193)
NZTM2000_proj4_string = NZTM2000.ExportToProj4()  # '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS84 = osr.SpatialReference()
WGS84.ImportFromEPSG(4326)
WGS84_proj4_string = WGS84.ExportToProj4()        # '+proj=longlat +datum=WGS84 +no_defs'
NZTM2000_to_WGS84 = osr.CoordinateTransformation(NZTM2000, WGS84)


###############################################################################
# 1. Resample each LiDAR-derived DTM to match each of the available DSMs      #
###############################################################################

# Loop through each available DSM
for DSM in DSMs:
    
    print('\nProcessing comparison of {} DSM with available LiDAR DTM:'.format(DSM))
    
    # Define path to folder containing the original merged TIF for that DSM
    folder_dsm = '{}/{}'.format(folder_dsm_general, DSM)
    
    # Create a folder for the DSM vs DTM assessment, if it doesn't yet exist
    if not os.path.exists('{}/assess'.format(folder_dsm)):
        os.mkdir('{}/assess'.format(folder_dsm))
    
    # Read the merged DSM raster into memory & retrieve its properties
    dsm_tif = '{}/proc/{}_Z.tif'.format(folder_dsm, DSM)
    dsm_proj, dsm_res_x, dsm_res_y, dsm_x_min, dsm_x_max, dsm_y_min, dsm_y_max, dsm_width, dsm_height = extract_projection_info(dsm_tif)
    
    # Now loop through each LiDAR survey zone
    for zone in zones:
        
        print(' - Zone: {}'.format(zone_dict[zone]['label']))
        folder_dtm_zone = '{}/{}'.format(folder_dtm, zone)
        
        # Read the merged LiDAR raster into memory & retrieve its properties
        print('   - Analysing merged LiDAR DTM raster to align grids...', end=' ')
        dtm_filename = '{}/DTM_{}_Merge.tif'.format(folder_dtm_zone, zone)
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
        print('   - Snapping to SRTM DSM grid resolution & alignment...', end=' ')
        dst_x_min_WGS84 = (ceil((dtm_x_min_WGS84 - dsm_x_min)/dsm_res_x) * dsm_res_x) + dsm_x_min
        dst_x_max_WGS84 = (floor((dtm_x_max_WGS84 - dsm_x_min)/dsm_res_x) * dsm_res_x) + dsm_x_min
        dst_y_min_WGS84 = (ceil((dtm_y_min_WGS84 - dsm_y_min)/dsm_res_y) * dsm_res_y) + dsm_y_min - dsm_res_y
        dst_y_max_WGS84 = (floor((dtm_y_max_WGS84 - dsm_y_min)/dsm_res_y) * dsm_res_y) + dsm_y_min + dsm_res_y
        print('DONE')
        
        # Develop description of DTM spatial reference, including vertical datum
        geoid_NZVD2016 = '{}/proj-datumgrid-nz-20191203/nzgeoid2016.gtx'.format(folder_geoid)
        dtm_srs = osr.SpatialReference()
        dtm_srs.ImportFromWkt(dtm_proj)
        dtm_srs_string = dtm_srs.ExportToProj4() + ' +vunits=m +geoidgrids={}'.format(geoid_NZVD2016)
        
        # Develop description of DSM spatial reference, including assumed vertical datum
        geoid_EGM96 = '{}/EGM96/egm96_15.gtx'.format(folder_geoid)
        dsm_srs = osr.SpatialReference()
        dsm_srs.ImportFromWkt(dsm_proj)
        dsm_srs_string = dsm_srs.ExportToProj4() + ' +vunits=m +geoidgrids={}'.format(geoid_EGM96)
        
        # Warp the LiDAR DTM to WGS84 coordinate reference system, taking AVERAGE & MEDIAN value and using the same raster resolution & alignment as the DSM grid
        print('   - Resampling LiDAR DTM to match {} DSM...'.format(DSM), end=' ')
        # Run command for MEAN raster
        dtm_mean = '{}/assess/DTM_Mean_{}_{}.tif'.format(folder_dsm, DSM, zone)
        warp_mean_command = [gdal_warp, '-overwrite', dtm_filename, dtm_mean, '-s_srs', dtm_srs_string, '-t_srs', dsm_srs_string, '-to', 'ERROR_ON_MISSING_VERT_SHIFT=YES', '-tr', str(dsm_res_x), str(-dsm_res_y), '-te', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84), '-te_srs', dsm_srs_string, '-r', 'average']
        warp_mean_result = subprocess.run(warp_mean_command, stdout=subprocess.PIPE)
        if warp_mean_result.returncode != 0:
            print(warp_mean_result.stdout)
            break
        # Run command for MEDIAN raster
        dtm_median = '{}/assess/DTM_Median_{}_{}.tif'.format(folder_dsm, DSM, zone)
        warp_median_command = [gdal_warp, '-overwrite', dtm_filename, dtm_median, '-s_srs', dtm_srs_string, '-t_srs', dsm_srs_string, '-to', 'ERROR_ON_MISSING_VERT_SHIFT=YES', '-tr', str(dsm_res_x), str(-dsm_res_y), '-te', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84), '-te_srs', dsm_srs_string, '-r', 'med']
        warp_median_result = subprocess.run(warp_median_command, stdout=subprocess.PIPE)
        if warp_median_result.returncode != 0:
            print(warp_median_result.stdout)
            break
        print('DONE')
        del warp_mean_command, warp_mean_result, warp_median_command, warp_median_result
        
        # Clip the DSM to the same extent as the new, resampled (DSM-aligned) DTM raster
        print('   - Clipping {} DSM to same extent as resampled LiDAR DTM...'.format(DSM), end=' ')
        dsm_clip = '{}/assess/{}_Clip_{}.tif'.format(folder_dsm, DSM, zone)
        clip_command = [gdal_warp, '-overwrite', '-te', str(dst_x_min_WGS84), str(dst_y_min_WGS84), str(dst_x_max_WGS84), str(dst_y_max_WGS84), dsm_tif, dsm_clip]
        clip_result = subprocess.run(clip_command, stdout=subprocess.PIPE)
        if clip_result.returncode != 0:
            print(clip_result.stdout)
            break
        print('DONE')
        del clip_command, clip_result
        
        # Calculate the difference between the DSM and the LiDAR DTM (i.e. DSM - LiDAR)
        print('   - Calculating difference between {} DSM & resampled DTM (both mean & median)...'.format(DSM), end=' ')
        # For resampled DTM raster based on MEAN values
        diff_mean = '{}/assess/Diff_Mean_{}_{}.tif'.format(folder_dsm, DSM, zone)
        diff_mean_command = ['python', gdal_calc, '-A', dsm_clip, '-B', dtm_mean, '--outfile={}'.format(diff_mean), '--calc=A-B', '--NoDataValue=-9999']
        diff_mean_result = subprocess.run(diff_mean_command, stdout=subprocess.PIPE)
        if diff_mean_result.returncode != 0:
            print('\nProcess failed, with error message: {}\n'.format(diff_mean_result.stdout))
            break
        # For resampled DTM raster based on MEDIAN values
        diff_median = '{}/assess/Diff_Median_{}_{}.tif'.format(folder_dsm, DSM, zone)
        diff_median_command = ['python', gdal_calc, '-A', dsm_clip, '-B', dtm_median, '--outfile={}'.format(diff_median), '--calc=A-B', '--NoDataValue=-9999']
        diff_median_result = subprocess.run(diff_median_command, stdout=subprocess.PIPE)
        if diff_median_result.returncode != 0:
            print('\nProcess failed, with error message: {}\n'.format(diff_median_result.stdout))
            break
        print('DONE')
        del diff_mean_command, diff_mean_result, diff_median_command, diff_median_result


###############################################################################
# 2. Generate plots comparing error distributions for each DSM and zone       #
###############################################################################

# Consider resampled DTM rasters based on both mean & median aggregation methods
for aggregation in ['Mean','Median']:
    
    # Set up a 4x1 figure - one axis for each of the survey zones
    fig, axes = plt.subplots(nrows=len(zones), ncols=1, figsize=(8,3*len(zones)))
    
    # Loop through the four LiDAR survey zones available
    for i, zone in enumerate(zones):
        
        # Establish empty list to hold the DIFF vectors & mean DIFF values for each DSM
        diffs = []
        
        # Loop through the three DSMs available
        for DSM in DSMs:
            
            # Update the DSM folder reference
            folder_dsm = '{}/{}'.format(folder_dsm_general, DSM)
            
            # Read in the appropriate DIFF raster and convert to a vector (with no_data values filtered out)
            diff_tif = '{}/assess/Diff_{}_{}_{}.tif'.format(folder_dsm, aggregation, DSM, zone)
            diff_ds = gdal.Open(diff_tif, gdalconst.GA_ReadOnly)
            diff_vector = np.array(diff_ds.ReadAsArray()).ravel()
            diff_vals = [val for val in diff_vector if val != -9999]
            diff_ds = None
            
            # Add that DSM's vector of DIFF values to the list
            diffs.append(diff_vals)
            
        # Add that survey zone's DIFF data to the figure
        diff_means = [np.mean(diff) for diff in diffs]
        diff_stds = [np.std(diff) for diff in diffs]
        diff_labels = [r'{} ($\mu$={:.2f}, $\sigma$={:.2f})'.format(DSM, diff_mean, diff_std) for (DSM, diff_mean, diff_std) in zip(DSMs, diff_means, diff_stds)]
        axes[i].hist(diffs, histtype='step', bins=100, label=diff_labels, color=['blue','green','purple'], lw=2, alpha=0.5)
        axes[i].axvline(x=0, color='black')
        axes[i].set_yscale('log')
        axes[i].set_ylabel('Cell count')
        axes[i].set_title('Comparison of DSM biases in {}'.format(zone_dict[zone]['label']))
        axes[i].legend(loc='best', frameon=False)
    
    # Specify general figure properties
    axes[i].set_xlabel('Difference between each DSM & the {} LiDAR DTM'.format(aggregation.lower()))
    fig.tight_layout()
    fig.savefig('{}/All/zones_compare_DSM_accuracy_{}.png'.format(folder_fig, aggregation.lower()), dpi=300)
    plt.close()


###############################################################################
# 3. Generate plot comparing overall error distribution for each DEM          #
###############################################################################

# Define colours to be used for each DSM
DSM_colours = {'SRTM':'blue', 'ASTER':'purple', 'AW3D30':'green'}

# Consider resampled DTM rasters based on median aggregation
aggregation = 'Median'

# Establish empty list to hold the DIFF vectors & mean DIFF values for each DSM
diffs = {DSM:[] for DSM in DSMs}

# Loop through the three DEMs available, adding each to the plot
for DSM in DSMs:
    
    # Update the DSM folder reference
    folder_dsm = '{}/{}'.format(folder_dsm_general, DSM)
    
    # Loop through the four LiDAR survey zones available, adding the DIFF values to each to the list
    for zone in zones:
        
        # Read in the appropriate DIFF raster and convert to a vector (with no_data values filtered out)
        diff_tif = '{}/assess/Diff_{}_{}_{}.tif'.format(folder_dsm, aggregation, DSM, zone)
        diff_ds = gdal.Open(diff_tif, gdalconst.GA_ReadOnly)
        diff_vector = np.array(diff_ds.ReadAsArray()).ravel()
        diff_vals = [val for val in diff_vector if val != -9999]
        diff_ds = None
        
        # Add that DSM's vector of DIFF values to the list
        diffs[DSM].append(diff_vals)


# Develop a range of common bins for the comparison histogram
diff_min = np.nanmin([np.nanmin([val for dsm_diffs in diffs[DSM] for val in dsm_diffs]) for DSM in DSMs])
diff_max = np.nanmax([np.nanmax([val for dsm_diffs in diffs[DSM] for val in dsm_diffs]) for DSM in DSMs])
diff_range = np.linspace(diff_min, diff_max, num=100)

# Set up a single-axis figure
fig, axes = plt.subplots(figsize=(9,4.5))

# Loop through the three DEMs available, adding each to the plot
for DSM in DSMs:
    
    # Add that DSM's DIFF data to the figure
    diffs_flatten = [val for dsm_diffs in diffs[DSM] for val in dsm_diffs]
    diff_mean = np.mean(diffs_flatten)
    diff_std = np.std(diffs_flatten)
    diff_label = r'{} ($\mu$={:.2f}, $\sigma$={:.2f})'.format(DSM, diff_mean, diff_std)
    axes.hist(diffs_flatten, histtype='step', bins=diff_range, label=diff_label, color=DSM_colours[DSM], lw=2, alpha=0.5)

# Specify general figure properties
axes.axvline(x=0, color='black')
axes.set_yscale('log')
axes.set_ylabel('Grid cell count (log-scale)')
axes.set_title('Histograms of errors for each DEM (compared to resampled LiDAR DTM)')
axes.legend(loc='best', frameon=False)
axes.set_xlabel('Errors (differences between each DEM & the resampled LiDAR DTM) [m]')
fig.tight_layout()
fig.savefig('{}/All/compare_DSM_accuracy.png'.format(folder_fig), dpi=300)
plt.close()