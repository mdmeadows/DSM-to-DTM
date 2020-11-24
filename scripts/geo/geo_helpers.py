# Helper functions for 'geo' scripts

# Import required modules
from osgeo import gdal, gdalconst
import numpy as np
import datetime

# Define a function to read the projection, bounding box & resolution of an input GeoTIFF raster
def extract_projection_info(tif_path):
    """Function to read a GeoTIFF raster (given its path)
    and return its projection, resolution, bounding box,
    width & height, as a list of separate variables."""
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    # Extract projection & geotransform from input dataset
    ds_proj = ds.GetProjection()
    ds_geotrans = ds.GetGeoTransform()
    ds_res_x = ds_geotrans[1]
    ds_res_y = ds_geotrans[5]  # Note: identical to x-resolution, but negative
    ds_width = ds.RasterXSize
    ds_height = ds.RasterYSize
    # Get bounding box of input dataset
    ds_x_min = ds_geotrans[0]
    ds_y_max = ds_geotrans[3]
    ds_x_max = ds_x_min + (ds_geotrans[1] * ds_width)
    ds_y_min = ds_y_max + (ds_geotrans[5] * ds_height)
    # Close access to GeoTIFF file
    ds = None
    # Return all results
    return ds_proj, ds_res_x, ds_res_y, ds_x_min, ds_x_max, ds_y_min, ds_y_max, ds_width, ds_height

# Define a function to read a GeoTIFF's projection, given its filepath
def get_geotiff_projection(tif_path):
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    ds_proj = ds.GetProjection()
    ds = None
    return ds_proj

# Define a function to read a GeoTIFF's no data value, given its filepath
def get_geotiff_nodatavalue(tif_path):
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    ds_band = ds.GetRasterBand(1)
    ds_nodata = ds_band.GetNoDataValue()
    ds = None
    return ds_nodata

# Define a function to read projection, bounding box & resolution from a GeoTIFF raster, given its filepath
def get_geotiff_props(tif_path):
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    # Extract projection & geotransform from input dataset
    ds_proj = ds.GetProjection()
    ds_geotrans = ds.GetGeoTransform()
    ds_res_x = ds_geotrans[1]
    ds_res_y = ds_geotrans[5]  # Note: identical to x-resolution, but negative
    ds_width = ds.RasterXSize
    ds_height = ds.RasterYSize
    # Close access to GeoTIFF file
    ds = None
    # Get bounding box of input dataset
    ds_x_min = ds_geotrans[0]
    ds_y_max = ds_geotrans[3]
    ds_x_max = ds_x_min + (ds_geotrans[1] * ds_width)
    ds_y_min = ds_y_max + (ds_geotrans[5] * ds_height)
    # Return all results as a dictionary
    return {'proj':ds_proj, 'res_x':ds_res_x, 'res_y':ds_res_y, 'x_min':ds_x_min, 'x_max':ds_x_max, 'y_min':ds_y_min, 'y_max':ds_y_max, 'width':ds_width, 'height':ds_height}

# Define a function that returns a GeoTIFF's data as a numpy array, given its filepath
def geotiff_to_array(tif_path):
    tif_ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    tif_array = np.array(tif_ds.ReadAsArray())
    tif_ds = None
    return tif_array

# A function to save a numpy array to a GeoTIFF, given a dictionary of geotransform properties
def array_to_geotiff(array, tif_path, no_data_value, props, output_format=gdal.GDT_Float32):
    # Get the appropriate GDAL driver
    driver = gdal.GetDriverByName('GTiff')
    # Create a new GeoTIFF file to which the array is to be written
    tif_width = props['width']
    tif_height = props['height']
    ds = driver.Create(tif_path, tif_width, tif_height, 1, output_format)
    # Set the geotransform
    tif_x_min = props['x_min']
    tif_res_x = props['res_x']
    tif_y_max = props['y_max']
    tif_res_y = props['res_y']
    ds.SetGeoTransform((tif_x_min, tif_res_x, 0, tif_y_max, 0, tif_res_y))
    # Set the projection
    tif_proj = props['proj']
    ds.SetProjection(tif_proj)
    # Set the no data value
    ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    # Write array to GeoTIFF
    ds.GetRasterBand(1).WriteArray(array)
    ds.FlushCache()

# A function that applies bounds (lower & upper) to an input GeoTIFF, writing to the specifed output GeoTIFF
def create_bounded_geotiff(tif_path_in, tif_path_out, lower_bound, upper_bound, no_data_value):
    # Extract all relevant properties from the input GeoTIFF
    ds_props = get_geotiff_props(tif_path_in)
    ds_nodatavalue = get_geotiff_nodatavalue(tif_path_in)
    ds_array = geotiff_to_array(tif_path_in)
    # Make a copy of the input array
    ds_array_bounded = np.copy(ds_array)
    # If a LOWER bound is defined, set any cells BELOW that to the lower bound
    if lower_bound != -9999:
        ds_array_bounded[ds_array < lower_bound] = lower_bound
    # If an UPPER bound is defined, set any cells ABOVE that to the upper bound
    if upper_bound != -9999:
        ds_array_bounded[ds_array > upper_bound] = upper_bound
    # Set any cells which were no_data in the input tif to be no_data in the output tif too
    ds_array_bounded[(ds_array==ds_nodatavalue)|(ds_array==no_data_value)] = no_data_value
    # Write result to output GeoTIFF (using same geographical properties as input GeoTIFF)
    array_to_geotiff(ds_array_bounded, tif_path_out, no_data_value, ds_props)
    
# Define a function to fill missing values (no_data_value or nan) in an input array with the mean of their surrounding, valid values
def fill_array_nodata(array, no_data_value):
    # If there are no missing values (no_data_value, nan or inf), simply return the input array
    if not np.any(array==no_data_value) and not np.any(np.isnan(array)) and not np.any(np.isinf(array)):
        return array
    else:
        # Replace no_data_value with np.nan for easier processing
        array = np.where(array==no_data_value, np.nan, array)
        # Get list of array indices for which no_data values are present
        row_idxs = np.where((np.isnan(array))|(np.isinf(array)))[0].tolist()
        col_idxs = np.where((np.isnan(array))|(np.isinf(array)))[1].tolist()
        # Get dimensions of input array
        row_max = array.shape[0]
        col_max = array.shape[1]
        # Make a copy of the input array
        fill_array = np.copy(array).astype(np.float64)
        # Loop through those locations & replace missing values with their neighbourhood average
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            # Get immediately neighbouring cells, accounting for cases where the missing value is along an array edge
            neighbours = array[max(0, row_idx-1):min(row_max, row_idx+2), max(0, col_idx-1):min(col_max, col_idx+2)]
            # If there are valid values amongst these immediate neighbours, use their mean to fill in missing values
            if not np.all(np.isnan(neighbours)):
                fill_array[row_idx, col_idx] = np.nanmean([val for val in neighbours.flatten() if val != no_data_value and not np.isnan(val) and not np.isinf(val)])
            # If not, make no further assumptions & just return the no_data_value for that cell
            else:
                fill_array[row_idx, col_idx] = no_data_value
        return fill_array

# A function to determine if a given Landsat subfolder is valid, based on the current location (paths/rows) & search window being used
def ls_subfolder_valid(subfolder, wrs_prs, search_start, search_end):
    # Check temporal match
    # Get start & end dates of subfolder's data
    start_str = subfolder[17:25]
    start = datetime.date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:8]))
    end_str = subfolder[26:34]
    end = datetime.date(int(end_str[:4]), int(end_str[4:6]), int(end_str[6:8]))
    # Get boolean describing whether or not subfolder data range overlaps with search window
    match_time = (start >= search_start and start <= search_end) or (end >= search_start and end <= search_end)
    # Check spatial match
    wrs_pr = subfolder[10:16]
    # Get boolean describing whether or not subfolder covers spatial area relevant to that zone (based on row & path number)
    match_space = wrs_pr in wrs_prs
    # Return boolean describing whether is a match both temporally and spatially
    return match_time and match_space

# Define a function to calculate various spectral index products using Landsat 7 inputs
def ls7_spectral_index(b1, b2, b3, b4, b5, b7, product_name, product_props, product_path, no_data_value):
    # Mask input arrays wherever no_data_values are present
    b1m, b2m, b3m, b4m, b5m, b7m = [np.ma.masked_where(b==no_data_value, b) for b in [b1, b2, b3, b4, b5, b7]]
    # Normalised Difference Vegetation Index (NDVI): Using Landsat 4-7, NDVI = (B4–B3)/(B4+B3)             # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-difference-vegetation-index?qt-science_support_page_related_con=0#qt-science_support_page_related_con
    if product_name == 'NDVI':
        product_array = (b4m - b3m)/(b4m + b3m)
    # Enhanced Vegetation Index (EVI): Using Landsat 4-7, EVI = 2.5 * ((B4–B3)/(B4 + 6*B3 – 7.5*B1 + 1))   # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-enhanced-vegetation-index?qt-science_support_page_related_con=0#qt-science_support_page_related_con
    elif product_name == 'EVI':
        product_array = 2.5*((b4m - b3m)/(b4m + 6.*b3m - 7.5*b1m + 1.))
    # Advanced Vegetation Index (AVI): Using Landsat 4-7, AVI = [B4 * (1–B3) * (B4–B3)]^1/3                # Source: https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
    elif product_name == 'AVI':
        product_array = np.cbrt(b4m * (1-b3m) * (b4m-b3m))
    # Soil Adjusted Vegetation Index (SAVI): Using Landsat 4-7, SAVI = ((B4–B3)/(B4 + B3 + 0.5))*(1.5)     # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-soil-adjusted-vegetation-index
    elif product_name == 'SAVI':
        product_array = ((b4m - b3m)/(b4m + b3m + 0.5))*(1.5)
    # Modified Soil Adjusted Vegetation Index (MSAVI): Using Landsat 4-7, MSAVI = (2*B4 + 1 – sqrt((2*B4 + 1)^2 – 8*(B4 – B3))) / 2         # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-modified-soil-adjusted-vegetation-index
    elif product_name == 'MSAVI':
        product_array = (2.*b4m + 1. - np.sqrt(np.square(2.*b4m + 1.) - 8.*(b4m - b3m)))/2.
    # Shadow Index (SI): Using Landsat 7, SI = ((1-B1) * (1-B2) * (1-B3))^1/3                              # Source: https://www.geo.university/pages/spectral-indices-with-multispectral-satellite-data
    elif product_name == 'SI':
        product_array = np.cbrt((1-b1m) * (1-b2m) * (1-b3m))
    # Bare Soil Index (BSI): Using Landsat 7, BSI = ((B5+B3) – (B4+B1)) / ((B5+B3) + (B4+B1))              # Source: https://www.geo.university/pages/spectral-indices-with-multispectral-satellite-data
    elif product_name == 'BSI':
        product_array = ((b5m + b3m) - (b4m + b1m))/((b5m + b3m) + (b4m + b1m))
    # Normalised Difference Moisture Index (NDMI): Using Landsat 4-7, NDMI = (B4 – B5) / (B4 + B5)         # Source: https://www.usgs.gov/land-resources/nli/landsat/normalized-difference-moisture-index
    elif product_name == 'NDMI':
        product_array = (b4m - b5m)/(b4m + b5m)
    # Modified Normalised Difference Water Index (MNDWI): MNDWI = (Green - SWIR)/(Green + SWIR)            # Source: Xu 2006
    elif product_name == 'MNDWI':
        product_array = (b2m - b5m)/(b2m + b5m)
    # Automated Water Extraction Index - no shadows (AWEInsh): AWEInsh = 4*(B2-B5) - (0.25*B4 + 2.75*B7)   # Source: https://www.sciencedirect.com/science/article/pii/S0034425713002873
    elif product_name == 'AWEInsh':
        product_array = 4.*(b2m - b5m) - (0.25*b4m + 2.75*b7m)
    # Automated Water Extraction Index - shadows (AWEIsh): AWEIsh = B1 + 2.5*B2 - 1.5*(B4 + B5) - 0.25*B7  # Source: https://www.sciencedirect.com/science/article/pii/S0034425713002873
    elif product_name == 'AWEIsh':
        product_array = b1m + 2.5*b2m - 1.5*(b4m + b5m) - 0.25*b7m
    # Normalised Difference Built-up Index (NDBI): NDBI = (SWIR - NIR)/(SWIR + NIR)                        # Source: Zha et al 2003
    elif product_name == 'NDBI':
        product_array = (b5m - b4m)/(b5m + b4m)
    else:
        print('Unknown spectral index product')
        exit
    # Replace masked values with the no_data_value
    product_array_unmask = product_array.filled(fill_value=no_data_value)
    # Fill any invalid values (no_data, nan, inf) based on their neighbours
    product_array_filled = fill_array_nodata(product_array_unmask, no_data_value)
    # Write filled array to provided output path
    array_to_geotiff(product_array_filled, product_path, no_data_value, product_props)

# Define a function to calculate various spectral index products using Landsat 7 inputs
def ls8_spectral_index(b2, b3, b4, b5, b6, b7, product_name, product_props, product_path, no_data_value):
    # Mask input arrays wherever no_data_values are present
    b2m, b3m, b4m, b5m, b6m, b7m = [np.ma.masked_where(b==no_data_value, b) for b in [b2, b3, b4, b5, b6, b7]]
    # Normalised Difference Vegetation Index (NDVI): Using Landsat 8, NDVI = (Band 5 – Band 4) / (Band 5 + Band 4)           # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-normalized-difference-vegetation-index?qt-science_support_page_related_con=0#qt-science_support_page_related_con
    if product_name == 'NDVI':
        product_array = (b5m - b4m)/(b5m + b4m)
    # Enhanced Vegetation Index (EVI): Using Landsat 8, EVI = 2.5 * ((B5–B4)/(B5 + 6*B4 – 7.5*B2 + 1))   # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-enhanced-vegetation-index?qt-science_support_page_related_con=0#qt-science_support_page_related_con
    elif product_name == 'EVI':
        product_array = 2.5*((b5m - b4m)/(b5m + 6.*b4m - 7.5*b2m + 1.))
    # Advanced Vegetation Index (AVI): Using Landsat 8, AVI = [B5 * (1–B4) * (B5–B4)]^1/3                # Source: https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
    elif product_name == 'AVI':
        product_array = np.cbrt(b5m * (1-b4m) * (b5m-b4m))
    # Soil Adjusted Vegetation Index (SAVI): Using Landsat 8, SAVI = ((B5–B4)/(B5 + B4 + 0.5))*(1.5)     # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-soil-adjusted-vegetation-index
    elif product_name == 'SAVI':
        product_array = ((b5m - b4m)/(b5m + b4m + 0.5))*(1.5)
    # Modified Soil Adjusted Vegetation Index (MSAVI): Using Landsat 8, MSAVI = (2*B5 + 1 – sqrt((2*Band 5 + 1)^2 – 8*(B5 – B4)))/2         # Source: https://www.usgs.gov/land-resources/nli/landsat/landsat-modified-soil-adjusted-vegetation-index
    elif product_name == 'MSAVI':
        product_array = (2.*b5m + 1. - np.sqrt(np.square(2.*b5m + 1.) - 8.*(b5m - b4m)))/2.
    # Shadow Index (SI): Using Landsat 8, SI = ((1-B2) * (1-B3) * (1-B4))^1/3                            # Source: https://www.geo.university/pages/spectral-indices-with-multispectral-satellite-data
    elif product_name == 'SI':
        product_array = np.cbrt((1-b2m) * (1-b3m) * (1-b4m))
    # Bare Soil Index (BSI): Using Landsat 8, BSI = ((B6+B4) – (B5+B2)) / ((B6+B4) + (B5+B2))            # Source: https://www.geo.university/pages/spectral-indices-with-multispectral-satellite-data
    elif product_name == 'BSI':
        product_array = ((b6m + b4m) - (b5m + b2m))/((b6m + b4m) + (b5m + b2m))
    # Normalised Difference Moisture Index (NDMI): Using Landsat 8, NDMI = (B5 – B6)/(B5 + B6)           # Source: https://www.usgs.gov/land-resources/nli/landsat/normalized-difference-moisture-index
    elif product_name == 'NDMI':
        product_array = (b5m - b6m)/(b5m + b6m)
    # Modified Normalised Difference Water Index (MNDWI): MNDWI = (Green - SWIR)/(Green + SWIR)          # Source: Xu 2006
    elif product_name == 'MNDWI':
        product_array = (b3m - b6m)/(b3m + b6m)
    # Automated Water Extraction Index - no shadows (AWEInsh): AWEInsh = 4×(Green−SWIR1) - (0.25×NIR + 2.75×SWIR1)    # Source: https://com-mendeley-prod-publicsharing-pdfstore.s3.eu-west-1.amazonaws.com/91d8-CC-BY-2/10.3390/s18082580.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCWV1LXdlc3QtMSJIMEYCIQDMSUvuxmjtCbKp4ELxH7RffvhJ%2Fg%2BgOx%2BsEg9q042vCQIhAOfJDEC6KNdoYsyjnVn68tW%2BSTqG2lleqbulNUqUuNrbKv8DCN3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMMTA4MTY2MTk0NTA1IgzHP%2BhH2%2F0s8f78mmEq0wM8%2FrjOi%2F0ioXv10KSM7UaO2BB5tqRcy7GTbmhGwhLtIIZT8lE4Kc7%2F%2BVn8nHC7iL1QHrhRUeoE%2FCIXa%2BoS3vBT%2BqPS7aNzHDE2Sb69TWBo%2BvgIDLT%2FnBsryAqqzzWJK6U02svzRU4w6yn%2F4AxzTkWMb70LdW3a8ZGplFnQ9RIT8CORy6VrJK%2FmCJuwaxHbNnhySDT0CtLShBIfkzkZl2FqGLyLeYxee%2FC%2FNOnpzBIPQKViDj3fsBwb3IAK9mNQWAoANtHi9FEJldHoSsi0AX%2FUgIO6crBgU3ZRyLUU%2BRjOrSs7PN1ErjEQv84wKyc1S%2B6XPt4VZqmXYP%2BAZel2mOkGpFSgeLHNdRIydYxgy5KiJ7C5AOUhkKMrsINl1TRM6QWHr6QecPL5ZffnaEkBdqBTpreZe%2Fg%2Fc3XkRhBQL9DvxaQLDdgVMn0vzxpaS0WRLZ2otpdMWKLkv7KqoodqbVPKhB8BokdeeYtGcQCIlUU06rmX7HjfBtbn8dtvWNaUFAR%2FnWXOSa2SqBOhkWc1SLRLphgxVkCee4iiiOjkyeOKSYz%2FlUvCf6alO299Rpae5jPj0M8FsONyG6lYM8ykZiWiLYkB7mRuGUQ2Ajz1UxnLO1d83TCU%2BoP0BTruAV4WmpBcyXEOvi%2Fr%2FMw7FF7fVcaiivVOt2qog1gGPB7VOgqf3bJX0yzb27SUEapuwRXT5elft1zHnx9qnpOPY%2FPfrnLqiRSqLqxvW394j4KFI3eoEdekWBL7SIgz3e2f7%2Fb41VjIDLUP3i6dY5eRifuUl8pGjzRolvWX%2F5gCrvJXZLtYO%2F3hCRiTlNOrR57SI0DqVGIa0RwqlOa%2FNj%2Fjis1d1XsHM%2FNV3w0bRpGApVkn1XoZ0B3ynj%2BFGeHxveZbrv1cHsCABnSWaujUEGsjV9taSiU9rip6qkaUuySXq8nA%2BhgexaWq%2B9Srp53SY7Q%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200329T222610Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIARSLZVEVE3TKOBXMJ%2F20200329%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Signature=70606fcbdc612d8c4c5f40231345c9dfb471be989bf4aa75d51fdcefc5a147a2
    elif product_name == 'AWEInsh':
        product_array = 4.*(b3m - b6m) - (0.25*b5m + 2.75*b6m)
    # Automated Water Extraction Index - shadows (AWEIsh): AWEIsh = Blue + 2.5×Green−1.5×(NIR + SWIR1)−0.25×SWIR2     # Source: https://com-mendeley-prod-publicsharing-pdfstore.s3.eu-west-1.amazonaws.com/91d8-CC-BY-2/10.3390/s18082580.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCWV1LXdlc3QtMSJIMEYCIQDMSUvuxmjtCbKp4ELxH7RffvhJ%2Fg%2BgOx%2BsEg9q042vCQIhAOfJDEC6KNdoYsyjnVn68tW%2BSTqG2lleqbulNUqUuNrbKv8DCN3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMMTA4MTY2MTk0NTA1IgzHP%2BhH2%2F0s8f78mmEq0wM8%2FrjOi%2F0ioXv10KSM7UaO2BB5tqRcy7GTbmhGwhLtIIZT8lE4Kc7%2F%2BVn8nHC7iL1QHrhRUeoE%2FCIXa%2BoS3vBT%2BqPS7aNzHDE2Sb69TWBo%2BvgIDLT%2FnBsryAqqzzWJK6U02svzRU4w6yn%2F4AxzTkWMb70LdW3a8ZGplFnQ9RIT8CORy6VrJK%2FmCJuwaxHbNnhySDT0CtLShBIfkzkZl2FqGLyLeYxee%2FC%2FNOnpzBIPQKViDj3fsBwb3IAK9mNQWAoANtHi9FEJldHoSsi0AX%2FUgIO6crBgU3ZRyLUU%2BRjOrSs7PN1ErjEQv84wKyc1S%2B6XPt4VZqmXYP%2BAZel2mOkGpFSgeLHNdRIydYxgy5KiJ7C5AOUhkKMrsINl1TRM6QWHr6QecPL5ZffnaEkBdqBTpreZe%2Fg%2Fc3XkRhBQL9DvxaQLDdgVMn0vzxpaS0WRLZ2otpdMWKLkv7KqoodqbVPKhB8BokdeeYtGcQCIlUU06rmX7HjfBtbn8dtvWNaUFAR%2FnWXOSa2SqBOhkWc1SLRLphgxVkCee4iiiOjkyeOKSYz%2FlUvCf6alO299Rpae5jPj0M8FsONyG6lYM8ykZiWiLYkB7mRuGUQ2Ajz1UxnLO1d83TCU%2BoP0BTruAV4WmpBcyXEOvi%2Fr%2FMw7FF7fVcaiivVOt2qog1gGPB7VOgqf3bJX0yzb27SUEapuwRXT5elft1zHnx9qnpOPY%2FPfrnLqiRSqLqxvW394j4KFI3eoEdekWBL7SIgz3e2f7%2Fb41VjIDLUP3i6dY5eRifuUl8pGjzRolvWX%2F5gCrvJXZLtYO%2F3hCRiTlNOrR57SI0DqVGIa0RwqlOa%2FNj%2Fjis1d1XsHM%2FNV3w0bRpGApVkn1XoZ0B3ynj%2BFGeHxveZbrv1cHsCABnSWaujUEGsjV9taSiU9rip6qkaUuySXq8nA%2BhgexaWq%2B9Srp53SY7Q%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200329T222610Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIARSLZVEVE3TKOBXMJ%2F20200329%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Signature=70606fcbdc612d8c4c5f40231345c9dfb471be989bf4aa75d51fdcefc5a147a2
    elif product_name == 'AWEIsh':
        product_array = b2m + 2.5*b3m - 1.5*(b5m + b6m) - 0.25*b7m
    # Normalised Difference Built-up Index (NDBI): NDBI = (SWIR - NIR)/(SWIR + NIR)                      # Source: Zha et al 2003
    elif product_name == 'NDBI':
        product_array = (b6m - b5m)/(b6m + b5m)
    else:
        print('Unknown spectral index product')
        exit
    # Replace masked values with the no_data_value
    product_array_unmask = product_array.filled(fill_value=no_data_value)
    # Fill any invalid values (no_data, nan, inf) based on their neighbours
    product_array_filled = fill_array_nodata(product_array_unmask, no_data_value)
    # Write filled array to provided output path
    array_to_geotiff(product_array_filled, product_path, no_data_value, product_props)

# Define a function to return a patch geom (as WKT string), relating to the target/output patch
def get_target_patch_geom(i, j, props, pad, dim_out):
    x_min, y_max, res_x, res_y = [props[key] for key in ['x_min','y_max','res_x','res_y']]
    w = x_min + res_x*(pad + i*dim_out)
    e = x_min + res_x*(pad + (i+1)*dim_out)
    n = y_max + res_y*(pad + j*dim_out)
    s = y_max + res_y*(pad + (j+1)*dim_out)
    target_patch_wkt = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'.format(w, s, w, n, e, n, e, s, w, s)
    return target_patch_wkt

# Define a function to return a patch geom (as WKT string), relating to the feature/input patch
def get_feature_patch_geom(i, j, props, pad, dim_in, dim_out):
    x_min, y_max, res_x, res_y = [props[key] for key in ['x_min','y_max','res_x','res_y']]
    w = x_min + res_x*i*dim_out
    e = x_min + res_x*(i*dim_out + dim_in)
    n = y_max + res_y*j*dim_out
    s = y_max + res_y*(j*dim_out + dim_in)
    feature_patch_wkt = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'.format(w, s, w, n, e, n, e, s, w, s)
    return feature_patch_wkt

# Define a function to return a particular target patch of an input data array
def get_target_patch_array(i, j, full_array, pad, dim_out):
    target_patch_array = full_array[pad + j*dim_out: pad + (j+1)*dim_out, pad + i*dim_out: pad + (i+1)*dim_out]
    return target_patch_array

# Define a function to return a particular feature patch of an input data array
def get_feature_patch_array(i, j, full_array, dim_in, dim_out):
    feature_patch_array = full_array[j*dim_out: j*dim_out + dim_in, i*dim_out: i*dim_out + dim_in]
    return feature_patch_array

# Define a function to buffer an array with no_data values, if its shape is smaller than expected
def pad_array(array, expected_dim, pad_value):
    pad_y = expected_dim - array.shape[0]
    pad_x = expected_dim - array.shape[1]
    padded_array = np.pad(array, pad_width=((0, pad_y), (0, pad_x)), mode='constant', constant_values=pad_value)
    return padded_array