# Process: OpenStreetMap (OSM) downloads

# Import required packages
import os, shutil
import osmnx as ox
from shapely.geometry import Polygon

# Define folders
folder_osm = 'E:/mdm123/D/data/OSM/raw'

# Define bounding box containing all LiDAR survey sites
bbox_n, bbox_s, bbox_e, bbox_w = -40.3, -42.2, 174.5, 172.1
bbox_polygon = Polygon([(bbox_w, bbox_s), (bbox_e, bbox_s), (bbox_e, bbox_n), (bbox_w, bbox_n)])

# Download all roads in AOI - to be filtered later during rasterisation process
print('\nDownloading road data...', end=' ')
if os.path.exists('{}/Rds'.format(folder_osm)):shutil.rmtree('{}/Rds'.format(folder_osm))
osm_roads = ox.graph_from_bbox(bbox_n, bbox_s, bbox_e, bbox_w, network_type='drive_service', retain_all=True, truncate_by_edge=True, simplify=True)
#ox.plot_graph(osm_roads)
ox.save_graph_shapefile(osm_roads, filename='Rds', folder=folder_osm)
print('DONE')

# Download all building footprints in AOI
print('\nDownloading building data...', end=' ')
if os.path.exists('{}/Bld'.format(folder_osm)): shutil.rmtree('{}/Bld'.format(folder_osm))
osm_blds = ox.footprints_from_polygon(bbox_polygon)
#ox.plot_shape(osm_blds)
osm_blds.drop(labels='nodes', axis=1).to_file('{}/Bld'.format(folder_osm))
print('DONE')

# Download all waterways in AOI
print('\nDownloading waterway data...', end=' ')
if os.path.exists('{}/Wtr'.format(folder_osm)): shutil.rmtree('{}/Wtr'.format(folder_osm))
osm_water = ox.graph_from_bbox(north=bbox_n, south=bbox_s, east=bbox_e, west=bbox_w, retain_all=True, truncate_by_edge=True, simplify=True, network_type='none', infrastructure='way["waterway"]')
#ox.plot_graph(osm_water)
ox.save_graph_shapefile(osm_water, filename='Wtr', folder=folder_osm)
print('DONE')

# Download rivers specifically in AOI
print('\nDownloading rivers data...', end=' ')
if os.path.exists('{}/Riv'.format(folder_osm)): shutil.rmtree('{}/Riv'.format(folder_osm))
osm_rivers = ox.graph_from_bbox(bbox_n, bbox_s, bbox_e, bbox_w, retain_all=True, truncate_by_edge=True, simplify=True, network_type='none', infrastructure='way["waterway"~"river"]')
#ox.plot_graph(osm_rivers)
ox.save_graph_shapefile(osm_rivers, filename='Riv', folder=folder_osm)
print('DONE')

# Download river banks in AOI, in case they're useful in identifying bridges
print('\nDownloading riverbank data...', end=' ')
if os.path.exists('{}/RvB'.format(folder_osm)): shutil.rmtree('{}/RvB'.format(folder_osm))
osm_riverbanks = ox.graph_from_bbox(bbox_n, bbox_s, bbox_e, bbox_w, retain_all=True, truncate_by_edge=True, simplify=True, network_type='none', infrastructure='way["waterway"~"riverbank"]')
#ox.plot_graph(osm_riverbanks)
ox.save_graph_shapefile(osm_riverbanks, filename='RvB', folder=folder_osm)
print('DONE')