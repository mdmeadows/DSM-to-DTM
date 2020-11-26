# Using machine learning to improve free topography data for flood modelling

As part of the requirements for the [Master of Disaster Risk & Resilience](https://www.canterbury.ac.nz/study/qualifications-and-courses/masters-degrees/master-of-disaster-risk-and-resilience/) programme at the [University of Canterbury](https://www.canterbury.ac.nz/), this research project explored the potential for machine learning models to make free Digital Surface Models (such as the widely-used SRTM) more applicable for flood modelling, by stripping away vertical biases relating to vegetation & built-up areas to get a Digital Terrain Model ("bare earth").

The image below visualises the performance of one of these models (a fully-convolutional neural network) in one of the three test zones considered (i.e. data unseen during model training & validation, used to assess the model's ability to generalise to new locations). Full details are available in the associated journal article: Meadows & Wilson 2020, etc...  

![graphical_abstract](/images/graphical_abstract_boxplots.png)  

## Python scripts

All Python code fragments used during this research are shared here (covering preparing input data, building & training three different ML models, and visualising the results), in the hope that they'll be useful for others doing related work. Please note this code includes exploratory steps & some dead ends, and is not a refined step-by-step template for applying this approach in a new location.

Scripts are stored in folders relating to the virtual environments within which they were run, along with a text file summarising all packages loaded in each environment:

- [geo](/scripts/geo/): geospatial processing & mapping
- [sklearn](/scripts/sklearn/): development of Random Forest model
- [tf2](/scripts/tf2/): development of neural network models
- [osm](/scripts/osm/): downloading OpenStreetMap data


## Brief summary of approach taken



## Brief summary of input datasets used

A guiding principle for the project was that all input data should be available for free and with global (or near-global) coverage, so as to maximise applicability in low-income countries/contexts. While these datasets were too big to store here, all can be downloaded for free and relatively easily (usually involving signing up to each platform) based on the notes below.

**Digital Surface Models (DSMs)**
- SRTM: Downloaded from [EarthExplorer](https://earthexplorer.usgs.gov/) under Digital Elevation > SRTM > SRTM 1 Arc-Second Global
- ASTER: Downloaded from [EarthData Search](https://search.earthdata.nasa.gov/search) ("ASTER Global Digital Elevation Model V003")
- AW3D30: Downloaded from [Earth Observation Research Centre](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/) (Version 2.2, the latest available at the time)

**Multi-spectral imagery**
- Landsat-7: Downloaded from [EarthExplorer](https://earthexplorer.usgs.gov/) under 
- Landsat-8: Downloaded from [EarthExplorer](https://earthexplorer.usgs.gov/) under 

**Night-time light**
- DMSP-OLS Nighttime Lights Time Series (annual composites covering SRTM data collection period): Downloaded from [NOAA EOG](https://ngdc.noaa.gov/eog/dmsp/downloadV4composites.html)
- VIIRS Day/Night Band Nighttime Lights (monthly composites covering LiDAR data collection period): Downloaded from [CSM EOG](https://eogdata.mines.edu/download_dnb_composites.html)

**Others**
- Global forest canopy height: Developed by [Simard et al. 2011](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JG001708) and available for download [here](https://landscape.jpl.nasa.gov/)
- Global forest cover: Developed by [Hansen et al. 2013](https://science.sciencemag.org/content/342/6160/850) and available for download [here](https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.6.html)
- Global surface water: Developed by [Pekel et al. 2016](https://www.nature.com/articles/nature20584) and available for download [here](https://global-surface-water.appspot.com/download)
- Night-time light (): 
- OpenStreetMap layers: Downloaded using the [OSMnx](https://github.com/gboeing/osmnx) Python module developed by [Boeing 2017](https://www.sciencedirect.com/science/article/pii/S0198971516303970)


A few other datasets are referred to in the code, not as inputs to the machine learning models but just as references to better understand the results.

**Additional datasets (not used as inputs to ML models)**
- MERIT DSM: Improved DSM developed by [Yamazaki et al. 2017](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL072874), with a request form for the data available [here](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/)
- Land Cover Database
