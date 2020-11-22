# Using machine learning to improve free topography data for flood modelling

As part of the requirements for the [Master of Disaster Risk & Resilience](https://www.canterbury.ac.nz/study/qualifications-and-courses/masters-degrees/master-of-disaster-risk-and-resilience/) programme at the [University of Canterbury](https://www.canterbury.ac.nz/), this research project explored the potential for machine learning models to make free Digital Surface Models (such as the widely-used SRTM) more applicable for flood modelling, by stripping away vertical biases relating to vegetation & built-up areas to get a Digital Terrain Model ("bare earth").

The image below visualises the performance of one of these models (a fully-convolutional neural network) in one of the three test zones considered (i.e. data unseen during model training & validation, used to assess the model's ability to generalise to new locations). Full details are available in the associated journal article: Meadows & Wilson 2020, etc...

All Python code fragments used during this research are shared here (covering preparing input data, building & training three different ML models, and visualising the results), in the hope that they'll be useful for others doing related work. Please note this code includes lots of exploratory steps & some dead ends, and is not a refined step-by-step template for applying this approach in a new location.


![graphical_abstract](/images/graphical_abstract_boxplots.png)

Scripts are stored in folders relating to the virtual environments within which they were run, along with a text file summarising all packages loaded in each environment:

* geo: geospatial processing & mapping
* sklearn: development of Random Forest model
* tf2: development of neural network models
* osm: downloading OpenStreetMap data

