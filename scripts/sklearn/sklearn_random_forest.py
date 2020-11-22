# Predictive model: Random Forest regression

# Import required packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from boruta import BorutaPy
from mlxtend.feature_selection import SequentialFeatureSelector
import optuna
from math import floor
import time
import matplotlib.pyplot as plt
import pickle
import joblib

# Define paths to relevant folders
folder_input_1D = 'E:/mdm123/D/ML/inputs/1D'
folder_models = 'E:/mdm123/D/ML/models/rf'
folder_logs = 'E:/mdm123/D/ML/logs/rf'
folder_results = 'E:/mdm123/D/ML/results/rf'
folder_fig = 'E:/mdm123/D/figures/models/rf'

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define a dictionary of labels for all available features (for figure generation)
feature_labels = {'srtm_z':'SRTM (Elevation)', 'srtm_slope':'SRTM (Slope)', 'srtm_aspect':'SRTM (Aspect)', 'srtm_roughness':'SRTM (Roughness)', 'srtm_tpi':'SRTM (TPI)', 'srtm_tri':'SRTM (TRI)',
                  'aster_z':'ASTER (Elevation)', 'aster_slope':'ASTER (Slope)', 'aster_aspect':'ASTER (Aspect)', 'aster_roughness':'ASTER (Roughness)', 'aster_tpi':'ASTER (TPI)', 'aster_tri':'ASTER (TRI)',
                  'aw3d30_z':'AW3D30 (Elevation)', 'aw3d30_slope':'AW3D30 (Slope)', 'aw3d30_aspect':'AW3D30 (Aspect)', 'aw3d30_roughness':'AW3D30 (Roughness)', 'aw3d30_tpi':'AW3D30 (TPI)', 'aw3d30_tri':'AW3D30 (TRI)',
                  'ls7_b1':'Landsat 7 (Band 1)', 'ls7_b2':'Landsat 7 (Band 2)', 'ls7_b3':'Landsat 7 (Band 3)', 'ls7_b4':'Landsat 7 (Band 4)', 'ls7_b5':'Landsat 7 (Band 5)', 'ls7_b6_vcid_1':'Landsat 7 (Band 6_VCID_1)', 'ls7_b6_vcid_2':'Landsat 7 (Band 6_VCID_2)', 'ls7_b7':'Landsat 7 (Band 7)', 'ls7_b8':'Landsat 7 (Band 8)', 'ls7_evi':'Landsat 7 (EVI)', 'ls7_mndwi':'Landsat 7 (MNDWI)', 'ls7_msavi':'Landsat 7 (MSAVI)', 'ls7_ndbi':'Landsat 7 (NDBI)', 'ls7_ndmi':'Landsat 7 (NDMI)', 'ls7_ndvi':'Landsat 7 (NDVI)', 'ls7_savi':'Landsat 7 (SAVI)', 'ls7_avi':'Landsat 7 (AVI)', 'ls7_si':'Landsat 7 (SI)', 'ls7_bsi':'Landsat 7 (BSI)', 'ls7_aweinsh':'Landsat 7 (AWEInsh)', 'ls7_aweish':'Landsat 7 (AWEIsh)',
                  'ls8_b1':'Landsat 8 (Band 1)', 'ls8_b2':'Landsat 8 (Band 2)', 'ls8_b3':'Landsat 8 (Band 3)', 'ls8_b4':'Landsat 8 (Band 4)', 'ls8_b5':'Landsat 8 (Band 5)', 'ls8_b6':'Landsat 8 (Band 6)', 'ls8_b7':'Landsat 8 (Band 7)', 'ls8_b8':'Landsat 8 (Band 8)', 'ls8_b9':'Landsat 8 (Band 9)', 'ls8_b10':'Landsat 8 (Band 10)', 'ls8_b11':'Landsat 8 (Band 11)', 'ls8_evi':'Landsat 8 (EVI)', 'ls8_mndwi':'Landsat 8 (MNDWI)', 'ls8_msavi':'Landsat 8 (MSAVI)', 'ls8_ndbi':'Landsat 8 (NDBI)', 'ls8_ndmi':'Landsat 8 (NDMI)', 'ls8_ndvi':'Landsat 8 (NDVI)', 'ls8_savi':'Landsat 8 (SAVI)', 'ls8_avi':'Landsat 8 (AVI)', 'ls8_si':'Landsat 8 (SI)', 'ls8_bsi':'Landsat 8 (BSI)', 'ls8_aweinsh':'Landsat 8 (AWEInsh)', 'ls8_aweish':'Landsat 8 (AWEIsh)',
                  'gsw_occurrence':'Global Surface Water (Occurrence)',
                  'osm_bld':'OpenStreetMap (Buildings)', 'osm_rds':'OpenStreetMap (Roads)', 'osm_brd':'OpenStreetMap (Bridges)',
                  'gch':'Global Canopy Heights',
                  'gfc':'Global Forest Cover',
                  'ntl_viirs':'Night-time Light (VIIRS)',
                  'ntl_dmsp_avg_vis':'Night-time Light (DMSP), Avg_Vis', 'ntl_dmsp_avg_vis_stable':'Night-time Light (DMSP), Avg_Vis_Stable', 'ntl_dmsp_pct_lights':'Night-time Light (DMSP), Pct_Lights', 'ntl_dmsp_avg_lights_pct':'Night-time Light (DMSP), Avg_Lights_Pct'}

# Define a dictionary of colours for each dataset
dataset_colours = {'train':'blue', 'dev':'green', 'test':'firebrick'}

# Define nodata value
no_data = -9999


###############################################################################
# 1. Import 1D data - separate CSV files for training & dev (validation) sets #
###############################################################################

# 1a. Import processed CSVs into dataframes
data_train = pd.read_csv('{}/Input1D_Train.csv'.format(folder_input_1D))
data_dev = pd.read_csv('{}/Input1D_Dev.csv'.format(folder_input_1D))

# 1b. Define target & feature columns to be used - removing Landsat bands not processed to surface reflectance (based on visual assessment of final quality)
target_col = 'diff'
features_col = [col for col in data_train.columns if col not in ['idx_all','idx_valid','zone','i','j','target_pixels','diff','lcdb','n_pixels','usage','ls7_b6_vcid_1','ls7_b6_vcid_2','ls7_b8','ls8_b8','ls8_b9','ls8_b10','ls8_b11']]

# 1c. Filter the dataframes available (train & dev) to contain only target & feature columns
target_train = data_train[target_col].copy()
target_dev = data_dev[target_col].copy()
features_train = data_train[features_col].copy()
features_dev = data_dev[features_col].copy()


###############################################################################
# 2. Feature selection - choose optimal subset of available features          #
###############################################################################

# 2a. Establish the initial & baseline errors for the training & dev sets

# Establish the INITIAL error (MSE) for all data sets
pred_train_initial = np.zeros(np.array(target_train).shape)
pred_dev_initial = np.zeros(np.array(target_dev).shape)
RMSE_train_initial = np.sqrt(mean_squared_error(target_train, pred_train_initial))    # 8.67994869425992
RMSE_dev_initial = np.sqrt(mean_squared_error(target_dev, pred_dev_initial))          # 8.566652821179863

# Establish the BASELINE error (MSE) for the training & dev sets - based on AVERAGE error in TRAINING set
pred_train_baseline = np.repeat(target_train.mean(), len(target_train))
pred_dev_baseline = np.repeat(target_train.mean(), len(target_dev))
RMSE_train_baseline = np.sqrt(mean_squared_error(target_train, pred_train_baseline))  # 7.514175931567595
RMSE_dev_baseline = np.sqrt(mean_squared_error(target_dev, pred_dev_baseline))        # 7.474215379503667


# 2b. Feature selection using the Boruta all-relevant feature selection method

# Initialise a Random Forest regression model, using all cores and a max_depth of 5 (based on recommended range of 3-7)
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

# Initialise the Boruta feature selection method
feature_selection_boruta = BorutaPy(estimator=rf, max_iter=50, n_estimators='auto', verbose=2, random_state=1)

# Run the all-relevant feature selection, using the training dataset
feature_selection_boruta.fit(features_train.values, target_train.values.ravel())

# Check which features were found to be useful
features_confirmed_boruta = list(features_train.columns[feature_selection_boruta.support_])
features_tentative_boruta = list(features_train.columns[feature_selection_boruta.support_weak_])

# Generate a dictionary recording the ranking for each input feature name
features_ranking_boruta = {}
for i, feature in enumerate(features_train.columns):
    features_ranking_boruta[feature] = feature_selection_boruta.ranking_[i]

# Get a list of whichever features aren't ranked equal last
last_ranking = max(features_ranking_boruta.values())
features_confirmed_boruta_weak = []
for feature in features_train.columns:
    if features_ranking_boruta[feature] != last_ranking:
        features_confirmed_boruta_weak.append(feature)

# Save all results for later reference
pickle.dump(features_confirmed_boruta, open('{}/feature_selection_boruta_confirmed.p'.format(folder_logs), 'wb'))
pickle.dump(features_tentative_boruta, open('{}/feature_selection_boruta_tentative.p'.format(folder_logs), 'wb'))
pickle.dump(features_ranking_boruta, open('{}/feature_selection_boruta_ranking.p'.format(folder_logs), 'wb'))
pickle.dump(features_confirmed_boruta_weak, open('{}/feature_selection_boruta_confirmed_weak.p'.format(folder_logs), 'wb'))

# features_confirmed: ['srtm_z','srtm_aspect','srtm_tpi','aw3d30_tpi','aw3d30_tri','ls7_b3','ls7_b5','ls7_ndvi','ls7_bsi','ls7_aweinsh','ls8_b7','ls8_aweinsh','gch','gfc']
# features_tentative: []
# features_confirmed_weak: ['srtm_z','srtm_slope','srtm_aspect','srtm_tpi','aw3d30_aspect','aw3d30_tpi','aw3d30_tri','ls7_b3','ls7_b5','ls7_ndvi','ls7_bsi','ls7_aweinsh','ls8_b3','ls8_b7','ls8_ndvi','ls8_bsi','ls8_aweinsh','gch','gfc']

# Fit & evaluate RF model using the CONFIRMED features (14) selected above by the boruta process
rf_boruta = RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=2)
rf_boruta.fit(features_train[features_confirmed_boruta], target_train)
pred_train_boruta = rf_boruta.predict(features_train[features_confirmed_boruta])
pred_dev_boruta = rf_boruta.predict(features_dev[features_confirmed_boruta])
RMSE_train_boruta = np.sqrt(mean_squared_error(target_train, pred_train_boruta))  # 1.3569360561689674
RMSE_dev_boruta = np.sqrt(mean_squared_error(target_dev, pred_dev_boruta))        # 4.1348509644464375

# Fit & evaluate RF model using the WEAKLY CONFIRMED features (xx) selected above by the boruta process
rf_boruta_weak = RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=2)
rf_boruta_weak.fit(features_train[features_confirmed_boruta_weak], target_train)
pred_train_boruta_weak = rf_boruta_weak.predict(features_train[features_confirmed_boruta_weak])
pred_dev_boruta_weak = rf_boruta_weak.predict(features_dev[features_confirmed_boruta_weak])
RMSE_train_boruta_weak = np.sqrt(mean_squared_error(target_train, pred_train_boruta_weak))  # 1.3248452387054266
RMSE_dev_boruta_weak = np.sqrt(mean_squared_error(target_dev, pred_dev_boruta_weak))        # 4.078737855325556

print('RF trained on STRONG boruta subset ({}) reduces train RMSE by {:.1%} and dev RMSE by {:.1%}'.format(len(features_confirmed_boruta), (RMSE_train_initial-RMSE_train_boruta)/RMSE_train_initial, (RMSE_dev_initial-RMSE_dev_boruta)/RMSE_dev_initial))
# RF trained on STRONG boruta subset (14) reduces train RMSE by 84.4% and dev RMSE by 51.7%
print('RF trained on WEAK boruta subset ({}) reduces train RMSE by {:.1%} and dev RMSE by {:.1%}'.format(len(features_confirmed_boruta_weak), (RMSE_train_initial-RMSE_train_boruta_weak)/RMSE_train_initial, (RMSE_dev_initial-RMSE_dev_boruta_weak)/RMSE_dev_initial))
# RF trained on WEAK boruta subset (19) reduces train RMSE by 84.7% and dev RMSE by 52.4%


# 2c: Feature selection using step floating forward selection (SFFS) in the mlxtend package

# SFFS is comprehensive but time-consuming - not possible to use large RFs & all training data

# Define the number of trees for each forest & the percentage of training data to use
n_trees = 10
sample_percentage = 10

# To make the SFFS feasible, use a random sample of the available training data
all_rows_n = target_train.size
sample_rows_n = int(sample_percentage/100. * all_rows_n)
np.random.seed(seed=1)
sample_rows_idx = np.random.choice(all_rows_n, size=sample_rows_n, replace=False)

# Filter the training data to get this sample for feature & target data
sample_features_train = features_train.iloc[sample_rows_idx].copy()
sample_target_train = target_train.iloc[sample_rows_idx].copy()

# Initialise a dictionary to hold results - or load previous progress to keep adding to it
#sfs_dict = {i+1:{'features_selected':None, 'cv_score':None, 'dev_score':None} for i in range(len(features_col))}
sfs_dict = pickle.load(open('{}/feature_selection_sffs_overview.p'.format(folder_logs), 'rb'))

# Investigate different feature subset sizes, saving out results for each for monitoring along the way
for k in range(1, len(features_col)+1):
    
    # Initialise the Sequential Feature Selector object
    sfs = SequentialFeatureSelector(RandomForestRegressor(n_estimators=n_trees, random_state=1),
                                    k_features = k,
                                    forward = True,
                                    floating = True,
                                    verbose = 1,
                                    scoring = 'neg_root_mean_squared_error',
                                    cv = 5,
                                    n_jobs = -1)
    
    # Fit the SFS object to the training data sample
    sfs = sfs.fit(sample_features_train, sample_target_train)
    
    # Get a list of selected features & write these to a pickle object for later reference
    features_selected = list(sfs.k_feature_names_)
    sfs_dict[k]['features_selected'] = features_selected
    print('\n\n--------------------\n{} features:'.format(k), ', '.join(features_selected))
    
    # Get the average cross-validation score
    cv_score = abs(sfs.k_score_)
    sfs_dict[k]['cv_score'] = cv_score
    print(' - CV RMSE: {:.4f} m'.format(cv_score))
    
    # Now train a RF using all training data & evaluate using the dev set
    rf_sfs = RandomForestRegressor(n_estimators=n_trees, random_state=2)
    rf_sfs.fit(features_train[features_selected], target_train)
    pred_dev_sfs = rf_sfs.predict(features_dev[features_selected])
    RMSE_dev_sfs = np.sqrt(mean_squared_error(target_dev, pred_dev_sfs))
    sfs_dict[k]['dev_score'] = RMSE_dev_sfs
    print(' - Dev RMSE: {:.4f} m\n--------------------\n\n'.format(RMSE_dev_sfs))
    
    # Write dictionary to pickle object
    with open('{}/feature_selection_sffs_overview.p'.format(folder_logs), 'wb') as f:
        pickle.dump(sfs_dict, f, pickle.HIGHEST_PROTOCOL)
    
    # Generate plot showing cross-validation score vs subset size for all sizes evaluated so far
    plot_sizes = range(1, k+1)
    plot_cv_scores = [sfs_dict[size]['cv_score'] for size in plot_sizes]
    plot_dev_scores = [sfs_dict[size]['dev_score'] for size in plot_sizes]
    fig, axes = plt.subplots(figsize=(9,6))
    # Training data - average cross-validation score based on training data sample
    axes.scatter(x=plot_sizes, y=plot_cv_scores, label='SFFS CV error', color=dataset_colours['train'], alpha=0.5)
    axes.scatter(x=plot_sizes[np.argmin(plot_cv_scores)], y=min(plot_cv_scores), fc='none', ec='black')
    # Dev data - prediction error for RF trained on all training data & checked using dev data
    axes.scatter(x=plot_sizes, y=plot_dev_scores, label='Dev error', color=dataset_colours['dev'], alpha=0.5)
    axes.scatter(x=plot_sizes[np.argmin(plot_dev_scores)], y=min(plot_dev_scores), fc='none', ec='black')
    # General figure properties
    axes.set_xlabel('Number of features selected')
    axes.set_ylabel('Validation error, RMSE [m]')
    axes.set_title('Validation results for SFFS feature selection ({}-tree forests using {}% of training data)'.format(n_trees, sample_percentage))
    axes.grid(axis='y', which='major', alpha=0.25)
    axes.legend(frameon=False)
    fig.tight_layout()
    fig.savefig('{}/feature_selection/sffs_{}trees_{}percentsample.png'.format(folder_fig, str(n_trees).zfill(2), str(sample_percentage).zfill(2)), dpi=300)
    plt.close()


# Train 100-tree RFs on each k-feature subset and return training & dev scores
for k in range(1, len(features_col)+1):
    
    # Process if results are available for that k-value
    if sfs_dict[k]['features_selected']:
    
        print('\nTraining using {} features:'.format(k))
        
        # Retrieve the feature subset of that length
        features_selected = sfs_dict[k]['features_selected']
        
        # Determine the maximum number of features allowed for each tree (using recommended default of p/3)
        max_features = max([1, floor(len(features_selected)/3)])
        
        # Initialise a 100-tree Random Forest regressor
        rf = RandomForestRegressor(n_estimators=100, max_features=max_features, random_state=1)
        
        # Train it on full training dataset (selected features only)
        rf.fit(features_train[features_selected], target_train)
        
        # Get predictions for the training & dev (validation) sets
        pred_train_sfs = rf.predict(features_train[features_selected])
        pred_dev_sfs = rf.predict(features_dev[features_selected])
        
        # Get RMSE for each set
        RMSE_train_sfs = np.sqrt(mean_squared_error(target_train, pred_train_sfs))
        RMSE_dev_sfs = np.sqrt(mean_squared_error(target_dev, pred_dev_sfs))
        print(' - {:.4f}m (train)\n - {:.4f}m (dev)'.format(RMSE_train_sfs, RMSE_dev_sfs))
        
        # Write new results to the SFS dictionary
        sfs_dict[k]['RMSE_train'] = RMSE_train_sfs
        sfs_dict[k]['RMSE_dev'] = RMSE_dev_sfs
        
        # Write dictionary to pickle object
        with open('{}/feature_selection_sffs_overview.p'.format(folder_logs), 'wb') as f:
            pickle.dump(sfs_dict, f, pickle.HIGHEST_PROTOCOL)
        
        # Generate new version of summary figure (train & dev RMSEs)
        plot_sizes = range(1, k+1)
        plot_RMSE_train = [sfs_dict[size]['RMSE_train']/RMSE_train_initial for size in plot_sizes]
        plot_RMSE_dev = [sfs_dict[size]['RMSE_dev']/RMSE_dev_initial for size in plot_sizes]
        fig, axes = plt.subplots(figsize=(9,6))
        # Training data
        axes.scatter(x=plot_sizes, y=plot_RMSE_train, label='Training', color=dataset_colours['train'], alpha=0.5)
        axes.scatter(x=plot_sizes[np.argmin(plot_RMSE_train)], y=min(plot_RMSE_train), fc='none', ec='black')
        # Dev data
        axes.scatter(x=plot_sizes, y=plot_RMSE_dev, label='Validation', color=dataset_colours['dev'], alpha=0.5)
        axes.scatter(x=plot_sizes[np.argmin(plot_RMSE_dev)], y=min(plot_RMSE_dev), fc='none', ec='black')
        # Add the results achieved using the STRONG boruta subset
        if k >= len(features_confirmed_boruta):
            axes.scatter(x=len(features_confirmed_boruta), y=RMSE_train_boruta/RMSE_train_initial, marker='x', color=dataset_colours['train'])
            axes.scatter(x=len(features_confirmed_boruta), y=RMSE_dev_boruta/RMSE_dev_initial, marker='x', color=dataset_colours['dev'])
        # Add the results achieved using the STRONG boruta subset
        if k >= len(features_confirmed_boruta_weak):
            axes.scatter(x=len(features_confirmed_boruta_weak), y=RMSE_train_boruta_weak/RMSE_train_initial, marker='x', color=dataset_colours['train'])
            axes.scatter(x=len(features_confirmed_boruta_weak), y=RMSE_dev_boruta_weak/RMSE_dev_initial, marker='x', color=dataset_colours['dev'])
        # General figure properties
        axes.set_xlabel('Number of features selected')
        axes.set_ylabel('Root Mean Square Error [m]')
        axes.set_title('Full results for 100-tree Random Forests using different feature subsets')
        axes.grid(axis='y', which='major', alpha=0.3)
        axes.minorticks_on()
        axes.grid(axis='y', which='minor', alpha=0.08)
        axes.legend(frameon=False)
        fig.tight_layout()
        fig.savefig('{}/feature_selection/sffs_full_results.png'.format(folder_fig), dpi=300)
        plt.close()
        
        # Generate new version of summary figure (ONLY dev RMSEs)
        plot_sizes = range(1, k+1)
        plot_RMSE_dev = [sfs_dict[size]['RMSE_dev']/RMSE_dev_initial for size in plot_sizes]
        fig, axes = plt.subplots(figsize=(9,6))
        # Dev data
        axes.scatter(x=plot_sizes, y=plot_RMSE_dev, label='Validation', color=dataset_colours['dev'], alpha=0.5)
        axes.scatter(x=plot_sizes[np.argmin(plot_RMSE_dev)], y=min(plot_RMSE_dev), fc='none', ec='black')
        # Add the results achieved using the STRONG boruta subset
        if k >= len(features_confirmed_boruta):
            axes.scatter(x=len(features_confirmed_boruta), y=RMSE_dev_boruta/RMSE_dev_initial, marker='x', color=dataset_colours['dev'])
        # Add the results achieved using the STRONG boruta subset
        if k >= len(features_confirmed_boruta_weak):
            axes.scatter(x=len(features_confirmed_boruta_weak), y=RMSE_dev_boruta_weak/RMSE_dev_initial, marker='x', color=dataset_colours['dev'])
        # General figure properties
        axes.set_xlabel('Number of features selected')
        axes.set_ylabel('Root Mean Square Error [m]')
        axes.set_title('Validation results for 100-tree Random Forests using different feature subsets')
        axes.grid(axis='y', which='major', alpha=0.3)
        axes.minorticks_on()
        axes.grid(axis='y', which='minor', alpha=0.08)
        axes.legend(frameon=False)
        fig.tight_layout()
        fig.savefig('{}/feature_selection/sffs_full_results_dev.png'.format(folder_fig), dpi=300)
        plt.close()
        
        # Generate new version of summary figure (ONLY dev RMSEs & ONLY with 5+ features included)
        plot_sizes = range(8, k+1)
        plot_RMSE_dev = [sfs_dict[size]['RMSE_dev']/RMSE_dev_initial for size in plot_sizes]
        fig, axes = plt.subplots(figsize=(9,6))
        # Dev data
        axes.scatter(x=plot_sizes, y=plot_RMSE_dev, label='Validation', color=dataset_colours['dev'], alpha=0.5)
        axes.scatter(x=plot_sizes[np.argmin(plot_RMSE_dev)], y=min(plot_RMSE_dev), fc='none', ec='black')
        # Add the results achieved using the STRONG boruta subset
        if k >= len(features_confirmed_boruta):
            axes.scatter(x=len(features_confirmed_boruta), y=RMSE_dev_boruta/RMSE_dev_initial, marker='x', color=dataset_colours['dev'])
        # Add the results achieved using the STRONG boruta subset
        if k >= len(features_confirmed_boruta_weak):
            axes.scatter(x=len(features_confirmed_boruta_weak), y=RMSE_dev_boruta_weak/RMSE_dev_initial, marker='x', color=dataset_colours['dev'])
        # General figure properties
        axes.set_xlabel('Number of features selected')
        axes.set_ylabel('Root Mean Square Error [m]')
        axes.set_title('Validation results for 100-tree Random Forests using different feature subsets')
        axes.grid(axis='y', which='major', alpha=0.3)
        axes.minorticks_on()
        axes.grid(axis='y', which='minor', alpha=0.08)
        axes.legend(frameon=False)
        fig.tight_layout()
        fig.savefig('{}/feature_selection/sffs_full_results_dev_zoom.png'.format(folder_fig), dpi=300)
        plt.close()
    

# 2d: Finalise feature selection by inspecting results obtained for boruta & SFFS feature subsets
        
# Based on this comparison of train/dev performance for different k-features RFs (based on subsets found using SFFS and boruta), a
# minimal feature set seems to be the 15-feature subset found by SFFS, after which additional features allow only incremental improvements
# 15-feature subset: ['srtm_z','srtm_aspect','srtm_roughness','srtm_tpi','aw3d30_slope','aw3d30_tpi','ls7_b3','ls7_bsi','ls7_aweish','ls8_b3','ls8_ndmi','ls8_mndwi','gch','gfc','dmsp_avg_vis']
with open('{}/feature_selection_final_15_features.p'.format(folder_logs), 'wb') as f:
    pickle.dump(sfs_dict[15]['features_selected'], f, pickle.HIGHEST_PROTOCOL)

# Some further improvement is possible by allowing 23 features - worth testing this subset with the neural network models
# later, in case they're able to make better use of these additional features and extract larger gains from that information
# 23-feature subset: ['srtm_z','srtm_aspect','srtm_roughness','srtm_tpi','aw3d30_z','aw3d30_slope','aw3d30_tpi','aw3d30_tri','ls7_b1','ls7_b3','ls7_b4','ls7_bsi','ls7_mndwi','ls7_aweish','ls7_ndbi','ls8_b3','ls8_ndmi','ls8_mndwi','gch','gfc','ntl_viirs','dmsp_avg_vis','dmsp_pct_lights']
with open('{}/feature_selection_final_23_features.p'.format(folder_logs), 'wb') as f:
    pickle.dump(sfs_dict[23]['features_selected'], f, pickle.HIGHEST_PROTOCOL)
    
# Best score over trialled range involved 27 features - it might be worth testing this subset with the neural network models
# later, in case they're able to make better use of these additional features and extract larger gains from that information
# 27-feature subset: ['srtm_z','srtm_aspect','srtm_roughness','srtm_tpi','aw3d30_z','aw3d30_slope','aw3d30_tpi','aw3d30_tri','ls7_b1','ls7_b3','ls7_b4','ls7_b5','ls7_bsi','ls7_mndwi','ls7_aweish','ls7_ndbi','ls8_b1','ls8_b3','ls8_si','ls8_ndmi','ls8_mndwi','ls8_aweish','gch','gfc','ntl_viirs','dmsp_avg_vis','dmsp_pct_lights']
with open('{}/feature_selection_final_27_features.p'.format(folder_logs), 'wb') as f:
    pickle.dump(sfs_dict[27]['features_selected'], f, pickle.HIGHEST_PROTOCOL)


###############################################################################
# 3. Check impact of RF size on model performance (dev RMSE) & training times #
###############################################################################

# Retrieve the list of selected features, if it's not already in memory
features_selected = pickle.load(open('{}/feature_selection_final_15_features.p'.format(folder_logs), 'rb'))

# Define the range of forest sizes (as number of trees) to be trialled
rf_sizes = [5,10,25,50,100,250,500,1000]

# Initialise dictionary to hold the parameters to be tracked
rf_size_dict = {size:{'RMSE_train':None, 'RMSE_dev':None, 'run_time':None, 'feature_importances':None} for size in rf_sizes}

# Determine the maximum number of features allowed for each tree (using recommended default of p/3)
max_features = floor(len(features_selected)/3)

# Loop through the forest sizes to be tested
for rf_size in rf_sizes:
    
    # Instantiate RF model of the appropriate size
    rf = RandomForestRegressor(n_estimators=rf_size, max_features=max_features, random_state=1, n_jobs=-1)
    
    # Train the model, reporting how long it takes
    start_time = time.time()
    rf.fit(features_train[features_selected], target_train)
    print('\nRandom Forest of {} trees'.format(rf_size))
    run_time_mins = (time.time()-start_time)/60.
    print(' - Time taken to fit model: {:.1f} mins'.format(run_time_mins))
    rf_size_dict[rf_size]['run_time'] = run_time_mins
    
    # Evaluate model's performance based on the training data
    pred_train = rf.predict(features_train[features_selected])
    RMSE_train = np.sqrt(mean_squared_error(target_train, pred_train))
    rf_size_dict[rf_size]['RMSE_train'] = RMSE_train
    
    # Evaluate model's performance based on the dev data
    pred_dev = rf.predict(features_dev[features_selected])
    RMSE_dev = np.sqrt(mean_squared_error(target_dev, pred_dev))
    rf_size_dict[rf_size]['RMSE_dev'] = RMSE_dev
    print(' - Dev MAE: {:.3f}'.format(RMSE_dev))
    
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    
    # List of tuples with variable and importance
    feature_importances = [(feature, importance) for feature, importance in zip(features_selected, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse=True)
    rf_size_dict[rf_size]['feature_importances'] = feature_importances

# Save result dictionary to disk, in case it's needed later
with open('{}/rf_size_dict.p'.format(folder_logs), 'wb') as f:
    pickle.dump(rf_size_dict, f, pickle.HIGHEST_PROTOCOL)

# Figure: Show impact of random forest size (number of trees) on the various metrics tracked
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(9,9))
# Plot 1: Mean Absolute Errors
# Show training error results
axes[0].scatter(rf_sizes, [rf_size_dict[s]['RMSE_train'] for s in rf_sizes], c='blue', label='Train')
[axes[0].annotate('{:.3f}m'.format(rf_size_dict[s]['RMSE_train']), xy=(s, rf_size_dict[s]['RMSE_train']-0.1), c='blue', size=8, va='top', ha='center') for s in rf_sizes]
# Show dev error results
axes[0].scatter(rf_sizes, [rf_size_dict[s]['RMSE_dev'] for s in rf_sizes], c='red', label='Validation')
[axes[0].annotate('{:.3f}m'.format(rf_size_dict[s]['RMSE_dev']), xy=(s, rf_size_dict[s]['RMSE_dev']-0.1), c='red', size=8, va='top', ha='center') for s in rf_sizes]
axes[0].set_xscale('log')
axes[0].set_ylim(bottom=0)
axes[0].set_ylabel('Root Mean Square Error (RMSE)')
axes[0].grid(axis='y', alpha=0.25)
axes[0].legend(frameon=False, ncol=2, loc='lower left')
# Plot 2: Run times
axes[1].scatter(rf_sizes, [rf_size_dict[s]['run_time'] for s in rf_sizes], c='black')
axes[1].set_xlabel('Model size (number of trees)')
axes[1].set_ylabel('Simulation time [mins]')
axes[1].grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig('{}/impacts/impact_rf_size_on_RMSE.png'.format(folder_fig), dpi=300)
plt.close()


# Define a function to categorise features, returning the category name and colour
def feature_category(feature):
    if feature.startswith('srtm') or feature.startswith('aster') or feature.startswith('aw3d30'):
        return 'Topography', 'sienna'
    elif feature in ['gch','gfc','ls7_b3','ls7_evi','ls7_msavi','ls7_ndvi','ls7_savi','ls7_avi','ls8_b3','ls8_evi','ls8_msavi','ls8_ndvi','ls8_savi','ls8_avi']:
        return 'Vegetation', 'green'
    elif feature in ['ntl_viirs','dmsp_avg_vis','dmsp_avg_vis_stable','dmsp_pct_lights','dmsp_avg_lights_pct','osm_bld','osm_rds','osm_brd','ls7_ndbi','ls8_ndbi']:
        return 'Built-up', 'grey'
    elif feature in ['gsw_occurrence','ls7_mndwi','ls7_ndmi','ls7_aweinsh','ls7_aweish','ls8_mndwi','ls8_ndmi','ls8_aweinsh','ls8_aweish']:
        return 'Water', 'blue'
    elif feature in ['ls7_si','ls7_bsi','ls8_si','ls8_bsi']:
        return 'Other', 'orange'
    elif feature in ['ls7_b1','ls7_b2','ls7_b3','ls7_b4','ls7_b5','ls7_b6_vcid_1','ls7_b6_vcid_2','ls7_b7','ls7_b8','ls8_b1','ls8_b2','ls8_b3','ls8_b4','ls8_b5','ls8_b6','ls8_b7','ls8_b8','ls8_b9','ls8_b10','ls8_b11']:
        return 'Spectral band', 'purple'


# Figure: Show impact of random forest size (number of trees) on feature importances (looking at top & bottom five for each)
fig, axes = plt.subplots(nrows=2, ncols=int(len(rf_sizes)/2), figsize=(15,9))
for i, ax in enumerate(axes.reshape(-1)):
    FIs = rf_size_dict[rf_size]['feature_importances']
    FI_vals = [val for (feature,val) in FIs]
    FI_labels = [feature for (feature,val) in FIs]
    FI_colours = [feature_category(feature)[1] for feature in FI_labels]
    ax.barh(range(len(FIs)), FI_vals, color=FI_colours, alpha=0.5, align='center')
    ax.set_yticks(range(len(FIs)))
    ax.set_yticklabels(FI_labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title('{}-tree RF'.format(rf_sizes[i]))
fig.tight_layout()
fig.savefig('{}/impacts/impact_rf_size_on_feature_importances.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 4. Re-import input data, filtering so that only selected features included  #
###############################################################################

# Read the list of selected features back into memory
with open('{}/feature_selection_final_23_features.p'.format(folder_logs), 'rb') as f:
    selected_features = pickle.load(f)

# Import processed CSVs into dataframes
data_train = pd.read_csv('{}/Input1D_Train.csv'.format(folder_input_1D))
data_dev = pd.read_csv('{}/Input1D_Dev.csv'.format(folder_input_1D))

# Filter the dataframes available (train & dev) to contain only target & the selected feature columns
target_train = data_train['diff'].copy()
target_dev = data_dev['diff'].copy()
features_train = data_train[selected_features].copy()
features_dev = data_dev[selected_features].copy()


###############################################################################
# 5. Use Optuna to tune Random Forest hyperparameters                         #
###############################################################################

# Get number of features used
n_features = len(selected_features)

# Run a series of hyperparameter tuning trials, for a range of Random Forest sizes (10, 25, 100, 250 trees)
n_trees = 250        # All values trialled: 10, 25, 100, 250
n_trials = 50       # For realistic run times: 10-tree (500), 25-tree (500), 100-tree (250), 250-tree (100)

# Define an objective function to be minimised
def objective(trial):
    # Suggest values for each of the hyperparameters to be explored
    min_samples_split = trial.suggest_int('min_samples_split', 2, 4)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)
    max_features = trial.suggest_int('max_features', 6, 15)
    # Initialise a Random Forest regressor model with the input hyperparameters
    rf_opt = RandomForestRegressor(n_estimators=n_trees, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
    # Fit the model to the training data
    rf_opt.fit(features_train, target_train)
    # Get the dev/validation RMSE
    pred_dev = rf_opt.predict(features_dev)
    RMSE_dev = np.sqrt(mean_squared_error(target_dev, pred_dev))
    return RMSE_dev

# Create a study object and optimise the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials)

# Get the best set of parameters found
study.best_params
#  10 trees (500 trials): {'min_samples_split': 12, 'min_samples_leaf': 5, 'max_features': 13} - 4.124659893304994
#  25 trees (500 trials): {'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 10} - 4.059818326303117
# 100 trees (250 trials): {'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 7} - 4.011363397414606
# 250 trees (100 trials): {'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 6} - 4.002210770610134

# The tuned hyperparameters (for 15 features) were found to vary somewhat predictably based on number of trees/estimators
#  - min_samples_split = 19.901 - 3.298 * ln(n_trees)
#  - min_samples_leaf = 8.175 - 1.387 * ln(n_trees)
#  - max_features = 17.501 - 2.173 * ln(n_trees)

# Create a dataframe from the study results
df = study.trials_dataframe()

# Export dataframe for later reference
df.to_csv('{}/hparam_optuna_{}trees.csv'.format(folder_logs, str(n_trees).zfill(3)))

# Visualise distribution of sampled hyperparameter values
idx_50, idx_75, idx_90 = [int(fraction * len(df.index)) for fraction in [0.5, 0.75, 0.9]]
fig, axes = plt.subplots(nrows=3, figsize=(9,9))
# Loop through the four hyperparameters, adding two plots for each to the figure
for i, hparam in enumerate(['min_samples_split','min_samples_leaf','max_features']):
    col = 'params_{}'.format(hparam)
    # Calculate bin edges based on all parameter values investigated
    bins = [edge - 0.5 for edge in range(df[col].min(), df[col].max()+2, 1)]
    # Get subsets based on position in full sequence of guesses
    val_0_75 = df[col].iloc[:idx_75]
    val_75_90 = df[col].iloc[idx_75:idx_90]
    val_90_100 = df[col].iloc[idx_90:]
    # Add histogram of all guesses, colour-coded by phase of guessing
    axes[i].hist([val_90_100, val_75_90, val_0_75], bins=bins, histtype='barstacked', color=['mediumblue','royalblue','lightsteelblue'], label=['Final','Mid','First'])
    # Add vertical line showing best guess
    axes[i].axvline(x=study.best_params[hparam], color='red', linestyle='dashed')
    axes[i].legend(frameon=False)
    axes[i].set_title(hparam)
fig.suptitle('Using a {}-tree Random Forest ({} trials)'.format(n_trees, len(df.index)))
fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.savefig('{}/hyperparameter_tuning/hparam_optuna_{}trees.png'.format(folder_fig, str(n_trees).zfill(3)), dpi=300)
plt.close()


###############################################################################
# 6. Train 250-tree Random Forest Regressor using tuned hyperparameters       #
###############################################################################

# Read the list of selected features back into memory, if necessary
with open('{}/feature_selection_final_23_features.p'.format(folder_logs), 'rb') as f:
    selected_features = pickle.load(f)

# Import processed CSVs into dataframes
data_train = pd.read_csv('{}/Input1D_Train.csv'.format(folder_input_1D))
data_dev = pd.read_csv('{}/Input1D_Dev.csv'.format(folder_input_1D))

# Filter the dataframes available (train & dev) to contain only target & the selected feature columns
target_train = data_train['diff'].copy()
target_dev = data_dev['diff'].copy()
features_train = data_train[selected_features].copy()
features_dev = data_dev[selected_features].copy()

# Define the tuned hyperparameters to be used in the final model
n_trees = 250
min_samples_split = 2
min_samples_leaf = 1
max_features = 11

# Instantiate RF model of the appropriate size (using tuned hyperparameters)
rf = RandomForestRegressor(n_estimators=n_trees, random_state=1, min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)

# Train the model on all available training data
rf.fit(features_train, target_train)

# Evaluate model's performance based on the training data (using RMSE)
pred_train = rf.predict(features_train)
RMSE_train = np.sqrt(mean_squared_error(target_train, pred_train))             # 1.237401440168071

# Evaluate model's performance based on the dev data (using RMSE)
pred_dev = rf.predict(features_dev)
RMSE_dev = np.sqrt(mean_squared_error(target_dev, pred_dev))                   # 3.9710284505097424


# Using 15 features (max_features=6):  RMSE results are 1.258 (train), 4.002 (dev)

#   Using 23 features (max_features=6):  RMSE results are 1.214 (train), 3.996 (dev)
#   Using 23 features (max_features=9):  RMSE results are 1.229 (train), 3.983 (dev)
#   Using 23 features (max_features=10): RMSE results are 1.233 (train), 3.978 (dev)
# x Using 23 features (max_features=11): RMSE results are 1.237 (train), 3.971 (dev)
#   Using 23 features (max_features=12): RMSE results are 1.240 (train), 3.973 (dev)
#   Using 23 features (max_features=13): RMSE results are 1.243 (train), 3.973 (dev)
#   Using 23 features (max_features=14): RMSE results are 1.246 (train), 3.975 (dev)

# Using 27 features (max_features=10): RMSE results are 1.224 (train), 3.965 (dev)
# Using 27 features (max_features=11): RMSE results are 1.228 (train), 3.961 (dev)
# Using 27 features (max_features=12): RMSE results are 1.231 (train), 3.963 (dev)

# Save model to disk
joblib.dump(rf, '{}/rf_23features_250trees.sav'.format(folder_models), compress=3)


###############################################################################
# 7. Visualise results obtained using the Random Forest model trained above   #
###############################################################################

# If it's not already in memory, load the RF model trained above
rf = joblib.load('{}/rf_23features_250trees.sav'.format(folder_models))

# Import & filter the test data (with the train & dev set imported in previous step)
data_test = pd.read_csv('{}/Input1D_Test.csv'.format(folder_input_1D))
target_test = data_test['diff'].copy()
features_test = data_test[selected_features].copy()

# Initialise a dataframe to hold performance metrics
df = pd.DataFrame(columns=['RMSE','Dataset','Type'])

# Calculate baseline predictions (mean of corrections observed from the training data)
baseline_predictions = np.mean(target_train)

# Build a dictionary to hold residuals for each dataset
residuals = {dataset:{model:None for model in ['initial','baseline']} for dataset in ['train','dev','test']}

# Loop through each dataset & record the RF's performance on that data
for (dataset, features, target) in zip(['train','dev','test'], [features_train, features_dev, features_test], [target_train, target_dev, target_test]):
    print('Processing {} dataset...'.format(dataset))
    # Calculate the initial RMSE, based on target data
    RMSE_initial = np.sqrt(mean_squared_error(target, np.zeros(len(target))))
    df = df.append({'RMSE':RMSE_initial,'Dataset':dataset,'Type':'initial'}, ignore_index=True)
    # Calculate RMSE after applying baseline correction (subtracting mean difference observed from training data)
    RMSE_baseline = np.sqrt(mean_squared_error(target, baseline_predictions*np.ones(len(target))))
    df = df.append({'RMSE':RMSE_baseline,'Dataset':dataset,'Type':'baseline'}, ignore_index=True)
    # Save residual arrays in the dictionary initialised previously
    residuals[dataset]['initial'] = target.to_list()
    residuals[dataset]['baseline'] = (target - baseline_predictions).to_list()
    # RF model: Determine RMSE after applying predicted corrections & save residuals
    predictions = rf.predict(features)
    RMSE_rf = np.sqrt(mean_squared_error(target, predictions))
    df = df.append({'RMSE':RMSE_rf,'Dataset':dataset,'Type':'rf'}, ignore_index=True)
    residuals[dataset]['rf'] = (target - predictions).to_list()

# Save these results to CSV files, for easy import & comparison later
df.to_csv('{}/rf_23features_250trees_metrics_by_dataset.csv'.format(folder_results))
pickle.dump(residuals, open('{}/rf_residuals.p'.format(folder_results), 'wb'))

# Generate summary plots for each of the three dataset results
for i, dataset in enumerate(['train','dev','test']):
    # Extract appropriate results from the dataframe
    RMSE_initial = df[(df['Dataset']==dataset)&(df['Type']=='initial')]['RMSE'].values.item()
    RMSE_baseline = df[(df['Dataset']==dataset)&(df['Type']=='baseline')]['RMSE'].values.item()
    RMSE_model = df[(df['Dataset']==dataset)&(df['Type']=='rf')]['RMSE'].values.item()
    # Set up the figure
    fig, axes = plt.subplots(figsize=(9,4.5))
    axes.bar([0,1,2], [RMSE_initial, RMSE_baseline, RMSE_model], color=dataset_colours[dataset], alpha=0.5)
    axes.set_xticks([0,1,2])
    axes.yaxis.set_tick_params(length=0)
    axes.set_xticklabels(['Initial','After baseline\ncorrection','Random Forest\ncorrection'])
    axes.set_ylabel('Root Mean Square Error [m]')
    axes.grid(axis='y', which='major', color='dimgrey', alpha=0.2)
    [axes.spines[edge].set_visible(False) for edge in ['left','top','right']]
    # Add a horizontal line showing the initial error & a label
    axes.axhline(y=RMSE_initial, color=dataset_colours[dataset], linestyle='dashed', alpha=0.5)
    axes.annotate('{:.3f}m'.format(RMSE_initial), xy=(0, RMSE_initial), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
    # Add labels indicating improvement achieved by each method
    for j, RMSE_new in enumerate([RMSE_baseline, RMSE_model]):
        # Add downward arrow from initial RMSE to improved RMSE
        axes.annotate('', xy=(j+1, RMSE_new), xytext=(j+1, RMSE_initial), arrowprops=dict(arrowstyle='->'))
        # Add label indicating new RMSE and the percentage improvement it equates to
        improvement_percentage = (RMSE_new-RMSE_initial)/RMSE_initial * 100.
        axes.annotate('{:.3f}m ({:.1f}%)'.format(RMSE_new, improvement_percentage), xy=(j+1, RMSE_new), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
    axes.set_title('Performance on {} dataset'.format('validation' if dataset=='dev' else dataset))
    fig.tight_layout()
    fig.savefig('{}/results/rf_23features_250trees_results_{}.png'.format(folder_fig, dataset), dpi=300)
    plt.close()


###############################################################################
# 8. Generate 'intact' prediction vectors for all zones, for mapping/GeoTIFFs #
###############################################################################

# If it's not already in memory, load the RF model trained above
rf = joblib.load('{}/rf_23features_250trees.sav'.format(folder_models))

# Read the list of selected features back into memory
with open('{}/feature_selection_final_23_features.p'.format(folder_logs), 'rb') as f:
    selected_features = pickle.load(f)

# Initialise a df to hold RMSE metrics for each DTM zone
df_metrics = pd.DataFrame(columns=['zone','RMSE'])

# Loop through all zones, import data, generate prediction vector & save as .npy file
for zone in zones:
    
    # Import the 'intact' versions of the data for that zone
    df = pd.read_csv('{}/Input1D_ByZone_{}.csv'.format(folder_input_1D, zone))
    
    # Evaluate RF performance on zone, based on filtered dataframe in which rows with nodata are dropped
    df_valid = df.replace(no_data, np.nan).dropna(axis=0, how='any', inplace=False)
    target_valid = df_valid['diff'].copy()
    features_valid = df_valid[selected_features].copy()
    predictions_valid = rf.predict(features_valid)
    RMSE = np.sqrt(mean_squared_error(target_valid, predictions_valid))
    print('{}: RMSE = {:.3f} m'.format(zone, RMSE))
    df_metrics.append({'zone':zone,'RMSE':RMSE}, ignore_index=True)
    
    # Generate full results for mapping, without filtering out nodata rows
    target = df['diff'].copy()
    features = df[selected_features].copy()
    predictions = rf.predict(features)
    np.save('{}/RF_Predictions_ByZone_{}.npy'.format(folder_results, zone), predictions)

# Save these results to CSV files, for easy import & comparison later
df_metrics.to_csv('{}/rf_23features_250trees_metrics_by_zone.csv'.format(folder_results))