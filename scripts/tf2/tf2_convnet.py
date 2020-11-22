# Predictive model: Convolutional neural network (convnet/FCN)

# Import required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import pickle
import joblib
from random import random
import os
from itertools import combinations

# Import TensorFlow modules & submodules
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Cropping2D, GaussianDropout, Input, MaxPooling2D
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define paths to relevant folders
folder_input_2D = 'E:/mdm123/D/ML/inputs/2D'
folder_logs_2D = 'E:/mdm123/D/ML/logs/convnet'
folder_logs_rf = 'E:/mdm123/D/ML/logs/rf'
folder_models = 'E:/mdm123/D/ML/models/convnet'
folder_fig = 'E:/mdm123/D/figures/models/convnet'
folder_results = 'E:/mdm123/D/ML/results/convnet'

# Define a dictionary of colours for each dataset
dataset_colours = {'train':'blue', 'dev':'green', 'test':'firebrick'}

# Preliminary step (before GPU initialised) to resolve Out of Memory issues later: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


###############################################################################
# 1. Import & pre-process data (2D patches)                                   #
###############################################################################

# Read the list of selected features into memory
with open('{}/feature_selection_final_23_features.p'.format(folder_logs_rf), 'rb') as f:
    selected_features = pickle.load(f)

# Read the ordered list of feature names into memory (ordered as they were added to the 2D arrays)
with open('{}/feature_names_ordered_all.p'.format(folder_input_2D), 'rb') as f:
    feature_names_ordered = pickle.load(f)

# Get a list of indices corresponding to the selected features
selected_features_idx = []
for i, feature_name in enumerate(feature_names_ordered):
    if feature_name in selected_features:
        print('{} [{}]'.format(feature_name, i))
        selected_features_idx.append(i)

# Define a function to import & process input data into format appropriate for the convnet
def process_input_data(label, selected_features_idx, no_data=-9999):
    # Open the two numpy arrays relating to the specified label (train/dev/test)
    features = np.load('{}/Input2D_Features_{}.npy'.format(folder_input_2D, label.capitalize())) # Shape: (patches, height, width, channels)
    target = np.load('{}/Input2D_Target_{}.npy'.format(folder_input_2D, label.capitalize()))     # Shape: (patches, height, width, channels)
    # Filter the features so that only those previously selected are retained
    features = features[:,:,:,selected_features_idx]
    # Get the number of patches available
    n_patches = target.shape[0]
    # Initialise two lists to hold the filtered arrays
    features_valid = []
    target_valid = []
    # Loop through all patches, rejecting any for which the target array contains a no_data value
    for n in range(n_patches):
        if np.any(target[n,:,:,:]==no_data) or np.any(np.isnan(target[n,:,:,:])):
            pass
        else:
            target_valid.append(target[n,:,:,:])
            features_valid.append(features[n,:,:,:])
    # Convert filter lists into numpy arrays
    features_out = np.array(features_valid)  # Shape: (patches, height, width, channels)
    target_out = np.array(target_valid)      # Shape: (patches, height, width, channels)
    # Print update
    print('\n{} data processed:\n - Before: features = {}, target = {}\n - After: features = {}, target = {}'.format(label.capitalize(), features.shape, target.shape, features_out.shape, target_out.shape))
    return features_out, target_out

# Use function defined above to import training & dev data in a convnet-ready format
features_train, target_train = process_input_data('train', selected_features_idx)
features_dev, target_dev = process_input_data('dev', selected_features_idx)

# Training:
# - Before: features = (14058, 100, 100, 23), target = (14058, 12, 12, 1)
# - After: features = (11611, 100, 100, 23), target = (11611, 12, 12, 1)
# Validation:
# - Before: features = (697, 100, 100, 23), target = (697, 12, 12, 1)
# - After: features = (697, 100, 100, 23), target = (697, 12, 12, 1)

# Calculate feature-wise means & standard deviations using only the training data
features_train_mean = np.mean(features_train, axis=(0,1,2), keepdims=True)
features_train_std = np.std(features_train, axis=(0,1,2), keepdims=True)

# Use these to normalise the three features datasets
features_train_norm = (features_train - features_train_mean)/features_train_std
features_dev_norm = (features_dev - features_train_mean)/features_train_std

# Confirm that this has worked as expected - for feature data (choose random patch)
patch_idx = 10
for i, feature in enumerate(selected_features):
    fig, axes = plt.subplots(ncols=2, figsize=(9,5))
    axes[0].imshow(features_train[patch_idx,:,:,i])
    axes[0].set_title('{} (Original)'.format(feature))
    axes[0].annotate('{:.2f} - {:.2f}'.format(features_train[patch_idx,:,:,i].min(), features_train[patch_idx,:,:,i].max()), xy=(0.5,0.5), xycoords='axes fraction', ha='center', va='center', color='white')
    axes[1].imshow(features_train_norm[10,:,:,i])
    axes[1].set_title('{} (Normalised)'.format(feature))
    axes[1].annotate('{:.2f} - {:.2f}'.format(features_train_norm[patch_idx,:,:,i].min(), features_train_norm[patch_idx,:,:,i].max()), xy=(0.5,0.5), xycoords='axes fraction', ha='center', va='center', color='white')
    fig.tight_layout()
    fig.savefig('{}/normalisation/check_normalisation_feature_{}_{}.png'.format(folder_fig, i, feature))
    plt.close()

# Remove original (unnormalised) training & validation feature datasets to free up memory
del features_train, features_dev


###############################################################################
# 2. Helper functions for hyperparameter tuning: ongoing result visualisation #
###############################################################################

# Define a function that takes a dataframe & a hyperparameter label, and returns a list of Series corresponding to the quintiles
def get_quintiles(df, hparam):
    # Get the column name corresponding to the selected hyperparameter
    col = 'params_{}'.format(hparam)
    # Get index values needed to define quintiles
    idx_20, idx_40, idx_60, idx_80 = [int(fraction * len(df.index)) for fraction in [0.2, 0.4, 0.6, 0.8]]
    # Get four series based on the slice indices defined above, with the later trials first
    return df[col].iloc[idx_80:], df[col].iloc[idx_60:idx_80], df[col].iloc[idx_40:idx_60], df[col].iloc[idx_20:idx_40], df[col].iloc[:idx_20]

# Define colours to be used for each quintile (last to first)
sequence_colours = ['midnightblue', 'royalblue','cornflowerblue','lightsteelblue','gainsboro']
sequence_labels = ['Q5', 'Q4','Q3','Q2','Q1']

# Define a plotting function to visualise the Round 1 hyperparameter tuning results
def visualise_hparam_tuning_round1(df, best, tuning_round):
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,12))
    # 1: Title
    axes[0,0].annotate('Round {}\n({} trials)'.format(tuning_round, len(df.index)), xy=(0.5, 0.5), ha='center', va='center', size=16, color='dimgrey', weight='bold')
    axes[0,0].axis('off')
    # 2: Best RMSE so far
    axes[0,1].annotate('Best RMSE: {:.4f}'.format(df['value'].min()), xy=(0.5, 0.5), ha='center', va='center', size=12, color='dimgrey', weight='bold')
    axes[0,1].axis('off')
    # 3: n_filters_first
    hp = 'n_filters_first'
    quintiles = get_quintiles(df, hp)
    axes[0,2].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,2].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,2].set_title(hp)
    [axes[0,2].spines[edge].set_visible(False) for edge in ['top','right']]
    # 4: n_filters_growthrate
    hp = 'n_filters_growthrate'
    quintiles = get_quintiles(df, hp)
    axes[0,3].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,3].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,3].set_title(hp)
    [axes[0,3].spines[edge].set_visible(False) for edge in ['top','right']]
    # 5: learning_rate
    hp = 'learning_rate'
    quintiles = get_quintiles(df, hp)
    axes[0,4].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,4].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,4].set_title(hp)
    [axes[0,4].spines[edge].set_visible(False) for edge in ['top','right']]
    # 6: batch_size
    hp = 'batch_size'
    quintiles = get_quintiles(df, hp)
    axes[0,5].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,5].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,5].set_title(hp)
    [axes[0,5].spines[edge].set_visible(False) for edge in ['top','right']]
    # 7-12: dropout_1 to dropout_7
    for i,l in enumerate(['1','2','3','5','6','7']):
        hp = 'dropout_{}'.format(l)
        quintiles = get_quintiles(df, hp)
        axes[1,i].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
        axes[1,i].axvline(x=best[hp], linestyle='dashed', color='red')
        axes[1,i].set_title(hp)
        [axes[1,i].spines[edge].set_visible(False) for edge in ['top','right']]
    # General figure properties
    fig.tight_layout()
    fig.savefig('{}/hyperparameter_tuning/convnet_tuning_round{}.png'.format(folder_fig, tuning_round), dpi=300)
    plt.close()

# Define a plotting function to visualise the Round 2 hyperparameter tuning results
def visualise_hparam_tuning_round2(df, best, dropout, tuning_round):
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18,9))
    # 1: Title & best score
    axes[0,0].annotate('Round {}\n({} trials)'.format(tuning_round, len(df.index)), xy=(0.5, 0.6), ha='center', va='center', size=18, color='dimgrey', weight='bold')
    axes[0,0].annotate('Best RMSE:\n{:.4f}'.format(df['value'].min()), xy=(0.5, 0.4), ha='center', va='center', size=14, color='dimgrey', weight='bold')
    axes[0,0].axis('off')
    # 2: RMSE values achieved
    axes[0,1].hist(df['value'], bins=10)
    #axes[0,1].axvline(x=df['value'].min(), linestyle='dashed', color='red')
    axes[0,1].set_title('Validation RMSE [m]')
    [axes[0,1].spines[edge].set_visible(False) for edge in ['top','right']]
    # 3: n_filters_first
    hp = 'n_filters_first'
    quintiles = get_quintiles(df, hp)
    axes[0,2].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,2].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,2].set_title(hp)
    [axes[0,2].spines[edge].set_visible(False) for edge in ['top','right']]
    # 4: n_filters_growthrate
    hp = 'n_filters_growthrate'
    quintiles = get_quintiles(df, hp)
    axes[0,3].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,3].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,3].set_title(hp)
    [axes[0,3].spines[edge].set_visible(False) for edge in ['top','right']]
    # 5: learning_rate
    hp = 'learning_rate'
    quintiles = get_quintiles(df, hp)
    axes[0,4].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,4].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,4].set_title(hp)
    [axes[0,4].spines[edge].set_visible(False) for edge in ['top','right']]
    # 6: batch_size
    hp = 'batch_size'
    quintiles = get_quintiles(df, hp)
    axes[0,5].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,5].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,5].set_title(hp)
    [axes[0,5].spines[edge].set_visible(False) for edge in ['top','right']]
    # 7-12: dropout_1 to dropout_7
    for i,l in enumerate(['1','2','3','5','6','7']):
        hp = 'dropout_{}'.format(l)
        quintiles = get_quintiles(df, hp)
        axes[1,i].hist(quintiles, bins=10, histtype='barstacked', color=sequence_colours, label=sequence_labels)
        axes[1,i].axvline(x=best[hp], linestyle='dashed', color='red')
        axes[1,i].set_title(hp)
        [axes[1,i].spines[edge].set_visible(False) for edge in ['top','right']]
    # General figure properties
    fig.tight_layout()
    fig.savefig('{}/hyperparameter_tuning/convnet_tuning_round{}.png'.format(folder_fig, tuning_round), dpi=300)
    plt.close()


###############################################################################
# 3. Define functions to build a modified U-net model                         #
###############################################################################

# Use the tf.keras loss function to compute the root mean squared error later
RMSE = RootMeanSquaredError()

# Define a helper function to do the 2D convolution processing
def conv2d_proc(input_tensor, n_filters, kernel_initialiser, activation, kernel_size=(3,3)):
    # Set up the first layer
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer=kernel_initialiser, padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # Set up the second layer
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer=kernel_initialiser, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

# Define a function to build a modified U-net model
def build_unet_model(n_filters_first, n_filters_growthrate, dropout_rates, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23)):
    # n_filters_first (int): Number of filters/channels to be used in first step
    # n_filters_growthrate (float): Factor by which number of filters should increase (in descending arm) and then increase (in ascending arm)
    # dropout_rates (list or False): Dropout rates for each descending & ascending step (either six or seven, for this modified U-net)
    # kernel_initialiser (string): String representing kernel/weight initialisation method to be used (e.g. 'he_normal', 'he_uniform')
    # activation (string): String representing activation method to be used (e.g. 'relu', 'elu', 'selu')
    # optimiser (string): String representing optimiser to be used ('Adam' or 'Nadam')
    # learning_rate (float): Learning rate used for optimisation
    # input_size (tuple): Shape of input feature data - tuple of 3 ints: (patch_height, patch_width, n_channels). The default is (100,100,23).
    
    # Define optimiser based on input string
    if optimiser == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimiser == 'Nadam':
        opt = Nadam(learning_rate=learning_rate)
    elif optimiser == 'SGD':
        opt = SGD(learning_rate=learning_rate)
    
    # Decompose input list of dropout rates into individual values
    if dropout_rates:
        if len(dropout_rates) == 6:
            dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7 = dropout_rates
            dropout_4 = False
        elif len(dropout_rates) == 7:
            dropout_1, dropout_2, dropout_3, dropout_4, dropout_5, dropout_6, dropout_7 = dropout_rates
    
    # Initialise variable which will define the number of filters to use at each step (continually updated)
    n_filters = n_filters_first
    
    # Define an Input of the expected size
    inputs = Input(input_size)
    
    # Contracting: Step 1
    c1 = conv2d_proc(inputs, n_filters, kernel_initialiser, activation, kernel_size=(3,3))
    p1 = MaxPooling2D((2,2))(c1)
    if dropout_rates: p1 = GaussianDropout(dropout_1)(p1)
    
    # Contracting: Step 2
    n_filters = int(round(n_filters * n_filters_growthrate))
    c2 = conv2d_proc(p1, n_filters, kernel_initialiser, activation, kernel_size=(3,3))
    p2 = MaxPooling2D((2,2))(c2)
    if dropout_rates: p2 = GaussianDropout(dropout_2)(p2)
    
    # Contracting: Step 3
    n_filters = int(round(n_filters * n_filters_growthrate))
    c3 = conv2d_proc(p2, n_filters, kernel_initialiser, activation, kernel_size=(3,3))
    p3 = MaxPooling2D((2,2))(c3)
    if dropout_rates: p3 = GaussianDropout(dropout_3)(p3)
    
    # Bottom of U-Net
    n_filters = int(round(n_filters * n_filters_growthrate))
    c4 = conv2d_proc(p3, n_filters, kernel_initialiser, activation, kernel_size=(3,3))
    if dropout_4: c4 = GaussianDropout(dropout_4)(c4)
    
    # Expanding: Step 1
    n_filters = int(round(n_filters / n_filters_growthrate))
    u5 = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(c4)
    c3_crop = Cropping2D(cropping=((4,4),(4,4)))(c3)
    u5 = concatenate([u5, c3_crop])
    if dropout_rates: u5 = GaussianDropout(dropout_5)(u5)
    c5 = conv2d_proc(u5, n_filters, kernel_initialiser, activation, kernel_size=(3,3))
    
    # Expanding: Step 2
    n_filters = int(round(n_filters / n_filters_growthrate))
    u6 = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(c5)
    c2_crop = Cropping2D(cropping=((16,16),(16,16)))(c2)
    u6 = concatenate([u6, c2_crop])
    if dropout_rates: u6 = GaussianDropout(dropout_6)(u6)
    c6 = conv2d_proc(u6, n_filters, kernel_initialiser, activation, kernel_size=(3,3))
    
    # Expanding: Step 3
    n_filters = int(round(n_filters / n_filters_growthrate))
    u7 = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(c6)
    c1_crop = Cropping2D(cropping=((40,40),(40,40)))(c1)
    u7 = concatenate([u7, c1_crop])
    if dropout_rates: u7 = GaussianDropout(dropout_7)(u7)
    c7 = conv2d_proc(u7, n_filters, kernel_initialiser, activation, kernel_size=(3,3))
    
    # Output is defined slightly differently to the other conv2d processing steps (no activation or batch normalisation in second step)
    outputs = Conv2D(filters=1, kernel_size=(3,3), kernel_initializer=kernel_initialiser, padding='same')(c7)
    
    # Build model using the inputs & outputs defined
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # Compile model
    model.compile(optimizer=opt, loss='mse', metrics=[RMSE])
    return model


###############################################################################
# 4. First round of hyperparameter tuning by Bayesian optimisation (Optuna)   #
###############################################################################

# Define an early stopping callback
patience = 50
early_stop = EarlyStopping(monitor='val_loss', patience=patience)

# Define an objective function to be minimised
def round1_objective(trial):
    
    # Sample hyperparameters to be used throughout the model
    n_filters_first = trial.suggest_int('n_filters_first', 8, 64)
    n_filters_growthrate = trial.suggest_float('n_filters_growthrate', 1.0, 3.0)
    dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.5)
    dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.5)
    dropout_3 = trial.suggest_float('dropout_3', 0.0, 0.5)
    dropout_5 = trial.suggest_float('dropout_5', 0.0, 0.5)
    dropout_6 = trial.suggest_float('dropout_6', 0.0, 0.5)
    dropout_7 = trial.suggest_float('dropout_7', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    batch_size = int(trial.suggest_loguniform('batch_size', 8, 64))
    
    # Define the fixed hyperparameters to be used
    kernel_initialiser = 'he_uniform'
    activation = 'relu'
    optimiser = 'Adam'
    dropout_rates = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    
    # Call function defined earlier to build a modified U-net model based on these sampled parameters
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout_rates, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    
    # Train the model
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=750, validation_data=(features_dev_norm, target_dev), callbacks=[early_stop], shuffle=True, verbose=0)

    # Convert training history to df for easier processing
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    
    # Use the lower quartile (Q1, 25th percentile) of validation RMSE as the metric used by hyperparameter tuning process
    min_RMSE = np.min(history_df['val_root_mean_squared_error'])
    model_RMSE = np.percentile(history_df['val_root_mean_squared_error'].iloc[-patience:], 25)
    print('\nStopped after Epoch {}: Min RMSE = {:.3f}m, Q1 RMSE = {:.3f}\n'.format(len(history_df.index), min_RMSE, model_RMSE))
    return model_RMSE


# Round 1 of hyperparameter tuning
round1_count = 0
round1_increment = 5
round1_target = 125

# Create a study object for Round 1 - or reload previous progress to continue running
round1_study = optuna.create_study(direction='minimize')
#round1_study = joblib.load('{}/hparam_optuna_round1_study.pkl'.format(folder_logs_2D))

# Run for 125 trials in total, saving results every 5 trials
while round1_count < round1_target:
    
    # Run a new set of trials on the study object defined above
    round1_study.optimize(round1_objective, n_trials=round1_increment)
    
    # Get a dict of the best-performing hyperparameters
    best = round1_study.best_params
    
    # Create a dataframe from the study results so far
    df = round1_study.trials_dataframe()
    
    # Limit dataframe to only contain successful trials
    df = df.loc[df['state']=='COMPLETE'].copy()
    
    # Generate visualisations of the results so far
    visualise_hparam_tuning_round1(df, best, tuning_round=1)
    
    # Save the optuna study object, dictionary of best-performing parameters, and dataframe of results so far
    joblib.dump(round1_study, '{}/hparam_optuna_round1_study.pkl'.format(folder_logs_2D))
    joblib.dump(best, '{}/hparam_optuna_round1_best.pkl'.format(folder_logs_2D))
    df.to_csv('{}/hparam_optuna_round1_df.csv'.format(folder_logs_2D))
    
    # Update the study count (based on number of successful trials) & print status update
    round1_count = len(df.index)
    print('\n\nRound 1: {:,} trials completed (out of target {:,})\n\n'.format(round1_count, round1_target))


###############################################################################
# 5. Second round of hyperparameter tuning by Bayesian optimisation (Optuna)  #
###############################################################################

# Define an early stopping callback
patience = 20
early_stop = EarlyStopping(monitor='val_loss', patience=patience)

# Note: adjusted minimisation objective from 25th percentile to mean, after 98 trials

# Define an objective function to be minimised
def round2_objective(trial):
    
    # Define fixed hyperparameters to be used
    kernel_initialiser = 'he_normal'
    activation = 'elu'
    optimiser = 'Nadam'
    
    # Sample hyperparameters to be used throughout the model
    n_filters_first = trial.suggest_int('n_filters_first', 32, 64)
    n_filters_growthrate = trial.suggest_float('n_filters_growthrate', 2.0, 3.0)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.5)
    dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.5)
    dropout_3 = trial.suggest_float('dropout_3', 0.0, 0.5)
    dropout_5 = trial.suggest_float('dropout_5', 0.0, 0.5)
    dropout_6 = trial.suggest_float('dropout_6', 0.0, 0.5)
    dropout_7 = trial.suggest_float('dropout_7', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    # Call function defined earlier to build a modified U-net model based on these sampled parameters
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    
    # Train the model
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=[early_stop], shuffle=True, verbose=0)

    # Convert training history to df for easier processing
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    
    # Use the lower quartile (Q1, 25th percentile) of validation RMSE (since minimum reached) as the metric used by hyperparameter tuning process
    min_RMSE = np.min(history_df['val_root_mean_squared_error'])
    mean_RMSE = history_df['val_root_mean_squared_error'].iloc[-patience:].mean()
    print('\nStopped after Epoch {}: Min RMSE = {:.3f}m, Mean RMSE = {:.3f}\n'.format(len(history_df.index), min_RMSE, mean_RMSE))
    del model, history, history_df
    return mean_RMSE


# Round 2 of hyperparameter tuning
round2_count = 0
round2_increment = 2
round2_target = 250

# Create a study object for Round 2 - or reload previous progress to continue running
#round2_study = optuna.create_study(direction='minimize')
round2_study = joblib.load('{}/hparam_optuna_round2_study.pkl'.format(folder_logs_2D))

# Continue running optimisation until target has been reached, saving results at regular intervals
while round2_count < round2_target:
    
    # Run a new set of trials on the study object defined above
    round2_study.optimize(round2_objective, n_trials=round2_increment)
    
    # Get a dict of the best-performing hyperparameters
    best = round2_study.best_params
    
    # Create a dataframe from the study results so far
    df = round2_study.trials_dataframe()
    
    # Limit dataframe to only contain successful trials
    df = df.loc[df['state']=='COMPLETE'].copy()
    
    # Generate visualisations of the results so far
    visualise_hparam_tuning_round2(df, best, False, tuning_round=2)
    
    # Save the optuna study object, dictionary of best-performing parameters, and dataframe of results so far
    joblib.dump(round2_study, '{}/hparam_optuna_round2_study.pkl'.format(folder_logs_2D))
    joblib.dump(best, '{}/hparam_optuna_round2_best.pkl'.format(folder_logs_2D))
    df.to_csv('{}/hparam_optuna_round2_df.csv'.format(folder_logs_2D))
    
    # Update the study count (based on number of successful trials) & print status update
    round2_count = len(df.index)
    
    # Save backups of all files, named by number of trials (so they won't be overwritten)
    joblib.dump(round2_study, '{}/hparam_optuna_round2_study_{}.pkl'.format(folder_logs_2D, str(round2_count).zfill(4)))
    joblib.dump(best, '{}/hparam_optuna_round2_best_{}.pkl'.format(folder_logs_2D, str(round2_count).zfill(4)))
    df.to_csv('{}/hparam_optuna_round2_df_{}.csv'.format(folder_logs_2D, str(round2_count).zfill(4)))
    
    print('\n\nRound 2: {:,} trials completed (out of target {:,})\n\n'.format(round2_count, round2_target))


###############################################################################
# 6. Explore potential value of adjusting learning_rate on plateaus           #
###############################################################################

# Read hyperparameter tuning results and get lowest score after Trial 98 (when objective function was modified slightly)
df = pd.read_csv('{}/hparam_optuna_round2_df.csv'.format(folder_logs_2D))
df = df.loc[(df['number']>98) & (df['state']=='COMPLETE')]
best = df.loc[df['value']==df['value'].min()].to_dict(orient='records')[0]

# Define hyperparameters
kernel_initialiser = 'he_normal'
activation = 'elu'
optimiser = 'Nadam'
n_filters_first = best['params_n_filters_first']
n_filters_growthrate = best['params_n_filters_growthrate']
batch_size = best['params_batch_size']
dropout_1 = best['params_dropout_1']
dropout_2 = best['params_dropout_2']
dropout_3 = best['params_dropout_3']
dropout_5 = best['params_dropout_5']
dropout_6 = best['params_dropout_6']
dropout_7 = best['params_dropout_7']
learning_rate = best['params_learning_rate']

# Scenario 1: Reduce lr on plateau w/ factor 0.9
for trial in range(1,6):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.900_{}.csv'.format(folder_logs_2D, trial))
    del model, history, history_df

# Scenario 2: Reduce lr on plateau w/ factor 0.875
for trial in range(1,6):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.875, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.875_{}.csv'.format(folder_logs_2D, trial))
    del model, history, history_df

# Scenario 3: Reduce lr on plateau w/ factor 0.85
for trial in range(1,6):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.850_{}.csv'.format(folder_logs_2D, trial))
    del model, history, history_df

# Scenario 4: Reduce lr on plateau w/ factor 0.825
for trial in range(1,6):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.825, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.825_{}.csv'.format(folder_logs_2D, trial))
    del model, history, history_df

# Scenario 5: Reduce lr on plateau w/ factor 0.8
for trial in range(1,6):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.800_{}.csv'.format(folder_logs_2D, trial))
    del model, history, history_df

# Scenario 6: Reduce lr on plateau w/ factor 0.775
for trial in range(1,6):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.775, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.775_{}.csv'.format(folder_logs_2D, trial))
    del model, history, history_df

# Scenario 7: Reduce lr on plateau w/ factor 0.75
for trial in range(1,6):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
    model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
    history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.750_{}.csv'.format(folder_logs_2D, trial))
    del model, history, history_df

# Scenario 8: Reduce lr on plateau w/ factor 0.8 (with slightly higher starting learning_rate)
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, 0.002, input_size=(100,100,23))
history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=1)
history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.800_higherinitiallr.csv'.format(folder_logs_2D))
del model, history, history_df

# Scenario 9: Reduce lr on plateau w/ factor 0.8 (reduce patience to 8)
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=8, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=1)
history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.800_patience8.csv'.format(folder_logs_2D))
del model, history, history_df

# Scenario 10: Reduce lr on plateau w/ factor 0.8 (increase patience to 12)
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=12, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=1)
history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.800_patience12.csv'.format(folder_logs_2D))
del model, history, history_df

# Scenario 11: Reduce lr on plateau w/ factor 0.8 (increase patience to 15)
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, min_lr=1e-5, verbose=1), EarlyStopping(monitor='val_loss', patience=50, verbose=1)]
dropout = [dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7]
model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=1)
history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
history_df.to_csv('{}/data_augmentation_base_reducelronplateau_factor0.800_patience15.csv'.format(folder_logs_2D))
del model, history, history_df


# Plot 6A: Generate plot showing impact of different factors when reducing learning_rate on plateaus
colours = ['forestgreen','purple','red','royalblue','orange','brown','black']
fig, axes = plt.subplots(nrows=2, figsize=(9,9))
# Loop through the reduction factors tested
for i, factor in enumerate(['0.750','0.775','0.800','0.825','0.850','0.875','0.900']):
    # Read in the five results available
    df1 = pd.read_csv('{}/data_augmentation_base_reducelronplateau_factor{}_1.csv'.format(folder_logs_2D, factor))
    df2 = pd.read_csv('{}/data_augmentation_base_reducelronplateau_factor{}_2.csv'.format(folder_logs_2D, factor))
    df3 = pd.read_csv('{}/data_augmentation_base_reducelronplateau_factor{}_3.csv'.format(folder_logs_2D, factor))
    df4 = pd.read_csv('{}/data_augmentation_base_reducelronplateau_factor{}_4.csv'.format(folder_logs_2D, factor))
    df5 = pd.read_csv('{}/data_augmentation_base_reducelronplateau_factor{}_5.csv'.format(folder_logs_2D, factor))
    # Evaluate validation RMSE at the end of each training process, visualised on the upper plot
    RMSEs = [df['val_root_mean_squared_error'].iloc[-10:].mean() for df in [df1, df2, df3, df4, df5]]
    axes[0].scatter(x=[np.float(factor)]*5, y=RMSEs, color=colours[i])
    axes[0].scatter(x=np.float(factor), y=np.mean(RMSEs), color=colours[i], marker='x', s=50)
    # Add rolling-averaged RMSE progression to lower plot
    axes[1].plot(df1['epoch'].iloc[20:], df1['val_root_mean_squared_error'].rolling(window=7, center=True).mean().iloc[20:], color=colours[i], linewidth=1, alpha=0.5, label='factor={}'.format(factor))
    axes[1].plot(df2['epoch'].iloc[20:], df2['val_root_mean_squared_error'].rolling(window=7, center=True).mean().iloc[20:], color=colours[i], linewidth=1, alpha=0.5)
    axes[1].plot(df3['epoch'].iloc[20:], df3['val_root_mean_squared_error'].rolling(window=7, center=True).mean().iloc[20:], color=colours[i], linewidth=1, alpha=0.5)
    axes[1].plot(df4['epoch'].iloc[20:], df4['val_root_mean_squared_error'].rolling(window=7, center=True).mean().iloc[20:], color=colours[i], linewidth=1, alpha=0.5)
    axes[1].plot(df5['epoch'].iloc[20:], df5['val_root_mean_squared_error'].rolling(window=7, center=True).mean().iloc[20:], color=colours[i], linewidth=1, alpha=0.5)
# Update labels & save figure
axes[0].set_xlabel('Learning rate reduction factor')
axes[0].set_ylabel('Mean of last 10 validation RMSEs')
axes[0].grid(axis='y', alpha=0.5)
axes[1].set_xlabel('Training epochs')
axes[1].set_ylabel('Validation RMSE')
axes[1].set_ylim(top=2.6)
axes[1].grid(axis='y', alpha=0.5)
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig('{}/data_augmentation/reducelronplateau_impact_factor.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 7. Experiment with using GaussianDropout instead of standard Dropout layers #
###############################################################################

# Read hyperparameter tuning results and get lowest score after Trial 98 (when objective function was modified slightly)
df = pd.read_csv('{}/hparam_optuna_round4_df.csv'.format(folder_logs_2D))
df = df.loc[(df['number']>98) & (df['state']=='COMPLETE')]
best = df.loc[df['value']==df['value'].min()].to_dict(orient='records')[0]

# Define hyperparameters
kernel_initialiser = 'he_normal'
activation = 'elu'
optimiser = 'Nadam'
n_filters_first = best['params_n_filters_first']
n_filters_growthrate = best['params_n_filters_growthrate']
batch_size = best['params_batch_size']
dropout_1 = best['params_dropout_1']
dropout_2 = best['params_dropout_2']
dropout_3 = best['params_dropout_3']
dropout_5 = best['params_dropout_5']
dropout_6 = best['params_dropout_6']
dropout_7 = best['params_dropout_7']
learning_rate = best['params_learning_rate']

# Define a range of possible dropout rate patterns to explore
gaussian_dropouts = {'tuned':[dropout_1, dropout_2, dropout_3, dropout_5, dropout_6, dropout_7],
                     'tuned_rounded':[0.15,0.25,0.25,0.35,0.4,0.25],
                     'standard_low':[0.1,0.2,0.3,0.3,0.2,0.1],
                     'standard_high':[0.2,0.3,0.4,0.4,0.3,0.2],
                     'standard_high_bn-same':[0.2,0.3,0.4,0.4,0.4,0.3,0.2],
                     'standard_high_bn-higher':[0.2,0.3,0.4,0.45,0.4,0.3,0.2],
                     'standard_high_bn-lower':[0.2,0.3,0.4,0.35,0.4,0.3,0.2],
                     'standard_high_bn-lowest':[0.2,0.3,0.4,0.2,0.4,0.3,0.2],
                     'standard_high_bn-lowerest':[0.2,0.3,0.4,0.25,0.4,0.3,0.2],
                     'standard_high_bn-lowester':[0.2,0.3,0.4,0.1,0.4,0.3,0.2],
                     'standard_higher':[0.2,0.35,0.5,0.5,0.35,0.2],
                     'decreasing_high':[0.5,0.4,0.3,0.2,0.1,0.05],
                     'increasing_high':[0.05,0.1,0.2,0.3,0.4,0.5],
                     'constant_low':[0.2]*6,
                     'constant_mid':[0.35]*6,
                     'constant_high':[0.5]*6}

# Define callbacks that will be used by all models
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-5, verbose=0), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]

# Run at least three trials for each (and more for those which seem promising), to get a robust estimate of which pattern does best
for trial in range(1,7):
    #for gaussian_dropout in ['tuned','tuned_rounded','standard_low','standard_high','standard_high_bn-same','standard_high_bn-higher','standard_high_bn-lower','standard_high_bn-lowest','standard_high_bn-lowester','standard_higher','decreasing_high','increasing_high','constant_low','constant_mid','constant_high']:
    for gaussian_dropout in ['standard_high','standard_high_bn-same','standard_high_bn-lower','standard_high_bn-lowest','standard_high_bn-lowerest','standard_high_bn-lowester']:
        if not os.path.exists('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, gaussian_dropout, trial)):
            print(gaussian_dropout, trial)
            dropout = gaussian_dropouts[gaussian_dropout]
            model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
            history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=1)
            history_df = pd.DataFrame(history.history)
            history_df['epoch'] = history.epoch
            history_df.to_csv('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, gaussian_dropout, trial))
            print('{}: Val RMSE = {:.4f}'.format(gaussian_dropout, history_df['val_root_mean_squared_error'].iloc[-10:].mean()))
            del model, history, history_df

# Plot 7A: Plot showing validation results for all Gaussian dropout patterns explored
gdos = ['tuned','tuned_rounded','standard_low','standard_high','standard_high_bn-same','standard_high_bn-higher','standard_high_bn-lower','standard_high_bn-lowest','standard_high_bn-lowerest','standard_high_bn-lowester','standard_higher','decreasing_high','increasing_high','constant_low','constant_mid','constant_high']
colours = ['forestgreen','purple','red','royalblue','orange','brown','black','seagreen','firebrick','skyblue','darkviolet','dimgrey','pink','yellow','darkblue','mediumorchid']
fig, axes = plt.subplots(nrows=3, figsize=(12,12))
# Loop through the reduction factors tested
for i, gdo in enumerate(gdos):
    # Read in the results available
    dfs = [pd.read_csv('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, gdo, j)) for j in range(1,7) if os.path.exists('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, gdo, j))]
    # Upper plot: add final RMSE (average over final step training steps)
    final_RMSEs = [df['val_root_mean_squared_error'].iloc[-10:].mean() for df in dfs]
    axes[0].scatter(x=[i]*len(dfs), y=final_RMSEs, color=colours[i])
    axes[0].scatter(x=i, y=np.mean(final_RMSEs), color=colours[i], marker='x', s=50)
    # Middle plot: add minimum RMSE (over all training steps)
    min_RMSEs = [df['val_root_mean_squared_error'].min() for df in dfs]
    axes[1].scatter(x=[i]*len(dfs), y=min_RMSEs, color=colours[i], alpha=0.5)
    axes[1].scatter(x=i, y=np.mean(min_RMSEs), color=colours[i], alpha=0.5, marker='x', s=50)
    # Lower plot: Add rolling-averaged RMSE progression if eventual result below threshold RMSE
    label_threshold = 2.25
    [axes[2].plot(df['epoch'].iloc[20:], df['val_root_mean_squared_error'].rolling(window=7, center=True).mean().iloc[20:], color=colours[i], linewidth=1, alpha=0.5, label=gdo) for df in dfs if df['val_root_mean_squared_error'].iloc[-10:].mean() < label_threshold]
# Update labels & save figure
for k in [0,1]:
    axes[k].set_xticks(range(len(gdos)))
    axes[k].set_xticklabels(['\n'.join(gdo.split('_')) for gdo in gdos])
    axes[k].set_xlabel('Gaussian dropout approach')
    axes[k].grid(axis='y', alpha=0.5)
axes[0].set_ylabel('Mean of last 10 validation RMSEs')
axes[1].set_ylabel('Minimum validation RMSE')
axes[2].set_xlabel('Training epochs')
axes[2].set_ylabel('Validation RMSE')
axes[2].set_ylim(top=2.5)
axes[2].grid(axis='y', alpha=0.5)
axes[2].legend(frameon=False)
fig.tight_layout()
fig.savefig('{}/data_augmentation/gaussian_dropout.png'.format(folder_fig), dpi=300)
plt.close()

# Plot 7B: Plot showing validation results for the shortlisted Gaussian dropout patterns
gdos_selected = ['standard_high','standard_high_bn-same','standard_high_bn-lower','standard_high_bn-lowest','standard_high_bn-lowerest','standard_high_bn-lowester']
colours = ['forestgreen','purple','red','royalblue','orange','brown','black','seagreen','firebrick','skyblue','darkviolet','dimgrey','pink','yellow','darkblue']
fig, axes = plt.subplots(nrows=3, figsize=(12,12))
# Loop through the reduction factors tested
for i, gdo in enumerate(gdos_selected):
    # Read in the results available
    dfs = [pd.read_csv('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, gdo, j)) for j in range(1,7) if os.path.exists('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, gdo, j))]
    # Upper plot: add final RMSE (average over final step training steps)
    final_RMSEs = [df['val_root_mean_squared_error'].iloc[-10:].mean() for df in dfs]
    axes[0].scatter(x=[i]*len(dfs), y=final_RMSEs, color=colours[i])
    axes[0].scatter(x=i, y=np.mean(final_RMSEs), color=colours[i], marker='x', s=50)
    # Middle plot: add minimum RMSE (over all training steps)
    min_RMSEs = [df['val_root_mean_squared_error'].min() for df in dfs]
    axes[1].scatter(x=[i]*len(dfs), y=min_RMSEs, color=colours[i], alpha=0.5)
    axes[1].scatter(x=i, y=np.mean(min_RMSEs), color=colours[i], alpha=0.5, marker='x', s=50)
    # Lower plot: Add rolling-averaged RMSE progression if eventual result below threshold RMSE
    label_threshold = 2.25
    [axes[2].plot(df['epoch'].iloc[20:], df['val_root_mean_squared_error'].rolling(window=7, center=True).mean().iloc[20:], color=colours[i], linewidth=1, alpha=0.5, label=gdo) for df in dfs if df['val_root_mean_squared_error'].iloc[-10:].mean() < label_threshold]
# Update labels & save figure
for k in [0,1]:
    axes[k].set_xticks(range(len(gdos_selected)))
    axes[k].set_xticklabels(['\n'.join(gdo.split('_')) for gdo in gdos_selected])
    axes[k].set_xlabel('Gaussian dropout approach')
    axes[k].grid(axis='y', alpha=0.5)
axes[0].set_ylabel('Mean of last 10 validation RMSEs')
axes[1].set_ylabel('Minimum validation RMSE')
axes[2].set_xlabel('Training epochs')
axes[2].set_ylabel('Validation RMSE')
axes[2].set_ylim(top=2.5)
axes[2].grid(axis='y', alpha=0.5)
axes[2].legend(frameon=False)
fig.tight_layout()
fig.savefig('{}/data_augmentation/gaussian_dropout_selected.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 8. Explore potential value of data augmentation                             #
###############################################################################

# Read hyperparameter tuning results and get lowest score after Trial 98 (when objective function was modified slightly)
df = pd.read_csv('{}/hparam_optuna_round4_df.csv'.format(folder_logs_2D))
df = df.loc[(df['number']>98) & (df['state']=='COMPLETE')]
best = df.loc[df['value']==df['value'].min()].to_dict(orient='records')[0]

# Define hyperparameters
kernel_initialiser = 'he_normal'
activation = 'elu'
optimiser = 'Nadam'
n_filters_first = best['params_n_filters_first']
n_filters_growthrate = best['params_n_filters_growthrate']
batch_size = best['params_batch_size']
learning_rate = best['params_learning_rate']

# Define dropout rates based on the pattern found to be most effective in Step 7 (using GaussianDropout)
dropout = gaussian_dropouts['standard_high_bn-lower']

# Define a custom image data generator, to include the desired data augmentation methods (applied to both features & target arrays of training data)
class GenerateAugmentedData(tf.keras.utils.Sequence):
    
    # Define how the object should be initialised (input data, batch_size & shuffling)
    def __init__(self, features, target, batch_size, fraction_to_augment, rotate=True, hflip=True, vflip=False, shuffle=True):
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.fraction = fraction_to_augment
        self.rotate = rotate
        self.hflip = hflip
        self.vflip = vflip
        self.indices = range(len(self.target))
        self.shuffle = shuffle
        self.on_epoch_end()
        
    # Define a private method to determine the number of steps in each epoch (based on batch_size & number of input patches available)
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    # Define a method to be called after every epoch (e.g. to shuffle the indices)
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
    
    # Define a private method which retrieves a batch of data (based on batch index specified)
    def __getitem__(self, index):
        # Generate indices of the batch
        index = self.index[index * self.batch_size: (index+1) * self.batch_size]
        # Get list of patch IDs to be included in that batch
        batch = [self.indices[k] for k in index]
        # Get data corresponding to that batch of patch indices
        features_batch, target_batch = self.get_data(batch)
        return features_batch, target_batch
    
    # Define a method that retrieves a batch of data, applying augmentation to some images
    def get_data(self, batch):
        fraction_to_augment = self.fraction
        rotate = self.rotate
        hflip = self.hflip
        vflip = self.vflip
        # Filter full input datasets to get selected patches only (based on indices stored in batch variable)
        batch_features = self.features[batch,:,:,:]
        batch_target = self.target[batch,:,:,:]
        # Loop through each pair of patches defined in batch
        for i in range(len(batch)):
            # Randomly decide if that pair of patches should be augmented, based on augmentation fraction defined by user
            if random() < fraction_to_augment:
                # Extract patches relating to that sample
                features_patch = batch_features[i,:,:,:]  # (height, width, channels)
                target_patch = batch_target[i,:,:,:]      # (height, width, channels)
                # If specified, apply a random rotation (by 90, 180, 270 or 360 degrees)
                if rotate:
                    rotation = np.random.randint(1,5)
                    features_patch = np.rot90(features_patch, k=rotation, axes=[1,0])
                    target_patch = np.rot90(target_patch, k=rotation, axes=[1,0])
                # If specified, apply a random horizontal flip
                if hflip:
                    if random() < 0.5:
                        features_patch = np.flip(features_patch, axis=1)
                        target_patch = np.flip(target_patch, axis=1)
                # If specified, apply a random vertical flip
                if vflip:
                    if random() < 0.5:
                        features_patch = np.flip(features_patch, axis=0)
                        target_patch = np.flip(target_patch, axis=0)
                # Update patches
                batch_features[i,:,:,:] = features_patch
                batch_target[i,:,:,:] = target_patch
        # Return the batch of augmented data (features & target)
        return batch_features, batch_target

# Visualise results, to check it's working as intended
patch = 0
channel = 0
GAD = GenerateAugmentedData(features_train_norm, target_train, batch_size, 1.0, rotate=False, hflip=True, vflip=False, shuffle=True)
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10,2))
for i in range(10):
    # Retrieve augmented datasets for a single index
    batch_features, batch_target = GAD.get_data([0])
    # Top row: show feature patch (first channel)
    axes[0,i].imshow(batch_features[patch,:,:,channel])
    axes[0,i].axis('off')
    # Bottom row: show target patch (first channel)
    axes[1,i].imshow(batch_target[patch,:,:,0])
    axes[1,i].axis('off')


# Define callbacks that will be used by all models
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-5, verbose=0), EarlyStopping(monitor='val_loss', patience=30, verbose=1)]

# Try data augmentation that excludes any aspect-related feature (not suitable for geometric augmentation)
filter_features_idx = [i for i,feature in enumerate(selected_features) if not feature.endswith('aspect')]

# Run a series of tests exploring potential value of data augmentation
for trial in range(1,4):
    # Loop through each data subset considered
    for aug_data in ['all','subset']:
        # Loop through each augmentation type considered
        for aug_type in ['d4','rot','hflip','vflip','bflip']:
            # Loop through each augmentation fraction considered
            for aug_fraction in ['0.05']:#,'0.10']:
                # Run test if not done already
                if not os.path.exists('{}/data_aug_{}_{}_{}_{}.csv'.format(folder_logs_2D, aug_data, aug_type, aug_fraction, trial)):
                    print('Processing {}_{}_{}_{}'.format(aug_data, aug_type, aug_fraction, trial))
                    # Process data subset property
                    if aug_type == 'd4':
                        aug_rotate = True
                        aug_hflip = True
                        aug_vflip = False
                    elif aug_type == 'rot':
                        aug_rotate = True
                        aug_hflip = False
                        aug_vflip = False
                    elif aug_type == 'hflip':
                        aug_rotate = False
                        aug_hflip = True
                        aug_vflip = False
                    elif aug_type == 'vflip':
                        aug_rotate = False
                        aug_hflip = False
                        aug_vflip = True
                    elif aug_type == 'bflip':
                        aug_rotate = False
                        aug_hflip = True
                        aug_vflip = True
                    # Convert augmentation fraction to a float
                    aug_fraction = float(aug_fraction)
                    # Establish appropriate data generator, then build & train a model to use it
                    if aug_data == 'subset':
                        aug_data_gen = GenerateAugmentedData(features_train_norm[:,:,:,filter_features_idx], target_train, batch_size, aug_fraction, rotate=aug_rotate, hflip=aug_hflip, vflip=aug_vflip, shuffle=True)
                        model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,22))
                        history = model.fit(aug_data_gen, steps_per_epoch=len(features_train_norm)//batch_size, epochs=1000, validation_data=(features_dev_norm[:,:,:,filter_features_idx], target_dev), callbacks=callbacks, verbose=0)
                    elif aug_data == 'all':
                        aug_data_gen = GenerateAugmentedData(features_train_norm, target_train, batch_size, aug_fraction, rotate=aug_rotate, hflip=aug_hflip, vflip=aug_vflip, shuffle=True)
                        model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
                        history = model.fit(aug_data_gen, steps_per_epoch=len(features_train_norm)//batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
                    # Process and save results
                    history_df = pd.DataFrame(history.history)
                    history_df['epoch'] = history.epoch
                    history_df.to_csv('{}/data_aug_{}_{}_{}_{}.csv'.format(folder_logs_2D, aug_data, aug_type, aug_fraction, trial))
                    print(' - Val RMSE = {:.4f}'.format(history_df['val_root_mean_squared_error'].iloc[-10:].mean()))
                    del model, history, history_df


# Plot: Visualise impact of different data augmentation approaches
base = 'standard_high_bn-lower'
colours = ['forestgreen','purple','red','royalblue','orange','brown','black','seagreen','firebrick','skyblue','darkviolet','dimgrey','pink','yellow','darkblue']
fig, axes = plt.subplots(figsize=(9,6))
# Add the base case as the reference point
dfs = [pd.read_csv('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, base, j)) for j in range(1,7) if os.path.exists('{}/dropout_gaussian_{}_{}.csv'.format(folder_logs_2D, base, j))]
final_RMSEs = [df['val_root_mean_squared_error'].iloc[-10:].mean() for df in dfs]
axes.scatter(x=[0]*len(dfs), y=final_RMSEs, color=colours[0])
axes.scatter(x=0, y=np.mean(final_RMSEs), color=colours[0], marker='x', s=50)
# Initialise an index for x-axis value
plt_idx = 0
xtick_labels = ['base_({})'.format(len(dfs))]
# Loop through the different approaches tested
for aug_data in ['all','subset']:
    for aug_type in ['d4','rot','hflip','vflip','bflip']:
        for aug_fraction in ['0.05','0.10']:
                # Process if at least one result is available for that combination
                if os.path.exists('{}/data_aug_{}_{}_{}_1.csv'.format(folder_logs_2D, aug_data, aug_type, aug_fraction)):
                    plt_idx += 1
                    # Read in the results available
                    dfs = [pd.read_csv('{}/data_aug_{}_{}_{}_{}.csv'.format(folder_logs_2D, aug_data, aug_type, aug_fraction, j)) for j in range(1,7) if os.path.exists('{}/data_aug_{}_{}_{}_{}.csv'.format(folder_logs_2D, aug_data, aug_type, aug_fraction, j))]
                    # Add final RMSE (average over final 10 training steps)
                    final_RMSEs = [df['val_root_mean_squared_error'].iloc[-10:].mean() for df in dfs]
                    axes.scatter(x=[plt_idx]*len(dfs), y=final_RMSEs, color=colours[plt_idx])
                    axes.scatter(x=plt_idx, y=np.mean(final_RMSEs), color=colours[plt_idx], marker='x', s=50)
                    xtick_labels.append('{}_{}_{}_({})'.format(aug_data, aug_type, aug_fraction, len(dfs)))
# Update labels & save figure
axes.set_xticks(range(len(xtick_labels)))
axes.set_xticklabels(['\n'.join(xtick_label.split('_')) for xtick_label in xtick_labels])
axes.set_xlabel('Data augmentation approach')
axes.grid(axis='y', alpha=0.5)
axes.set_ylabel('Mean of last 10 validation RMSEs [m]')
axes.set_title('Impact of various data augmentation approaches on validation RMSE')
fig.tight_layout()
fig.savefig('{}/data_augmentation/data_augmentation.png'.format(folder_fig), dpi=300)
plt.close()


###############################################################################
# 9. Production ensemble runs, using final tuned model hyperparameters        #
###############################################################################

# Read hyperparameter tuning results and get lowest score after Trial 98 (when objective function was modified slightly)
df = pd.read_csv('{}/hparam_optuna_round4_df.csv'.format(folder_logs_2D))
df = df.loc[(df['number']>98) & (df['state']=='COMPLETE')]
best = df.loc[df['value']==df['value'].min()].to_dict(orient='records')[0]

# Define hyperparameters
kernel_initialiser = 'he_normal'
activation = 'elu'
optimiser = 'Nadam'
n_filters_first = best['params_n_filters_first']
n_filters_growthrate = best['params_n_filters_growthrate']
batch_size = best['params_batch_size']
learning_rate = best['params_learning_rate']

# Define dropout rates based on the pattern found to be most effective in Step 7 (using GaussianDropout)
dropout = [0.2,0.3,0.4,0.35,0.4,0.3,0.2]

# Build & train a series of different models, to explore potential value of ensemble approach (given stochastic elements)
for i in range(25):
    
    # Convert index number to zero-padded string, for easier filenaming
    i = str(i).zfill(2)
    
    # Proceed if corresponding model doesn't yet exist
    if not os.path.exists('{}/convnet_{}.h5'.format(folder_models, i)):
        
        # Define callbacks that will be used by all models, including model checkpoint to save only the best model
        callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-5, verbose=0),
                     EarlyStopping(monitor='val_loss', patience=50, verbose=0),
                     ModelCheckpoint('{}/convnet_{}.h5'.format(folder_models, i), monitor='val_loss', save_best_only=True, verbose=0)]
        
        # Build & train a model, using early stopping (patience=50) and saving the best model
        print('Training model {}...'.format(i))
        model = build_unet_model(n_filters_first, n_filters_growthrate, dropout, kernel_initialiser, activation, optimiser, learning_rate, input_size=(100,100,23))
        history = model.fit(features_train_norm, target_train, batch_size=batch_size, epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = history.epoch
        print(' - Lowest validation RMSE = {:.4f}'.format(history_df['val_root_mean_squared_error'].min()))
        history_df.to_csv('{}/convnet_traininghistory_{}.csv'.format(folder_logs_2D, i))
        del model, history, history_df


###############################################################################
# 10. Use each trained model to generate predictions for each data subset     #
###############################################################################

# Use function defined previously to import test data in a convnet-ready format, normalising it based on the training data as for the other inputs
features_test, target_test = process_input_data('test', selected_features_idx)
features_test_norm = (features_test - features_train_mean)/features_train_std

# Initialise a dataframe that will hold performance metrics for all convnet models trained (as well as metrics for the initial error & naive correction)
df = pd.DataFrame(columns=['Type','Dataset','RMSE'])

# Calculate the naive predictions possible (simply the mean of the corrections observed from the training data)
naive_predictions = np.nanmean(target_train)

# Build a dictionary to hold residuals for each dataset
residuals = {dataset:{model:None for model in ['initial','naive']} for dataset in ['train','dev','test']}

# Loop through each dataset, populating dataframe with initial RMSE & naive RMSE for each
for (dataset, features_norm, target) in zip(['train','dev','test'], [features_train_norm, features_dev_norm, features_test_norm], [target_train, target_dev, target_test]):
    # Calculate initial RMSE, based on target data
    MSE_initial = np.nanmean(np.square(target))
    RMSE_initial = np.sqrt(MSE_initial)
    df = df.append({'Type':'initial', 'Dataset':dataset, 'RMSE':RMSE_initial}, ignore_index=True)
    # Calculate RMSE after applying a naive correction (subtracting mean difference observed from training data)
    MSE_naive = np.nanmean(np.square(target - naive_predictions))
    RMSE_naive = np.sqrt(MSE_naive)
    df = df.append({'Type':'naive', 'Dataset':dataset, 'RMSE':RMSE_naive}, ignore_index=True)
    # Save residual arrays in the dictionary initialised previously
    residuals[dataset]['initial'] = target.flatten()
    residuals[dataset]['naive'] = (target - naive_predictions).flatten()

# Add to this dataframe the RMSE scores returned for each of the available convnet models, saving predictions for each
for i in range(25):
    # Convert index number to zero-padded string, for easier filenaming
    i = str(i).zfill(2)
    # Process if corresponding model is available
    if os.path.exists('{}/convnet_{}.h5'.format(folder_models, i)):
        # Load convnet model
        model = tf.keras.models.load_model('{}/convnet_{}.h5'.format(folder_models, i))
        # Loop through each available dataset, as above
        for (dataset, features_norm, target) in zip(['train','dev','test'], [features_train_norm, features_dev_norm, features_test_norm], [target_train, target_dev, target_test]):
            # Evalute model for RMSE score
            loss_convnet, RMSE_convnet = model.evaluate(features_norm, target, verbose=0)
            df = df.append({'Type':'convnet_{}'.format(i), 'Dataset':dataset, 'RMSE':RMSE_convnet}, ignore_index=True)
            print('Model {}: {} data: RMSE = {:.4f} m'.format(i, dataset, RMSE_convnet))
            # Generate predictions using the convnet, in order to save residuals
            predictions = model.predict(features_norm)
            residuals[dataset]['convnet_{}'.format(i)] = (target - predictions).flatten()
            # Store predictions for further analysis (ensemble approach)
            np.save('{}/predictions/convnet_{}_{}_prediction.npy'.format(folder_results, i, dataset), predictions)

# Save metric results for individual models to a CSV file, for easy import & comparison later
df.to_csv('{}/convnet_metrics_models.csv'.format(folder_results))

# Save residuals to a pickle object, for later import and comparison
pickle.dump(residuals, open('{}/convnet_residuals_models.p'.format(folder_results), 'wb'))


###############################################################################
# 11. Confirm value of ensemble approach (predictions from multiple models)   #
###############################################################################

# Define a list of labels for the models/results available
model_idxs = [i for i in range(50) if os.path.exists('{}/convnet_{}.h5'.format(folder_models, str(i).zfill(2)))]

# Specify number of models/results available
n_models = len(model_idxs)

# Re-open the metrics summary dataframe, if no longer in memory
df_metrics = pd.read_csv('{}/convnet_metrics_models.csv'.format(folder_results))

# Load all predictions into a single array
predictions_array = np.array([np.load('{}/predictions/convnet_{}_test_prediction.npy'.format(folder_results, str(model_idx).zfill(2))) for model_idx in range(25)])

# Now loop through a range of all possible ensemble sizes
for size in [1,2,3,4,5,6,7,8,18,19,20,21,22,23,24,25]:  # Just look at each end of the scale, to confirm expected trend
    # Only process if that result doesn't yet exist (very time-consuming)
    if not os.path.exists('{}/convnet_metrics_ensembles_{}models.csv'.format(folder_results, str(size).zfill(2))):
        # Initialise a dataframe to save results for ensembles of that particular size
        df = pd.DataFrame(columns=['Size','Ensemble','RMSE'])
        # Get a list of possible model combinations, of the appropriate ensemble size
        ensembles_idx = list(combinations(model_idxs, size))
        print('Processing {}-model ensembles ({} combinations)...'.format(size, len(ensembles_idx)))
        # Loop through each of the possible combinations identified
        for ensemble_idx in ensembles_idx:
            ensemble_label = '_'.join([str(e).zfill(2) for e in ensemble_idx])
            # Get the mean prediction using all individual model predictions included in the ensemble
            predictions_mean = np.mean(predictions_array[ensemble_idx,:,:,:], axis=0)
            # Evaluate the RMSE using this ensemble prediction & append RMSE to overall list
            ensemble_RMSE = np.sqrt(np.nanmean(np.square(target_test - predictions_mean)))
            # Update the dataframe
            df = df.append({'Size':size, 'Ensemble':ensemble_label, 'RMSE':ensemble_RMSE}, ignore_index=True)
        # Save ensemble dataframe to disk
        df.to_csv('{}/convnet_metrics_ensembles_{}models.csv'.format(folder_results, str(size).zfill(2)))
        print('{}-model ensemble: mean test RMSE = {:.4f}m'.format(size, df['RMSE'].mean()))

# Alternative approach 1: Check test score using only the model which returned the lowest validation score
min_dev_model_idx = df_metrics['RMSE'].loc[(df_metrics['Dataset']=='dev') & (df_metrics['Type'].str.startswith('convnet', na=False)) & (~df_metrics['Type'].str.endswith('ensemble', na=False))].argmin()
min_dev_model = df_metrics['Type'].loc[(df_metrics['Dataset']=='dev') & (df_metrics['Type'].str.startswith('convnet', na=False)) & (~df_metrics['Type'].str.endswith('ensemble', na=False))].iloc[min_dev_model_idx]
min_dev_prediction = np.load('{}/predictions/{}_test_prediction.npy'.format(folder_results, min_dev_model))
min_dev_RMSE = np.sqrt(np.nanmean(np.square(target_test - min_dev_prediction)))

# Alternative approach 2: Use all available models to generate an average prediction, but weighted based on their respective validation scores
dev_RMSEs = [df_metrics['RMSE'].loc[(df_metrics['Dataset']=='dev')&(df_metrics['Type']=='convnet_{}'.format(str(i).zfill(2)))].item() for i in range(25)]
weights = [1/dev_RMSE for dev_RMSE in dev_RMSEs]
total_weight = np.sum(weights)
predictions_weighted_mean = np.sum([weight * np.load('{}/predictions/convnet_{}_test_prediction.npy'.format(folder_results, str(m).zfill(2))) for weight, m in zip(weights, range(25))], axis=0) / total_weight
predictions_weighted_RMSE = np.sqrt(np.nanmean(np.square(target_test - predictions_weighted_mean)))

# Initialise a figure to visualise impact of ensemble approach, as well as two alternatives
fig, axes = plt.subplots(nrows=2, figsize=(9,9))
# Loop through a range of ensemble sizes
for size in [1,2,3,4,5,6,7,8,18,19,20,21,22,23,24,25]:
    # Read results for ensembles of that size into a dataframe
    df = pd.read_csv('{}/convnet_metrics_ensembles_{}models.csv'.format(folder_results, str(size).zfill(2)))
    # Extract RMSEs corresponding to all ensembles of that size
    ensembles_RMSEs = df['RMSE'].loc[df['Size']==size].to_list()
    print('{}-model ensembles: Mean RMSE = {:.5f}m'.format(size, np.mean(ensembles_RMSEs)))
    # Upper: Visualise results using a boxplot
    axes[0].boxplot(x=ensembles_RMSEs, positions=[size], showmeans=True)
    # Lower: Just look at mean RMSE for each ensemble size
    axes[1].scatter(size, np.mean(ensembles_RMSEs), color='blue')
# Add line showing test score achieved by alternative approach 1 (simply using model which returned lowest validation error)
axes[0].axhline(y=min_dev_RMSE, color='green', linestyle='dashed', label='Use model with lowest validation error')
# Add line showing test score achieved by alternative approach 2 (weighted mean of all predictions, based on validation error achieved by each model)
axes[0].axhline(y=predictions_weighted_RMSE, color='purple', linestyle='dashed', label='Weighted mean of all model predictions')
axes[0].legend(frameon=False)
# Add general figure properties
[axes[i].set_xlabel('Ensemble size (number of models)') for i in [0,1]]
[axes[i].set_ylabel('Test RMSE [m]') for i in [0,1]]
[axes[i].grid(axis='y', which='major', color='dimgrey', alpha=0.25) for i in [0,1]]
[[axes[i].spines[edge].set_visible(False) for edge in ['top','right']] for i in [0,1]]
axes[0].set_title('Impact of ensemble size on expected test RMSE')
fig.tight_layout()
fig.savefig('{}/convnet_RMSE_ensembles.png'.format(folder_fig), dpi=300)
plt.close()

# Read the dataframe of model metrics back into memory, to add a final set of results (using 25-model ensemble)
df_metrics = pd.read_csv('{}/convnet_metrics_models.csv'.format(folder_results))

# Read the dictionary of residuals back into memory too, to update it with ensemble results
with open('{}/convnet_residuals_models.p'.format(folder_results), 'rb') as f:
    residuals = pickle.load(f)

# Loop through each dataset, populating dataframe with RMSE derived from 25-model ensemble predictions
for (dataset, target) in zip(['train','dev','test'], [target_train, target_dev, target_test]):
    # Get the 25-model ensemble mean prediction for that dataset
    predictions_mean = np.mean([np.load('{}/predictions/convnet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), dataset)) for i in range(25)], axis=0)
    # Evaluate RMSE for that ensemble prediction, compared to target
    ensemble_RMSE = np.sqrt(np.nanmean(np.square(target - predictions_mean)))
    # Update the metric summary dataframe
    df_metrics = df_metrics.append({'Type':'convnet_ensemble', 'Dataset':dataset, 'RMSE':ensemble_RMSE}, ignore_index=True)
    print('Ensemble: {} data: RMSE = {:.4f} m'.format(dataset, ensemble_RMSE))
    # Save residuals to same dictionary as before
    residuals[dataset]['convnet_ensemble'] = (target - predictions_mean).flatten()
    # Save the ensemble prediction array as a .npy file for easy import later
    np.save('{}/predictions/convnet_ensemble_{}_prediction.npy'.format(folder_results, dataset), predictions_mean)

# Save new versions of the metrics summary & residuals dict
df_metrics.to_csv('{}/convnet_metrics_models.csv'.format(folder_results))
pickle.dump(residuals, open('{}/convnet_residuals_models.p'.format(folder_results), 'wb'))


###############################################################################
# 12. Compare overall RMSE reduction efficacy of all models considered        #
###############################################################################

# Read in metrics summaries for random forest (RF) and densely-connected network (densenet), for comparison
df_rf = pd.read_csv('{}/rf_23features_250trees_metrics_by_dataset.csv'.format(folder_results.replace('convnet','rf')))
df_densenet = pd.read_csv('{}/densenet_metrics_models.csv'.format(folder_results.replace('convnet','densenet')))
df_convnet = pd.read_csv('{}/convnet_metrics_models.csv'.format(folder_results))

# Generate summary plots for each of the 'dev' and 'test' datasets
# Note: remember that the pixel-based approaches (RF & densenet) looked at slightly different training data than the patch-based approach (convnet)
for i, dataset in enumerate(['dev','test']):
    # Extract appropriate results from all modelling result dataframes
    # Note: Take the 'naive' results from the RF/densenet approach, as those approaches learned from slightly larger training dataset (not restricted to intact patches)
    RMSE_initial = df_rf[(df_rf['Dataset']==dataset)&(df_rf['Type']=='initial')]['RMSE'].values.item()
    RMSE_baseline = df_rf[(df_rf['Dataset']==dataset)&(df_rf['Type']=='baseline')]['RMSE'].values.item()
    RMSE_rf = df_rf[(df_rf['Dataset']==dataset)&(df_rf['Type']=='rf')]['RMSE'].values.item()
    RMSE_densenet = df_densenet[(df_densenet['Dataset']==dataset)&(df_densenet['Type']=='densenet_ensemble')]['RMSE'].values.item()
    RMSE_convnet = df_convnet[(df_convnet['Dataset']==dataset)&(df_convnet['Type']=='convnet_ensemble')]['RMSE'].values.item()
    # Set up the figure
    fig, axes = plt.subplots(figsize=(9,4.5))
    axes.bar([0,1,2,3,4], [RMSE_initial, RMSE_baseline, RMSE_rf, RMSE_densenet, RMSE_convnet], color=dataset_colours[dataset], alpha=0.5)
    axes.set_xticks([0,1,2,3,4])
    axes.yaxis.set_tick_params(length=0)
    axes.set_xticklabels(['Initial\nerror','Baseline\ncorrection','RF\ncorrection','DCN\ncorrection','FCN\ncorrection'])
    axes.set_ylabel('Root Mean Square Error [m]')
    axes.grid(axis='y', which='major', color='dimgrey', alpha=0.1)
    [axes.spines[edge].set_visible(False) for edge in ['left','top','right']]
    # Add a horizontal line showing the initial error & a label
    axes.axhline(y=RMSE_initial, color=dataset_colours[dataset], linestyle='dashed', alpha=0.5)
    axes.annotate('{:.3f}m'.format(RMSE_initial), xy=(0, RMSE_initial), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
    # Add labels indicating improvement achieved by each method
    for j, RMSE_new in enumerate([RMSE_baseline, RMSE_rf, RMSE_densenet, RMSE_convnet]):
        # Add downward arrow from initial RMSE to improved MAE
        axes.annotate('', xy=(j+1, RMSE_new), xytext=(j+1, RMSE_initial), arrowprops=dict(arrowstyle='->'))
        # Add label indicating new RMAE and the percentage improvement it equates to
        improvement_percentage = (RMSE_new-RMSE_initial)/RMSE_initial * 100.
        axes.annotate('{:.3f}m ({:.1f}%)'.format(RMSE_new, improvement_percentage), xy=(j+1, RMSE_new), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
    axes.set_title('Performance on {} dataset'.format('validation' if dataset=='dev' else dataset))
    fig.tight_layout()
    fig.savefig('{}/convnet_RMSE_reduction_{}.png'.format(folder_fig, dataset), dpi=300)
    plt.close()


###############################################################################
# 12. Generate intact prediction arrays for each zone (for mapping/GeoTIFFs)  #
###############################################################################

# Define list of zones to be processed (separate LiDAR coverage areas) - noting that TSM17_GLB was too big to process
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM16_ATG']

# Use function defined previously to import training data in a convnet-ready format (for use in normalisation)
features_train, target_train = process_input_data('train', selected_features_idx)

# Calculate feature-wise means & standard deviations using only the training data
features_train_mean = np.mean(features_train, axis=(0,1,2), keepdims=True)
features_train_std = np.std(features_train, axis=(0,1,2), keepdims=True)

# Once these are calculated, the training data can be deleted (very large)
del features_train, target_train

# 12a. Loop through all zones, import data, and generate prediction arrays from each model (saving as .npy files)
for zone in zones:
    
    # Open full numpy array of feature data for that zone (including partial patches) & filter for selected features only
    print('Processing {} zone...'.format(zone))
    features = np.load('{}/Input2D_Features_ByZone_{}.npy'.format(folder_input_2D, zone))
    features = features[:,:,:,selected_features_idx]
    
    # Normalise the feature data based on the training data, as done before, then delete unnormalised features
    features_norm = (features - features_train_mean)/features_train_std
    del features
    
    # Loop through all individual models available
    for i in range(25):
        
        # Only proceed if corresponding prediction doesn't yet exist
        if not os.path.exists('{}/predictions/convnet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), zone)):
            
            # Load that convnet model, generate predictions using that model, and append that array to the overall list
            model = tf.keras.models.load_model('{}/convnet_{}.h5'.format(folder_models, str(i).zfill(2)))
            model_prediction = model.predict(features_norm)
            np.save('{}/predictions/convnet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), zone), model_prediction)
            del model_prediction

# 12b. Loop through all zones, generating ensemble prediction & then re-assembling into intact image
for zone in zones:
    print('Processing {} zone...'.format(zone))
    
    # Calculate ensemble prediction for that zone (mean of predictions from individual models)
    ensemble_prediction = np.nanmean([np.load('{}/predictions/convnet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), zone)) for i in range(25)], axis=0)
    
    # Save ensemble prediction to a .npy file
    np.save('{}/predictions/convnet_ensemble_{}_prediction.npy'.format(folder_results, zone), ensemble_prediction)
    
    # Import the CSV describing this zone's data generation, as a dataframe
    df_patches = pd.read_csv('{}/target_patch_df_ByZone_{}.csv'.format(folder_logs_2D.replace('convnet','patches'), zone))
    
    # Get the range of patches in each axis
    n_patches_x = df_patches['i'].max() + 1
    n_patches_y = df_patches['j'].max() + 1
    
    # Reshape the predicted patches back into an intact image (based on known patch dimensions)
    ensemble_prediction_intact = np.block([[[ensemble_prediction[j*n_patches_x + i,:,:,0] for i in range(n_patches_x)] for j in range(n_patches_y)]])
    
    # Save the prediction array as a .npy file for easy import into the mapping & section extraction script
    np.save('{}/predictions/convnet_ensemble_{}_prediction_intact.npy'.format(folder_results, zone), ensemble_prediction_intact)
    del ensemble_prediction, df_patches, ensemble_prediction_intact