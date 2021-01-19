# Predictive model: Densely-connected neural network (densenet/DCN)

# Import TensorFlow modules & submodules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import other modules required
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import pickle
import joblib
from itertools import combinations

# Define paths to relevant folders
folder_input_1D = 'E:/mdm123/D/ML/inputs/1D'
folder_logs = 'E:/mdm123/D/ML/logs/densenet'
folder_logs_rf = 'E:/mdm123/D/ML/logs/rf'
folder_models = 'E:/mdm123/D/ML/models/densenet'
folder_results = 'E:/mdm123/D/ML/results/densenet'
folder_results_rf = 'E:/mdm123/D/ML/results/rf'
folder_fig = 'E:/mdm123/D/figures/models/densenet'

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM17_GLB', 'TSM16_ATG']

# Define a dictionary of colours for each dataset
dataset_colours = {'train':'blue', 'dev':'green', 'test':'firebrick'}

# Define no_data value
no_data = -9999

# Define a function to normalise a dataframe, given the dataframe & its corresponding statistical description (based on training data only)
def normalise_df(df, stats_train):
    return (df - stats_train['mean'])/stats_train['std']

# Preliminary step (before GPU initialised) to resolve Out of Memory issues later: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialised
    print(e)


###############################################################################
# 1. Import & pre-process data                                                #
###############################################################################

# Read the list of selected features into memory (based on subset chosen during SFFS with Random Forest models)
with open('{}/feature_selection_final_23_features.p'.format(folder_logs_rf), 'rb') as f:
    selected_features = pickle.load(f)

# Import processed CSVs into dataframes
data_train = pd.read_csv('{}/Input1D_Train.csv'.format(folder_input_1D))
data_dev = pd.read_csv('{}/Input1D_Dev.csv'.format(folder_input_1D))

# Split train & dev sets into target & features, filtering for selected feature columns
target_train = np.array(data_train['diff'].copy().values).reshape((len(data_train.index),1))
target_dev = np.array(data_dev['diff'].copy().values).reshape((len(data_dev.index),1))
features_train = data_train[selected_features].copy()
features_dev = data_dev[selected_features].copy()

# Check for any missing data
target_train[(target_train==-9999)|np.isnan(target_train)].sum()
target_dev[(target_dev==-9999)|np.isnan(target_dev)].sum()
features_train[(features_train==-9999)|features_train.isnull()].sum()
features_dev[(features_dev==-9999)|features_dev.isnull()].sum()

# Save the overall statistics based on the training data
stats_train = features_train.describe()
stats_train = stats_train.transpose()

# Normalise all sets of input features (train, dev)
features_train_norm = normalise_df(features_train, stats_train)
features_dev_norm = normalise_df(features_dev, stats_train)


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
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16,12))
    # 1: Title
    axes[0,0].annotate('Round {}\n({} trials)'.format(tuning_round, len(df.index)), xy=(0.5, 0.65), ha='center', va='center', size=16, color='dimgrey', weight='bold')
    axes[0,0].annotate('Min RMSE: {:.4f}'.format(df['value'].min()), xy=(0.5, 0.35), ha='center', va='center', size=12, color='dimgrey', weight='bold')
    axes[0,0].axis('off')
    # 2: n_layers
    hp = 'n_layers'
    quintiles = get_quintiles(df, hp)
    bins=[edge-0.5 for edge in range(df['params_{}'.format(hp)].min(), df['params_{}'.format(hp)].max()+2, 1)]
    axes[0,1].hist(quintiles, bins=bins, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0,1].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0,1].set_title('n_layers')
    [axes[0,1].spines[edge].set_visible(False) for edge in ['top','right']]
    # 3: batch_size
    hp = 'batch_size'
    quintiles = get_quintiles(df, hp)
    axes[1,0].hist(quintiles, bins=20, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[1,0].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[1,0].set_title(hp)
    [axes[1,0].spines[edge].set_visible(False) for edge in ['top','right']]
    # 4: adam_lr
    hp = 'adam_learning_rate'
    quintiles = get_quintiles(df, hp)
    axes[1,1].hist(quintiles, bins=20, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[1,1].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[1,1].set_title(hp)
    [axes[1,1].spines[edge].set_visible(False) for edge in ['top','right']]
    # 5: use_batch_norm
    hp = 'batch_norm'
    df['params_{}'.format(hp)].value_counts().sort_index().plot(kind='bar', ax=axes[2,0]).set_title(hp)
    plt.xticks(rotation=0)
    for j, label in enumerate(axes[2,0].get_xticklabels()):
        if label.get_text() == str(best[hp]):
            axes[2,0].axvline(x=j, linestyle='dashed', color='red')
    [axes[2,0].spines[edge].set_visible(False) for edge in ['top','right']]
    # 6: use_dropout
    hp = 'use_dropout'
    df['params_{}'.format(hp)].value_counts().sort_index().plot(kind='bar', ax=axes[2,1]).set_title(hp)
    plt.xticks(rotation=0)
    for j, label in enumerate(axes[2,1].get_xticklabels()):
        if label.get_text() == str(best[hp]):
            axes[2,1].axvline(x=j, linestyle='dashed', color='red')
    [axes[2,1].spines[edge].set_visible(False) for edge in ['top','right']]
    # 7: use_weight_decay
    hp = 'use_weight_decay'
    df['params_{}'.format(hp)].value_counts().sort_index().plot(kind='bar', ax=axes[3,0]).set_title(hp)
    plt.xticks(rotation=0)
    for j, label in enumerate(axes[3,0].get_xticklabels()):
        if label.get_text() == str(best[hp]):
            axes[3,0].axvline(x=j, linestyle='dashed', color='red')
    [axes[3,0].spines[edge].set_visible(False) for edge in ['top','right']]
    # 8: weight_decay
    hp = 'weight_decay'
    quintiles = get_quintiles(df, hp)
    axes[3,1].hist(quintiles, bins=20, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    if hp in best.keys():
        axes[3,1].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[3,1].set_title(hp)
    [axes[3,1].spines[edge].set_visible(False) for edge in ['top','right']]
    # Loop through each potential layer, plotting results for n_units_lx and dropout_rate_lx (for x in 1-4)
    for l in range(1,5):
        # n_units_lx (for x: 1-4)
        hp = 'n_units_l{}'.format(l)
        if 'params_{}'.format(hp) in df.columns:
            quintiles = get_quintiles(df, hp)
            axes[l-1,2].hist(quintiles, bins=20, histtype='barstacked', color=sequence_colours, label=sequence_labels)
            if hp in best.keys():
                axes[l-1,2].axvline(x=best[hp], linestyle='dashed', color='red')
        axes[l-1,2].set_title(hp)
        [axes[l-1,2].spines[edge].set_visible(False) for edge in ['top','right']]
        # dropout_rate_lx (for x: 1-4)
        hp = 'dropout_rate_l{}'.format(l)
        if 'params_{}'.format(hp) in df.columns:
            quintiles = get_quintiles(df, hp)
            axes[l-1,3].hist(quintiles, bins=20, histtype='barstacked', color=sequence_colours, label=sequence_labels)
            if hp in best.keys():
                axes[l-1,3].axvline(x=best[hp], linestyle='dashed', color='red')
        axes[l-1,3].set_title(hp)
        [axes[l-1,3].spines[edge].set_visible(False) for edge in ['top','right']]
    # General figure properties
    fig.tight_layout()
    fig.savefig('{}/hyperparameter_tuning/densenet_tuning_round{}.png'.format(folder_fig, tuning_round), dpi=300)
    plt.close()
    
# Define a plotting function to visualise the Round 2 hyperparameter tuning results
def visualise_hparam_tuning_round2(df, best, tuning_round):
    fig, axes = plt.subplots(ncols=2, figsize=(10,6))
    # 1: batch_size
    hp = 'batch_size'
    quintiles = get_quintiles(df, hp)
    axes[0].hist(quintiles, bins=20, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[0].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[0].set_title(hp)
    [axes[0].spines[edge].set_visible(False) for edge in ['top','right']]
    # 4: adam_lr
    hp = 'adam_learning_rate'
    quintiles = get_quintiles(df, hp)
    axes[1].hist(quintiles, bins=20, histtype='barstacked', color=sequence_colours, label=sequence_labels)
    axes[1].axvline(x=best[hp], linestyle='dashed', color='red')
    axes[1].set_title(hp)
    [axes[1].spines[edge].set_visible(False) for edge in ['top','right']]
    # General figure properties
    fig.tight_layout()
    fig.savefig('{}/hyperparameter_tuning/densenet_tuning_round{}.png'.format(folder_fig, tuning_round), dpi=300)
    plt.close()


###############################################################################
# 3. First round of hyperparameter tuning by Bayesian optimisation (Optuna)   #
###############################################################################

# Use the tf.keras loss function to compute the root mean squared error later
RMSE = RootMeanSquaredError()

# Define an early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Define an objective function to be minimised
def round1_objective(trial):
    
    # Sample hyperparameters which will be used throughout the model
    n_layers = trial.suggest_int('n_layers', 1, 4)                                                    # Number of layers
    use_batch_norm = trial.suggest_categorical('batch_norm', [True, False])                           # Whether or not to use batch normalisation
    adam_lr = trial.suggest_loguniform('adam_learning_rate', 1e-4, 1e-1)                              # Learning rate used for Adam optimiser
    batch_size = int(trial.suggest_loguniform('batch_size', 1024, 4096))                              # Mini-batch size
    
    # Decide on regularisation options to be used
    use_dropout = trial.suggest_categorical('use_dropout', [True, False])                             # Whether or not to use dropout layers
    use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])                   # Whether or not to use lasso (L2) regularisation
    if use_weight_decay:
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)                          # Weight decay rate used in L2 regularisation
    
    # Instantiate a keras Sequence model
    model = Sequential()
    
    # Sample the number of units to use in the first layer
    n_units = int(trial.suggest_loguniform('n_units_l1', 2, 4096))
    
    # Add the first layer
    model.add(Dense(n_units, input_shape=(features_train_norm.shape[1],), kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(weight_decay) if use_weight_decay else None))
    if use_batch_norm: model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # If required, add dropout layer for regularisation
    if use_dropout:
        dropout_rate = trial.suggest_float('dropout_rate_l1', 0.0, 0.5)
        model.add(Dropout(dropout_rate))
    
    # Loop through each of the remaining hidden layers to be added
    for i in range(n_layers-1):
        l = i+2
        # Sample the number of units to use for that layer
        n_units = int(trial.suggest_loguniform('n_units_l{}'.format(l), 2, 4096))
        # Add new layer with the sampled number of hidden units
        model.add(Dense(n_units, kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(weight_decay) if use_weight_decay else None))
        if use_batch_norm: model.add(BatchNormalization())
        model.add(Activation('relu'))
        # If required, add dropout layer for regularisation
        if use_dropout:
            dropout_rate = trial.suggest_float('dropout_rate_l{}'.format(l), 0.0, 0.5)
            model.add(Dropout(dropout_rate))
    # Final layer has a single unit & no activation
    model.add(Dense(1))
    # Compile the model - using MSE as the loss metric & with the Adam optimiser
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr), metrics=[RMSE])
    # Fit the model to the training data (features_train_norm & target_train)
    history = model.fit(features_train_norm, target_train, epochs=100, batch_size=batch_size, validation_data=(features_dev_norm, target_dev), callbacks=[early_stop], verbose=0)
    # Get the best validation RMSE
    min_RMSE = min(history.history['val_root_mean_squared_error'])
    return min_RMSE


# Round 1 of hyperparameter tuning
round1_count = 0
round1_increment = 10
round1_target = 1000

# Create a study object for Round 1 - or reload previous progress to continue running
round1_study = optuna.create_study(direction='minimize')
#round1_study = joblib.load('{}/hparam_optuna_round1_study.pkl'.format(folder_logs))

# Run for 500 trials in total, saving results every 10 trials
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
    joblib.dump(round1_study, '{}/hparam_optuna_round1_study.pkl'.format(folder_logs))
    joblib.dump(best, '{}/hparam_optuna_round1_best.pkl'.format(folder_logs))
    df.to_csv('{}/hparam_optuna_round1_df.csv'.format(folder_logs))
    
    # Update the study count (based on number of successful trials) & print status update
    round1_count = len(df.index)
    print('\n\nRound 1: {:,} trials completed (out of target {:,})\n\n'.format(round1_count, round1_target))


###############################################################################
# 4. Second round of hyperparameter tuning by Bayesian optimisation (Optuna)  #
###############################################################################

# Reload the best parameters found by the first round of hyperparameter tuning
round1_best = joblib.load('{}/hparam_optuna_round1_best.pkl'.format(folder_logs))

# Check those hyperparameter values here, to be hardcoded into the objective function below

# {'n_layers': 3,
#  'batch_norm': True,
#  'adam_learning_rate': 0.0015594062153786808,
#  'batch_size': 3074.609387457084,
#  'use_dropout': True,
#  'use_weight_decay': False,
#  'n_units_l1': 3074.077353995597,
#  'dropout_rate_l1': 0.11649378754625457,
#  'n_units_l2': 4089.1489715125836,
#  'dropout_rate_l2': 0.33403997505699057,
#  'n_units_l3': 1222.818661346792,
#  'dropout_rate_l3': 0.04978251399525324}

# Use the tf.keras loss function to compute the root mean squared error later
RMSE = RootMeanSquaredError()

# Define an early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Define an objective function to be minimised
def round2_objective(trial):
    
    # Sample hyperparameters which will be used throughout the model
    adam_lr = trial.suggest_loguniform('adam_learning_rate', 1e-6, 1e-2)       # Learning rate used for Adam optimiser
    batch_size = int(trial.suggest_loguniform('batch_size', 32, 4096))         # Mini-batch size
    
    # Instantiate a keras Sequence model
    model = Sequential()
    
    # Use the number of units optimised during the first round
    n_units_l1 = 3074
    n_units_l2 = 4089
    n_units_l3 = 1223
    
    # Use the dropout rates optimised during the first round
    dropout_rate_l1 = 0.11649378754625457
    dropout_rate_l2 = 0.33403997505699057
    dropout_rate_l3 = 0.04978251399525324
    
    # Layer 1
    model.add(Dense(n_units_l1, input_shape=(features_train_norm.shape[1],), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate_l1))
    
    # Layer 2
    model.add(Dense(n_units_l2, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate_l2))
    
    # Layer 3
    model.add(Dense(n_units_l3, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate_l3))
    
    # Final layer has a single unit & no activation
    model.add(Dense(1))
    # Compile the model - using MSE as the loss metric & with the Adam optimiser
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr), metrics=[RMSE])
    # Fit the model to the training data (features_train_norm & target_train)
    history = model.fit(features_train_norm, target_train, epochs=100, batch_size=batch_size, validation_data=(features_dev_norm, target_dev), callbacks=[early_stop], verbose=0)
    # Get the best validation RMSE
    min_RMSE = min(history.history['val_root_mean_squared_error'])
    return min_RMSE


# Round 2 of hyperparameter tuning
round2_count = 0
round2_increment = 5
round2_target = 250

# Create a study object for Round 2 - or reload previous progress to continue running
round2_study = optuna.create_study(direction='minimize')
#round2_study = joblib.load('{}/hparam_optuna_round2_study.pkl'.format(folder_logs))

# Run for 500 trials in total, saving results every 5 trials
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
    visualise_hparam_tuning_round2(df, best, tuning_round=2)
    
    # Save the optuna study object, dictionary of best-performing parameters, and dataframe of results so far
    joblib.dump(round2_study, '{}/hparam_optuna_round2_study.pkl'.format(folder_logs))
    joblib.dump(best, '{}/hparam_optuna_round2_best.pkl'.format(folder_logs))
    df.to_csv('{}/hparam_optuna_round2_df.csv'.format(folder_logs))
    
    # Update the study count (based on number of successful trials) & print status update
    round2_count = len(df.index)
    print('\n\nRound 2: {:,} trials completed (out of target {:,})\n\n'.format(round2_count, round2_target))


###############################################################################
# 5. Production ensemble runs, using final tuned model hyperparameters        #
###############################################################################

# Reload the best parameters found by each round of hyperparameter tuning
best1 = joblib.load('{}/hparam_optuna_round1_best.pkl'.format(folder_logs))
best2 = joblib.load('{}/hparam_optuna_round2_best.pkl'.format(folder_logs))

# Use the tf.keras loss function to compute the root mean squared error later
RMSE = RootMeanSquaredError()

# Build & train a series of different models, to explore potential value of ensemble approach (given stochastic elements)
for i in range(25):
    
    # Convert index number to zero-padded string, for easier filenaming
    i = str(i).zfill(2)
    
    # Proceed if corresponding model doesn't yet exist
    if not os.path.exists('{}/densenet_{}.h5'.format(folder_models, i)):
        
        # Define callbacks that will be used by all models, including model checkpoint to save only the best model
        callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-5, verbose=0),
                     EarlyStopping(monitor='val_loss', patience=50, verbose=0),
                     ModelCheckpoint('{}/densenet_{}.h5'.format(folder_models, i), monitor='val_loss', save_best_only=True, verbose=0)]
        
        # Build & train a model, using early stopping (patience=50) and saving the best model
        print('Training model {}...'.format(i))
        
        # Set up model based on the two rounds of hyperparameter tuning
        model = Sequential()
        # Layer 1
        model.add(Dense(best1['n_units_l1'], input_shape=(features_train_norm.shape[1],), kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(best1['dropout_rate_l1']))
        # Layer 2
        model.add(Dense(best1['n_units_l2'], kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(best1['dropout_rate_l2']))
        # Layer 3
        model.add(Dense(best1['n_units_l3'], kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(best1['dropout_rate_l3']))
        # Final layer has a single unit & no activation
        model.add(Dense(1))
        # Compile the model - using MSE as the loss metric & with the Adam optimiser
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=best2['adam_learning_rate']), metrics=[RMSE])
        # Fit the model to the training data (features_train_norm & target_train), without applying early stopping
        history = model.fit(features_train_norm, target_train, batch_size=int(best2['batch_size']), epochs=1000, validation_data=(features_dev_norm, target_dev), callbacks=callbacks, verbose=0)
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = history.epoch
        print(' - Lowest validation RMSE = {:.4f}'.format(history_df['val_root_mean_squared_error'].min()))
        history_df.to_csv('{}/densenet_traininghistory_{}.csv'.format(folder_logs, i))
        del model, history, history_df


###############################################################################
# 6. Use each trained model to generate predictions for each data subset      #
###############################################################################

# Read the list of selected features into memory (based on subset chosen during SFFS with Random Forest models)
with open('{}/feature_selection_final_23_features.p'.format(folder_logs_rf), 'rb') as f:
    selected_features = pickle.load(f)

# Import & process the test dataset
data_test = pd.read_csv('{}/Input1D_Test.csv'.format(folder_input_1D))
target_test = np.array(data_test['diff'].copy().values).reshape((len(data_test.index),1))
features_test = data_test[selected_features].copy()

# Check for any missing data
target_test[(target_test==-9999)|np.isnan(target_test)].sum()
features_test[(features_test==-9999)|features_test.isnull()].sum()

# Normalise all sets of input features (train, dev & test)
features_test_norm = normalise_df(features_test, stats_train)

# Initialise a dataframe that will hold performance metrics for all densenet models trained (as well as metrics for the initial error & baseline correction)
df = pd.DataFrame(columns=['Type','Dataset','RMSE'])

# Calculate the baseline predictions possible (simply the mean of the corrections observed from the training data)
baseline_predictions = np.nanmean(target_train)

# Build a dictionary to hold residuals for each dataset
residuals = {dataset:{model:None for model in ['initial','baseline']} for dataset in ['train','dev','test']}

# Loop through each dataset, populating dataframe with initial RMSE & naive RMSE for each
for (dataset, features_norm, target) in zip(['train','dev','test'], [features_train_norm, features_dev_norm, features_test_norm], [target_train, target_dev, target_test]):
    # Calculate initial RMSE, based on target data
    MSE_initial = np.nanmean(np.square(target))
    RMSE_initial = np.sqrt(MSE_initial)
    df = df.append({'Type':'initial', 'Dataset':dataset, 'RMSE':RMSE_initial}, ignore_index=True)
    # Calculate RMSE after applying a baseline correction (subtracting mean difference observed from training data)
    MSE_baseline = np.nanmean(np.square(target - baseline_predictions))
    RMSE_baseline = np.sqrt(MSE_baseline)
    df = df.append({'Type':'baseline', 'Dataset':dataset, 'RMSE':RMSE_baseline}, ignore_index=True)
    # Save residual arrays in the dictionary initialised previously
    residuals[dataset]['initial'] = target.flatten()
    residuals[dataset]['baseline'] = (target - baseline_predictions).flatten()

# Add to this dataframe the RMSE scores returned for each of the available densenet models, saving predictions for each
for i in range(25):
    # Convert index number to zero-padded string, for easier filenaming
    i = str(i).zfill(2)
    # Process if corresponding model is available
    if os.path.exists('{}/densenet_{}.h5'.format(folder_models, i)):
        # Load convnet model
        model = tf.keras.models.load_model('{}/densenet_{}.h5'.format(folder_models, i))
        # Loop through each available dataset, as above
        for (dataset, features_norm, target) in zip(['train','dev','test'], [features_train_norm, features_dev_norm, features_test_norm], [target_train, target_dev, target_test]):
            # Evalute model for RMSE score
            loss_densenet, RMSE_densenet = model.evaluate(features_norm, target, verbose=0)
            df = df.append({'Type':'densenet_{}'.format(i), 'Dataset':dataset, 'RMSE':RMSE_densenet}, ignore_index=True)
            print('Model {}: {} data: RMSE = {:.4f} m'.format(i, dataset, RMSE_densenet))
            # Generate predictions using the densenet, in order to save residuals
            predictions = model.predict(features_norm)
            residuals[dataset]['densenet_{}'.format(i)] = (target - predictions).flatten()
            # Store predictions for further analysis (ensemble approach)
            np.save('{}/predictions/densenet_{}_{}_prediction.npy'.format(folder_results, i, dataset), predictions)

# Save metric results for individual models to a CSV file, for easy import & comparison later
df.to_csv('{}/densenet_metrics_models.csv'.format(folder_results))

# Save residuals to a pickle object, for later import and comparison
pickle.dump(residuals, open('{}/densenet_residuals_models.p'.format(folder_results), 'wb'))


###############################################################################
# 7. Confirm value of ensemble approach (predictions from multiple models)    #
###############################################################################

# Define a list of labels for the models/results available
model_idxs = [i for i in range(50) if os.path.exists('{}/densenet_{}.h5'.format(folder_models, str(i).zfill(2)))]

# Specify number of models/results available
n_models = len(model_idxs)

# Re-open the metrics summary dataframe, if no longer in memory
df_metrics = pd.read_csv('{}/densenet_metrics_models.csv'.format(folder_results))

# Load all predictions into a single array
predictions_array = np.array([np.load('{}/predictions/densenet_{}_test_prediction.npy'.format(folder_results, str(model_idx).zfill(2))) for model_idx in range(25)])

# Now loop through a range of ensemble sizes
for size in [1,2,3,4,5,6,20,21,22,23,24,25]:  # Just look at each end of the scale, to confirm expected trend
    # Only process if that result doesn't yet exist (very time-consuming)
    if not os.path.exists('{}/densenet_metrics_ensembles_{}models.csv'.format(folder_results, str(size).zfill(2))):
        # Initialise a dataframe to save results for ensembles of that particular size
        df = pd.DataFrame(columns=['Size','Ensemble','RMSE'])
        # Get a list of possible model combinations, of the appropriate ensemble size
        ensembles_idx = list(combinations(model_idxs, size))
        print('Processing {}-model ensembles ({} combinations)...'.format(size, len(ensembles_idx)))
        # Loop through each of the possible combinations identified
        for ensemble_idx in ensembles_idx:
            ensemble_label = '_'.join([str(e).zfill(2) for e in ensemble_idx])
            # Get the mean prediction using all individual model predictions included in the ensemble
            predictions_mean = np.mean(predictions_array[ensemble_idx,:,:], axis=0)
            # Evaluate the RMSE using this ensemble prediction & append RMSE to overall list
            ensemble_RMSE = np.sqrt(np.nanmean(np.square(target_test - predictions_mean)))
            # Update the dataframe
            df = df.append({'Size':size, 'Ensemble':ensemble_label, 'RMSE':ensemble_RMSE}, ignore_index=True)
        # Save ensemble dataframe to disk
        df.to_csv('{}/densenet_metrics_ensembles_{}models.csv'.format(folder_results, str(size).zfill(2)))
        print('{}-model ensemble: mean test RMSE = {:.4f}m'.format(size, df['RMSE'].mean()))

# Alternative approach 1: Check test score using only the model which returned the lowest validation score
min_dev_model_idx = df_metrics['RMSE'].loc[(df_metrics['Dataset']=='dev') & (df_metrics['Type'].str.startswith('densenet', na=False)) & (~df_metrics['Type'].str.endswith('ensemble', na=False))].argmin()
min_dev_model = df_metrics['Type'].loc[(df_metrics['Dataset']=='dev') & (df_metrics['Type'].str.startswith('densenet', na=False)) & (~df_metrics['Type'].str.endswith('ensemble', na=False))].iloc[min_dev_model_idx]
min_dev_prediction = np.load('{}/predictions/{}_test_prediction.npy'.format(folder_results, min_dev_model))
min_dev_RMSE = np.sqrt(np.nanmean(np.square(target_test - min_dev_prediction)))

# Alternative approach 2: Use all available models to generate an average prediction, but weighted based on their respective validation scores
dev_RMSEs = [df_metrics['RMSE'].loc[(df_metrics['Dataset']=='dev')&(df_metrics['Type']=='densenet_{}'.format(str(i).zfill(2)))].item() for i in range(25)]
weights = [1/dev_RMSE for dev_RMSE in dev_RMSEs]
total_weight = np.sum(weights)
predictions_weighted_mean = np.sum([weight * np.load('{}/predictions/densenet_{}_test_prediction.npy'.format(folder_results, str(m).zfill(2))) for weight, m in zip(weights, range(25))], axis=0) / total_weight
predictions_weighted_RMSE = np.sqrt(np.nanmean(np.square(target_test - predictions_weighted_mean)))

# Initialise a figure to visualise impact of ensemble approach, as well as two alternatives
fig, axes = plt.subplots(nrows=2, figsize=(9,9))
# Loop through a range of ensemble sizes
#for size in range(1, 10):
for size in [1,2,3,4,5,6,20,21,22,23,24,25]:  # Initially just look at each end of the scale, to confirm expected trend
    # Read results for ensembles of that size into a dataframe
    df = pd.read_csv('{}/densenet_metrics_ensembles_{}models.csv'.format(folder_results, str(size).zfill(2)))
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
fig.savefig('{}/densenet_RMSE_ensembles.png'.format(folder_fig), dpi=300)
plt.close()

# Read the dataframe of model metrics back into memory, to add a final set of results (using 25-model ensemble)
df_metrics = pd.read_csv('{}/densenet_metrics_models.csv'.format(folder_results))

# Read the dictionary of residuals back into memory too, to update it with ensemble results
with open('{}/densenet_residuals_models.p'.format(folder_results), 'rb') as f:
    residuals = pickle.load(f)

# Loop through each dataset, populating dataframe with RMSE derived from 25-model ensemble predictions
for (dataset, target) in zip(['train','dev','test'], [target_train, target_dev, target_test]):
    # Get the 25-model ensemble mean prediction for that dataset
    predictions_mean = np.mean([np.load('{}/predictions/densenet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), dataset)) for i in range(25)], axis=0)
    # Evaluate RMSE for that ensemble prediction, compared to target
    ensemble_RMSE = np.sqrt(np.nanmean(np.square(target - predictions_mean)))
    # Update the metric summary dataframe
    df_metrics = df_metrics.append({'Type':'densenet_ensemble', 'Dataset':dataset, 'RMSE':ensemble_RMSE}, ignore_index=True)
    print('Ensemble: {} data: RMSE = {:.4f} m'.format(dataset, ensemble_RMSE))
    # Save residuals to same dictionary as before
    residuals[dataset]['densenet_ensemble'] = (target - predictions_mean).flatten()
    # Save the ensemble prediction array as a .npy file for easy import later
    np.save('{}/predictions/densenet_ensemble_{}_prediction.npy'.format(folder_results, dataset), predictions_mean)

# Save new versions of the metrics summary & residuals dict
df_metrics.to_csv('{}/densenet_metrics_models.csv'.format(folder_results))
pickle.dump(residuals, open('{}/densenet_residuals_models.p'.format(folder_results), 'wb'))


###############################################################################
# 8. Compare overall RMSE reduction efficacy of all models considered so far  #
###############################################################################

# Read in metrics summaries for random forest (RF) and densely-connected network (densenet), for comparison
df_rf = pd.read_csv('{}/rf_23features_250trees_metrics_by_dataset.csv'.format(folder_results.replace('densenet','rf')))
df = pd.read_csv('{}/densenet_metrics_models.csv'.format(folder_results))

# Generate summary plots for each of the three dataset results
for i, dataset in enumerate(['train','dev','test']):
    # Extract appropriate results from the new dataframe
    RMSE_initial = df[(df['Dataset']==dataset)&(df['Type']=='initial')]['RMSE'].values.item()
    RMSE_baseline = df[(df['Dataset']==dataset)&(df['Type']=='baseline')]['RMSE'].values.item()
    RMSE_densenet = df[(df['Dataset']==dataset)&(df['Type']=='densenet_ensemble')]['RMSE'].values.item()
    # Read results from the Random Forest data frame too
    RMSE_rf = df_rf[(df_rf['Dataset']==dataset)&(df_rf['Type']=='rf')]['RMSE'].values.item()
    # Set up the figure
    fig, axes = plt.subplots(figsize=(9,4.5))
    axes.bar([0,1,2,3], [RMSE_initial, RMSE_baseline, RMSE_rf, RMSE_densenet], color=dataset_colours[dataset], alpha=0.5)
    axes.set_xticks([0,1,2,3])
    axes.yaxis.set_tick_params(length=0)
    axes.set_xticklabels(['Initial','Baseline\ncorrection','Random Forest\ncorrection','Densenet\ncorrection'])
    axes.set_ylabel('Root Mean Squared Error [m]')
    axes.grid(axis='y', which='major', color='dimgrey', alpha=0.25)
    [axes.spines[edge].set_visible(False) for edge in ['left','top','right']]
    # Add a horizontal line showing the initial error & a label
    axes.axhline(y=RMSE_initial, color=dataset_colours[dataset], linestyle='dashed', alpha=0.5)
    axes.annotate('{:.3f}m'.format(RMSE_initial), xy=(0, RMSE_initial), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
    # Add labels indicating improvement achieved by each method
    for j, RMSE_new in enumerate([RMSE_baseline, RMSE_rf, RMSE_densenet]):
        # Add downward arrow from initial RMSE to improved RMSE
        axes.annotate('', xy=(j+1, RMSE_new), xytext=(j+1, RMSE_initial), arrowprops=dict(arrowstyle='->'))
        # Add label indicating new RMSE and the percentage improvement it equates to
        improvement_percentage = (RMSE_new-RMSE_initial)/RMSE_initial * 100.
        axes.annotate('{:.3f}m ({:.1f}%)'.format(RMSE_new, improvement_percentage), xy=(j+1, RMSE_new), xytext=(0, -5), textcoords='offset points', ha='center', va='top')
    axes.set_title('Performance on {} dataset'.format('validation' if dataset=='dev' else dataset))
    fig.tight_layout()
    fig.savefig('{}/densenet_results_{}.png'.format(folder_fig, dataset), dpi=300)
    plt.close()


###############################################################################
# 9. Generate intact prediction arrays for each zone (for mapping/GeoTIFFs)   #
###############################################################################

# Define list of zones to be processed (separate LiDAR coverage areas)
zones = ['MRL18_WPE', 'MRL18_WVL', 'MRL18_WKW', 'MRL18_FGA', 'TSM17_STA', 'TSM17_LDM', 'TSM16_ATG', 'TSM17_GLB']

# Remember to normalise input data based on training dataset only, for consistency

# Read the list of selected features back into memory
with open('{}/feature_selection_final_23_features.p'.format(folder_logs_rf), 'rb') as f:
    selected_features = pickle.load(f)

# Import processed CSVs into dataframes
data_train = pd.read_csv('{}/Input1D_Train.csv'.format(folder_input_1D))

# Extract features from training data, to calculate normalisation parameters
features_train = data_train[selected_features].copy()

# Check for any missing data
features_train[(features_train==-9999)|features_train.isnull()].sum()

# Save the overall statistics based on the training data
stats_train = features_train.describe()
stats_train = stats_train.transpose()

# Once these are calculated, the training data can be deleted (very large)
del features_train, data_train

# 9a. Loop through all zones, import data, and generate prediction arrays from each model (saving as .npy files)
for zone in zones:
    
    # Open CSV of data for that zone (dropping any rows with no_data) & filter for selected features only
    print('Processing {} zone...'.format(zone))
    df = pd.read_csv('{}/Input1D_ByZone_{}.csv'.format(folder_input_1D, zone))
    features = df[selected_features].copy()
    
    # Normalise the feature data based on the training data, as done before, then delete unnormalised features
    features_norm = normalise_df(features, stats_train)
    del df, features
    
    # Loop through all individual models available, generating predictions for that zone
    for i in range(25):
        
        # Only proceed if corresponding prediction doesn't yet exist
        if not os.path.exists('{}/predictions/densenet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), zone)):
            
            # Load that densenet model, generate predictions using that model, and save it as a numpy array
            model = tf.keras.models.load_model('{}/densenet_{}.h5'.format(folder_models, str(i).zfill(2)))
            model_prediction = model.predict(features_norm)
            np.save('{}/predictions/densenet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), zone), model_prediction)
            del model, model_prediction

# 9b. Loop through all zones, generating the ensemble prediction array for each
for zone in zones:
    print('Processing {} zone...'.format(zone))
    
    # Calculate ensemble prediction for that zone (mean of predictions from individual models)
    ensemble_prediction = np.nanmean([np.load('{}/predictions/densenet_{}_{}_prediction.npy'.format(folder_results, str(i).zfill(2), zone)) for i in range(25)], axis=0)
    
    # Save ensemble prediction to a .npy file
    np.save('{}/predictions/densenet_ensemble_{}_prediction.npy'.format(folder_results, zone), ensemble_prediction)