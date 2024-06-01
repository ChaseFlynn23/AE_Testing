import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from kerastuner.tuners import BayesianOptimization

filtered_data_folder_name = 'XGB_Filtered_Common_Positions'

def load_and_concatenate(file_prefix, var_names_prefix):
    files = [f'{file_prefix}_{i}f.csv' for i in range(2, 52)]
    var_names = [f'{var_names_prefix}_{i}f' for i in range(2, 52)]
    dataframes = []
    for var, file in zip(var_names, files):
        filepath = f'{filtered_data_folder_name}/{file}'
        df = pd.read_csv(filepath, index_col='Index')
        dataframes.append(df)
    return pd.concat(dataframes, axis=1)

def preprocessing(wt, mutant):
    wt_label = np.zeros(len(wt))
    mutant_label = np.ones(len(mutant))
    X_train_full = pd.concat([wt.reset_index(), mutant.reset_index()])
    y_train_full = np.concatenate((wt_label, mutant_label))
    X_train_full = X_train_full.drop(columns = 'Index')
    X_train_full= X_train_full.div(100)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.2)
    return X_train, X_valid, y_train, y_valid

best_overall = {
    'val_loss': float('inf'),
    'hyperparameters': None,
    'step': None,
    'details': {}
}

def update_best_overall(tuner, step, additional_details=None):
    global best_overall
    best_hp = tuner.get_best_hyperparameters()[0]
    best_loss = tuner.oracle.get_best_trials(1)[0].score
    if best_loss < best_overall['val_loss']:
        best_overall['val_loss'] = best_loss
        best_overall['hyperparameters'] = best_hp
        best_overall['step'] = step
        if additional_details:
            best_overall['details'][step] = additional_details

def build_autoencoder(hp, input_shape):
    encoder_input = keras.Input(shape=input_shape)
    x = encoder_input
    encoder_layers = []
    for i in range(hp.Int('num_layers', 2, 6)):
        layer_size = hp.Int(f'nodes_{i}', min_value=32, max_value=992, step=16)
        x = layers.Dense(layer_size, activation=keras.layers.LeakyReLU(alpha=0.01))(x)
        encoder_layers.append(layer_size)
    latent_space = layers.Dense(2)(x)
    x = latent_space
    for layer_size in reversed(encoder_layers):
        x = layers.Dense(layer_size, activation=keras.layers.LeakyReLU(alpha=0.01))(x)
    decoder_output = layers.Dense(input_shape[0], activation='linear')(x)
    autoencoder = models.Model(encoder_input, decoder_output)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.005), loss='mse')
    return autoencoder

def build_autoencoder_for_tuning(hp, best_num_layers, best_nodes_per_layer, input_shape):
    encoder_input = keras.Input(shape=input_shape)
    x = encoder_input
    for i in range(best_num_layers):
        layer_size = best_nodes_per_layer[i]
        x = layers.Dense(layer_size, activation=keras.layers.LeakyReLU(alpha=0.01))(x)
    latent_space = layers.Dense(2)(x)
    for layer_size in reversed(best_nodes_per_layer):
        x = layers.Dense(layer_size, activation=keras.layers.LeakyReLU(alpha=0.01))(x)
    decoder_output = layers.Dense(input_shape[0], activation='linear')(x)
    autoencoder = models.Model(encoder_input, decoder_output)
    lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')
    return autoencoder

def build_autoencoder_for_activation_tuning(hp, best_num_layers, best_nodes_per_layer, best_learning_rate, input_shape):
    encoder_input = keras.Input(shape=input_shape)
    x = encoder_input
    activation_functions = []
    for i in range(best_num_layers):
        layer_size = best_nodes_per_layer[i]
        activation_choice = hp.Choice(f'encoder_activation_{i}', values=['relu', 'sigmoid', 'tanh', 'LeakyReLU'])
        activation_functions.append(activation_choice)
        if activation_choice == 'LeakyReLU':
            x = layers.Dense(layer_size)(x)
            x = layers.LeakyReLU(alpha=0.01)(x)
        else:
            x = layers.Dense(layer_size, activation=activation_choice)(x)
    latent_space = layers.Dense(2)(x)
    for i, layer_size in enumerate(reversed(best_nodes_per_layer)):
        activation_choice = activation_functions[-(i + 1)]
        if activation_choice == 'LeakyReLU':
            x = layers.Dense(layer_size)(x)
            x = layers.LeakyReLU(alpha=0.01)(x)
        else:
            x = layers.Dense(layer_size, activation=activation_choice)(x)
    decoder_output = layers.Dense(input_shape[0], activation='linear')(x)
    autoencoder = models.Model(encoder_input, decoder_output)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=best_learning_rate), loss='mse')
    return autoencoder

def build_autoencoder_for_final_tuning(hp, best_num_layers, best_nodes_per_layer, best_activations, best_learning_rate, input_shape):
    encoder_input = keras.Input(shape=input_shape)
    x = encoder_input
    for i in range(best_num_layers):
        layer_size = best_nodes_per_layer[i]
        activation_choice = best_activations[i]
        if activation_choice == 'LeakyReLU':
            x = layers.Dense(layer_size)(x)
            x = layers.LeakyReLU(alpha=0.01)(x)
        else:
            x = layers.Dense(layer_size, activation=activation_choice)(x)
    latent_space = layers.Dense(2)(x)
    for i, layer_size in enumerate(reversed(best_nodes_per_layer)):
        activation_choice = best_activations[-(i + 1)]
        if activation_choice == 'LeakyReLU':
            x = layers.Dense(layer_size)(x)
            x = layers.LeakyReLU(alpha=0.01)(x)
        else:
            x = layers.Dense(layer_size, activation=activation_choice)(x)
    decoder_output = layers.Dense(input_shape[0], activation='linear')(x)
    autoencoder = models.Model(encoder_input, decoder_output)
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    loss_function_choice = hp.Choice('loss_function', values=['mse', 'binary_crossentropy', 'mae'])
    if optimizer_choice == 'adam':
        optimizer = optimizers.Adam(learning_rate=best_learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = optimizers.SGD(learning_rate=best_learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=best_learning_rate)
    autoencoder.compile(optimizer=optimizer, loss=loss_function_choice)
    return autoencoder
