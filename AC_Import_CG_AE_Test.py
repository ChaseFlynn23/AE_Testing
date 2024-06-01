import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, activations
from keras_tuner import BayesianOptimization
from keras_tuner import Objective

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

def preprocessing_alt(wt, mutant):
    wt_label = np.zeros(len(wt))
    mutant_label = np.ones(len(mutant))

    # Create label dataframes with indices
    wt_label_df = pd.DataFrame({'class': wt_label})
    mutant_label_df = pd.DataFrame({'class': mutant_label}, index=range(40000, 40000 + len(mutant)))

    # Concatenate data frames and label dataframes
    X_train_full = pd.concat([wt, mutant])
    y_train_full_df = pd.concat([wt_label_df, mutant_label_df])

    # Normalize training data
    X_train_full = X_train_full.div(100)  # Adjust as necessary

    # Separate training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full_df, stratify=y_train_full_df['class'], test_size=0.2)

    print(X_train.shape)
    print(X_valid.shape)
    print(y_train.shape)
    print(y_valid.shape)

    return X_train, X_valid, y_train, y_valid

class AutoencoderTuner:
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.directory = self._create_directory_for_trials()
        self.tuner = None
        
    def _create_directory_for_trials(self):
        base_dir = "AE_CG_Tuning"
        trial_num = 1
        while os.path.exists(f"{base_dir}_Trial_{trial_num}"):
            trial_num += 1
        path = f"{base_dir}_Trial_{trial_num}"
        os.makedirs(path, exist_ok=True)
        return path

    def build_model(self, hp):
        model = keras.Sequential()
        encoder_layers = []

        # Determine the number of layers in the encoder
        num_layers = hp.Int('num_layers', 2, 5)
        # Ensure that there's an odd number of layers to have a middle layer
        if num_layers % 2 == 0:
            num_layers += 1

        # Encoder Part
        for i in range(num_layers // 2):  # Integer division to get half the layers (rounded down)
            units = hp.Int(f'encoder_nodes_{i}', min_value=32, max_value=652, step=320)
            activation_choice = hp.Choice(f'encoder_activation_{i}', ['relu', 'tanh', 'sigmoid', 'leakyrelu'])

            encoder_layers.append((units, activation_choice))

            model.add(layers.Dense(units=units, activation=activation_choice if activation_choice != 'leakyrelu' else None))
            if activation_choice == 'leakyrelu':
                model.add(layers.LeakyReLU(alpha=0.01))

        # Latent Layer with 2 nodes
        model.add(layers.Dense(units=2, activation='relu'))

        # Decoder Part - mirroring the encoder
        for units, activation_choice in reversed(encoder_layers):
            model.add(layers.Dense(units=units, activation=activation_choice if activation_choice != 'leakyrelu' else None))
            if activation_choice == 'leakyrelu':
                model.add(layers.LeakyReLU(alpha=0.01))

        # Output layer
        model.add(layers.Dense(units=self.X_train.shape[1], activation='sigmoid'))

        # Optimizer Configuration with Learning Rate
        optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        if optimizer_choice == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=hp.Choice('loss', ['mse', 'binary_crossentropy', 'mae'])
        )

        return model

    def run_tuning(self, max_trials=5, executions_per_trial=1, num_initial_points=5, epochs=10):
        self.tuner = BayesianOptimization(
            self.build_model,
            objective=Objective("val_loss", direction="min"),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            num_initial_points=num_initial_points,
            directory=self.directory,
            project_name='AE_CG_Trial',
            overwrite=True
        )
        self.tuner.search(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_valid, self.y_valid), batch_size=153)

    # New method to get best hyperparameters
    def get_best_hyperparameters(self, num_trials=1):
        if self.tuner:
            return self.tuner.get_best_hyperparameters(num_trials)
        else:
            print("Tuning has not been run yet.")
            return None