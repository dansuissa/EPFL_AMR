import pickle
import pandas as pd
import numpy as np
import time
import optuna
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense, Add, Input, MaxPooling1D
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from tslearn.preprocessing import TimeSeriesResampler
from sktime.transformations.panel.rocket import MultiRocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.hybrid import HIVECOTEV2
from sktime.transformations.panel.catch22 import Catch22

# InceptionTime Model
class Classifier_INCEPTION:
    def __init__(self, input_shape, nb_classes, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32
        self.model = self.build_model(input_shape, nb_classes)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size, kernel_size=1, padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list = [Conv1D(filters=self.nb_filters, kernel_size=ks, strides=stride, padding='same', activation=activation, use_bias=False)(input_inception) for ks in kernel_size_s]

        max_pool_1 = MaxPooling1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        conv_6 = Conv1D(filters=self.nb_filters, kernel_size=1, padding='same', activation=activation, use_bias=False)(max_pool_1)
        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        return Activation('relu')(x)

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)
        x = Add()([shortcut_y, out_tensor])
        return Activation('relu')(x)

    def build_model(self, input_shape, nb_classes):
        input_layer = Input(input_shape)
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)
        output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

# Convert a list of time series to a nested pandas DataFrame (for MultiROCKET)
def convert_to_nested_dataframe(time_series_list):
    return pd.DataFrame({"time_series": [pd.Series(ts.squeeze()) for ts in time_series_list]})

# Function to resample time series using DBA
def resample_time_series_dba(X_train, X_test, target_length):
    resampler = TimeSeriesResampler(sz=target_length)
    X_train_resampled = resampler.fit_transform([np.array(ts) for ts in X_train])
    X_test_resampled = resampler.transform([np.array(ts) for ts in X_test])
    return X_train_resampled, X_test_resampled

# Extract features from your dataset
def extract_features_VIAB(docs_meas, docs_analyzed, environment, strains):
    processed_docs = []
    for data_meas, data_analyzed in zip(docs_meas, docs_analyzed):
        try:
            bacteria_or_object = data_meas.get('bacteria', data_meas.get('object', None))
            if bacteria_or_object in strains and data_meas['env'] in environment:
                transmission_data = data_meas['data']['transmission']
                total_length = sum(state['indices'][1] - state['indices'][0] for state in data_analyzed.get('classification', []))

                for state in data_analyzed.get('classification', []):
                    if state['type'] == 'trapping':
                        start_idx, end_idx = state['indices']
                        real_start_idx = total_length - (end_idx - start_idx)
                        real_end_idx = total_length

                        if real_start_idx < 0:
                            real_start_idx = 0
                        if real_end_idx > len(transmission_data):
                            real_end_idx = len(transmission_data)

                        time_series = transmission_data[real_start_idx:real_end_idx] or transmission_data  # Fallback if empty
                        features = {'time_series': time_series}
                        features['dead'] = 1 if 'dead' in data_analyzed or data_analyzed.get('state') == 'dead' else 0
                        processed_docs.append(features)

        except KeyError as e:
            print(f"Missing key in document: {str(e)}")

    return pd.DataFrame(processed_docs)
# Train and evaluate MultiROCKET model with ridge 
def run_multi_rocket(X_train_resampled, X_test_resampled, y_train, y_test):
    X_train_nested = convert_to_nested_dataframe(X_train_resampled)
    X_test_nested = convert_to_nested_dataframe(X_test_resampled)

    multi_rocket = MultiRocket(num_kernels=50_000)
    X_train_transformed = multi_rocket.fit_transform(X_train_nested)
    X_test_transformed = multi_rocket.transform(X_test_nested)

    classifier = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0])
    classifier.fit(X_train_transformed, y_train)

    y_pred = classifier.predict(X_test_transformed)
    print("MultiROCKET Classification Report with ridge:")
    print(classification_report(y_test, y_pred, target_names=['Alive', 'Dead']))
"""
# MultiROCKET with XGBoost
def run_multi_rocket(X_train_resampled, X_test_resampled, y_train, y_test):
    X_train_nested = convert_to_nested_dataframe(X_train_resampled)
    X_test_nested = convert_to_nested_dataframe(X_test_resampled)

    multi_rocket = MultiRocket(num_kernels=50_000)
    X_train_transformed = multi_rocket.fit_transform(X_train_nested)
    X_test_transformed = multi_rocket.transform(X_test_nested)

    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 0.7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        }

        clf = xgb.XGBClassifier(**param, verbosity=0)
        clf.fit(X_train_transformed, y_train)
        y_pred = clf.predict(X_test_transformed)
        return accuracy_score(y_test, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=300)

    print(f"Best trial accuracy: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_params}")

    best_params = study.best_params
    best_clf = xgb.XGBClassifier(**best_params)
    best_clf.fit(X_train_transformed, y_train)

    y_pred = best_clf.predict(X_test_transformed)
    print("XGBoost Classification Report (Optimized with Optuna):")
    print(classification_report(y_test, y_pred, target_names=['Alive', 'Dead']))
"""
# Train and evaluate InceptionTime model
def run_inception_time(X_train_resampled, X_test_resampled, y_train, y_test, target_length):
    X_train_resampled = np.array(X_train_resampled).reshape(len(X_train_resampled), target_length, 1)
    X_test_resampled = np.array(X_test_resampled).reshape(len(X_test_resampled), target_length, 1)

    nb_classes = 2  # Alive or dead
    inception_model = Classifier_INCEPTION(input_shape=(target_length, 1), nb_classes=nb_classes)
    inception_model.model.fit(X_train_resampled, np.array(y_train), epochs=50, batch_size=32, validation_data=(X_test_resampled, np.array(y_test)))

    y_pred = np.argmax(inception_model.model.predict(X_test_resampled), axis=-1)
    print("InceptionTime Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Alive', 'Dead']))

# Train and evaluate HIVE-COTE 2.0 model
def run_hivecote(X_train_resampled, X_test_resampled, y_train, y_test):
    X_train_nested = convert_to_nested_dataframe(X_train_resampled)
    X_test_nested = convert_to_nested_dataframe(X_test_resampled)

    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    hivecote = HIVECOTEV2()
    print("Training HIVE-COTE 2.0...")
    start_time = time.time()
    hivecote.fit(X_train_nested, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    y_pred = hivecote.predict(X_test_nested)
    print("HIVE-COTE 2.0 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Alive', 'Dead']))

# Main function
def main():
    print("Loading data from pickle files...")
    with open('docs_analysed.pkl', 'rb') as f:
        docs_analyzed_VIAB = pickle.load(f)
    with open('docs_meas.pkl', 'rb') as f:
        docs_meas_VIAB = pickle.load(f)

    strains_allowed = ['ecb', 'ECB']
    env1 = ['0.0045%NaCl', '0.0045%', '0.00225%', '0.00225%NaCl', 'h20', 'h2o', 'H2O', '0.009%', '0.009%NaCl', 'h2o0009NaCl']

    extracted_features = extract_features_VIAB(docs_meas_VIAB, docs_analyzed_VIAB, env1, strains_allowed)
    print(extracted_features.head())

    num_dead = extracted_features[extracted_features['dead'] == 1].shape[0]
    num_alive = extracted_features[extracted_features['dead'] == 0].shape[0]
    print(f"Number of dead samples: {num_dead}")
    print(f"Number of alive samples: {num_alive}")

    X = extracted_features['time_series'].tolist()
    y = extracted_features['dead']

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in strat_split.split(X, y):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    target_length = int(np.median([len(ts) for ts in X_train]))
    print(f"Resampling all time series to length: {target_length}")

    X_train_resampled, X_test_resampled = resample_time_series_dba(X_train, X_test, target_length)

    # Choose the model based on user input
    print("Choose model to run: 1 for MultiROCKET, 2 for InceptionTime, 3 for HIVE-COTE 2.0")
    model_choice = 1  # Set default choice; you can change this to allow user input
    if model_choice == 1:
        run_multi_rocket(X_train_resampled, X_test_resampled, y_train, y_test)
    elif model_choice == 2:
        run_inception_time(X_train_resampled, X_test_resampled, y_train, y_test, target_length)
    elif model_choice == 3:
        run_hivecote(X_train_resampled, X_test_resampled, y_train, y_test)
    else:
        print("Invalid model choice!")

if __name__ == '__main__':
    main()
