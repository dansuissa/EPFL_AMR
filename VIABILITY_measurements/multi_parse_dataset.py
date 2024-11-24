import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense, Add, Input, MaxPooling1D
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from tslearn.preprocessing import TimeSeriesResampler
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

# Function to resample time series
def resample_time_series_dba(X_train, X_test, target_length):
    resampler = TimeSeriesResampler(sz=target_length)
    X_train_resampled = resampler.fit_transform([np.array(ts) for ts in X_train])
    X_test_resampled = resampler.transform([np.array(ts) for ts in X_test])
    return X_train_resampled, X_test_resampled

# Run InceptionTime model
def run_inception_time(X_train_resampled, X_test_resampled, y_train, y_test, target_length):
    X_train_resampled = np.array(X_train_resampled).reshape(len(X_train_resampled), target_length, 1)
    X_test_resampled = np.array(X_test_resampled).reshape(len(X_test_resampled), target_length, 1)

    nb_classes = 2  # Alive or dead
    inception_model = Classifier_INCEPTION(input_shape=(target_length, 1), nb_classes=nb_classes)
    inception_model.model.fit(
        X_train_resampled,
        np.array(y_train),
        epochs=10,  # Adjust epochs as needed
        batch_size=32,
        validation_data=(X_test_resampled, np.array(y_test)),
        verbose=0,
    )

    y_pred = np.argmax(inception_model.model.predict(X_test_resampled), axis=-1)
    accuracy = np.mean(y_pred == np.array(y_test))
    return accuracy

# Main function to evaluate accuracy across increasing dataset sizes
def evaluate_and_plot_accuracy():
    # Load data from pickle files
    with open('docs_analysed.pkl', 'rb') as f:
        docs_analyzed_VIAB = pickle.load(f)
    with open('docs_meas.pkl', 'rb') as f:
        docs_meas_VIAB = pickle.load(f)

    # Data preparation
    strains_allowed = ['staphxylosus', 'staph', 'ecb', 'ECB']
    env1 = ['0.0045%NaCl', '0.0045%', '0.00225%', '0.00225%NaCl', 'h20', 'h2o', 'H2O', '0.009%', '0.009%NaCl', 'h2o0009NaCl']
    extracted_features = extract_features_VIAB(docs_meas_VIAB, docs_analyzed_VIAB, env1, strains_allowed)
    X = extracted_features['time_series'].tolist()
    y = extracted_features['dead']

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in strat_split.split(X, y):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    target_length = int(np.median([len(ts) for ts in X_train]))
    X_train_resampled, X_test_resampled = resample_time_series_dba(X_train, X_test, target_length)

    # Initialize variables for tracking dataset sizes and accuracies
    accuracies = []
    sizes = []

    # Evaluate model on dataset subsets from 10% to 100%
    for size in range(10, 105, 5):  # Increment by 5% from 10% to 100%
        subset_size = int(len(X_train_resampled) * (size / 100))
        X_subset = X_train_resampled[:subset_size]
        y_subset = y_train[:subset_size]

        # Run the InceptionTime model
        accuracy = run_inception_time(X_subset, X_test_resampled, y_subset, y_test, target_length)
        accuracies.append(accuracy)
        sizes.append(size)

    # Plot results and save to .jpg
    plt.figure()
    plt.plot(sizes, accuracies, marker='o')
    plt.xlabel("Percentage of Training Data (%)")
    plt.ylabel("Accuracy")
    plt.title("InceptionTime Accuracy vs. Training Dataset Size")
    plt.savefig("inceptiontime_accuracy_vs_data_size.jpg")
    print("Plot saved as 'inceptiontime_accuracy_vs_data_size.jpg'.")

# Function call to start evaluation and plotting
evaluate_and_plot_accuracy()
