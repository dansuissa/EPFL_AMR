import random
import warnings
from sklearn.metrics import classification_report
from torch import optim
from data_processing import *
from feature_extraction import *
from model_training import *
from visualization import *
from optuna_objectives import *
from NN_CNN import *
import pickle
import pandas as pd
import numpy as np
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from statsmodels.tsa.arima.model import ARIMA
from tsfeatures import tsfeatures
import cProfile
import pstats
import os
import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm
from LSTM import *
from Transformers import *
from U_net import *
from nixtla_nn_reg import *
from torchsummary import summary
warnings.filterwarnings("ignore", category=FutureWarning)



def load_existing_features(temp_dir):
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith('temp_features_') and f.endswith('.pkl')]
    existing_features = []
    for temp_file in temp_files:
        file_path = os.path.join(temp_dir, temp_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            existing_features.append(data)
    if existing_features:
        return pd.concat(existing_features, ignore_index=True)
    else:
        return pd.DataFrame()

def compute_missing_features(docs, temp_dir, freq=1, n_jobs=-1):
    unique_ids = [doc['name'] for doc in docs]
    total = len(unique_ids)
    
    # Load existing features
    existing_features = load_existing_features(temp_dir)
    if not existing_features.empty:
        processed_ids = set(existing_features['ts_unique_id'].unique())
    else:
        processed_ids = set()
    
    remaining_ids = [uid for uid in unique_ids if uid not in processed_ids]
    
    def compute_features(uid, doc):
        print(f"Computing features for unique_id: {uid}")
        doc_df = pd.DataFrame({'unique_id': [uid] * len(doc['transmission_normalized']), 'value': doc['transmission_normalized']})
        features = tsfeatures(doc_df, freq=freq)
        features['ts_unique_id'] = uid
        temp_features_file = os.path.join(temp_dir, f"temp_features_{uid}.pkl")
        with open(temp_features_file, 'wb') as f:
            pickle.dump(features, f)
        return features
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_features)(uid, doc) 
        for uid, doc in zip(remaining_ids, docs) 
        if print(f"Computing features for unique_id: {uid}") is None
    )
    
    if results:
        new_features = pd.concat(results, ignore_index=True)
        combined_features = pd.concat([existing_features, new_features], ignore_index=True)
        return combined_features
    else:
        return existing_features

def save_combined_features(features, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Combined features saved to {output_file}")


def extract_bacteria_labels(docs):
    return [doc['bacteria'] for doc in docs]

# Ensure correct data types function
def ensure_correct_data_types(panel_data):
    # Print the first few rows of the panel DataFrame
    print("First few rows of the panel DataFrame before ensuring that columns have correct data types:")
    print(pd.DataFrame(panel_data).head())

    # Ensure correct data types
    panel_data['unique_id'] = panel_data['unique_id'].astype(str)
    panel_data['ds'] = pd.to_datetime(panel_data['ds'], errors='coerce')
    panel_data['y'] = pd.to_numeric(panel_data['y'], errors='coerce')

    # Remove rows where 'ds' or 'y' could not be converted
    panel_data = panel_data.dropna(subset=['ds', 'y'])

    # Print the first few rows after cleaning
    print("First few rows of the panel DataFrame after ensuring that columns have correct data types:")
    print(pd.DataFrame(panel_data).head())

    # Print data types
    print("\nData types of the panel DataFrame columns after conversion:")
    print(panel_data.dtypes)

    # Print a summary of the 'y' column to verify all values are numeric
    print(panel_data['y'].describe())

    return panel_data


#process data to add time stamps
def process_doc_chunk(docs):
    panel_data = []
    base_date = pd.Timestamp('2020-01-01')
    for doc in docs:
        for i, val in enumerate(doc['transmission_normalized']):
            ds = base_date + pd.Timedelta(seconds=i)
            panel_data.append({'unique_id': doc['name'], 'ds': ds, 'y': val})
    return panel_data

def process_and_save_chunk(docs, chunk_idx):
    chunk_data = process_doc_chunk(docs)
    chunk_df = pd.DataFrame(chunk_data)
    chunk_file = f"panel_chunk_{chunk_idx}.pkl"
    chunk_df.to_pickle(chunk_file)
    print(f"Processed and saved chunk {chunk_idx}")
    return chunk_file

def process_in_chunks(docs, chunk_size):
    num_chunks = len(docs) // chunk_size + (len(docs) % chunk_size > 0)
    chunk_files = []
    for i in range(num_chunks):
        chunk = docs[i * chunk_size:(i + 1) * chunk_size]
        chunk_file = process_and_save_chunk(chunk, i)
        chunk_files.append(chunk_file)
    return chunk_files

"""
def parallel_tsfeatures(df, freq, n_jobs=-1):
    unique_ids = df['unique_id'].unique()
    results = Parallel(n_jobs=n_jobs)(delayed(tsfeatures)(df[df['unique_id'] == uid], freq=freq) for uid in unique_ids)
    return pd.concat(results, axis=0)
"""
"""
def parallel_tsfeatures(df, freq, n_jobs=-1):
    unique_ids = df['unique_id'].unique()
    total = len(unique_ids)

    results = Parallel(n_jobs=n_jobs)(delayed(tsfeatures)(df[df['unique_id'] == uid], freq=freq) for i, uid in enumerate(unique_ids) if print(f"Computing features for group {i + 1}/{total} with unique_id: {uid}") is None)

    return pd.concat(results, axis=0)
"""
def parallel_tsfeatures(df, freq, existing_features_file, n_jobs=-1):
    # Get unique IDs from the dataframe
    unique_ids = df['unique_id'].unique()
    total = len(unique_ids)
    
    # Load existing features if they exist
    if os.path.exists(existing_features_file):
        with open(existing_features_file, 'rb') as f:
            existing_features = pd.read_pickle(f)
    else:
        existing_features = pd.DataFrame()

    # Check which IDs have already been processed
    if 'ts_unique_id' in existing_features:
        processed_ids = set(existing_features['ts_unique_id'].unique())
    else:
        processed_ids = set()

    # Identify remaining IDs to be processed
    remaining_ids = [uid for uid in unique_ids if uid not in processed_ids]

    def compute_features(uid):
        print(f"Computing features for unique_id: {uid}")
        features = tsfeatures(df[df['unique_id'] == uid], freq=freq)
        features['ts_unique_id'] = uid
        
        # Save intermediate features to a temporary file
        temp_features_file = f"temp_features_{uid}.pkl"
        with open(temp_features_file, 'wb') as f:
            pickle.dump(features, f)
        return temp_features_file

    # Compute features in parallel and save results to temporary files
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_features)(uid) 
        for i, uid in enumerate(remaining_ids) 
        if print(f"Computing features for group {i + 1}/{total} with unique_id: {uid}") is None
    )

    # Load all new features from temporary files and combine them
    new_features = pd.concat([pd.read_pickle(file) for file in results], ignore_index=True)

    # Combine new features with existing features
    combined_features = pd.concat([existing_features, new_features], ignore_index=True)

    # Save the combined features to the existing features file
    with open(existing_features_file, 'wb') as f:
        pickle.dump(combined_features, f)

    # Clean up temporary files
    for file in results:
        os.remove(file)

    return combined_features
def correct_existing_features(ts_features_file):
    # Load the existing ts_features.pkl file
    with open(ts_features_file, 'rb') as f:
        ts_features = pickle.load(f)

    # Print the first few rows to understand the current structure
    print("First few rows of the existing ts_features DataFrame:")
    print(ts_features.head())

    # Check the types of the data
    print(f"Type of ts_features: {type(ts_features)}")

    # If it's a list of dictionaries, convert it to a DataFrame
    if isinstance(ts_features, list):
        ts_features = pd.DataFrame(ts_features)

    # Correct the bacteria naming based on unique_id
    def get_bacteria_name(unique_id):
        id_parts = unique_id.split('_')
        bacteria_id = id_parts[0]
        return bacteria_id

    # Apply the bacteria naming correction
    ts_features['bacteria'] = ts_features['ts_unique_id'].apply(get_bacteria_name)

    # Save the corrected ts_features.pkl file
    with open(ts_features_file, 'wb') as f:
        pickle.dump(ts_features, f)

    print("Updated ts_features saved to ts_features.pkl")
    print("First few rows of the updated ts_features DataFrame:")
    print(ts_features.head())
    
    
    
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    """
    Main script for loading data and training different models for Gram type and bacteria family classification.

    The user can choose from three models:
    1. A baseline model that uses simple statistical rules without machine learning.
    2. A model using RandomForestClassifier and hyperparameter optimization and cross validation with Optuna.
    3. An advanced model using XGBoost and hyperparameter optimization and cross validation with Optuna.

    The script loads preprocessed data from a pickle file and filters out problematic samples.
    It then trains the chosen model and evaluates its performance on test data.
    """
    print("Loading data from pickle file...")
    with open('data.pkl', 'rb') as f:
        docs = pickle.load(f)
    print("Data loaded successfully.")
    #modelnb= input("Type the number of the model you want to use : ")
    modelnb='5'
    #list of problematic values to exclude
    problematic_names = [
        'ns139_p8_2', 'ys134_p7_1', 'pp46_p3_0', 'li142_p2_3',
        'li142_p2_2', 'li142_p3_4', 'li142_p3_5', 'se9_p2_0', 'se9_p7_1'
    ]
    display_names = {
        'bs134': 'B. subtilis',
        'coli': 'E. coli',
        'li142': 'L. innocua',
        'ns139': 'N. sicca',
        'pp46': 'P. putida',
        'pp6': 'Pseudomonas putida',
        'se26': 'S. epidermidis B',
        'se9': 'S. epidermidis A',
        'ys134': 'Y. ruckeri'
    }

    #model 1 not using any learning just theory
    if modelnb == '1':
        normalized_docs = normalize_data_model1(docs)
        filtered_docs = [doc for doc in normalized_docs if doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']
        train_docs, test_docs = split_data(filtered_docs)
        interval_min, interval_max = calculate_means_and_interval_model1(train_docs)
        print(f"Interval: ({interval_min}, {interval_max})")

        #gram type classification
        true_labels = []
        predicted_labels = []
        for doc in test_docs:
            normalized_transmission = doc['transmission_normalized']
            true_label = doc['label']
            predicted_label = baseline_model_model1(normalized_transmission, interval_min, interval_max)
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
        print(f'Baseline model accuracy (Gram type): {accuracy * 100:.2f}%')

        #bacteria family classification
        cluster_centers = find_cluster_centers(train_docs)
        plot_training_clusters(train_docs, cluster_centers, display_names)
        true_bacteria_labels = []
        predicted_bacteria_labels = []
        for doc in test_docs:
            normalized_transmission = doc['transmission_normalized']
            true_bacteria = doc['bacteria']
            predicted_bacteria = predict_bacteria_family_model1(normalized_transmission, cluster_centers)
            true_bacteria_labels.append(true_bacteria)
            predicted_bacteria_labels.append(predicted_bacteria)
        bacteria_accuracy = np.mean(np.array(true_bacteria_labels) == np.array(predicted_bacteria_labels))
        print(f'Baseline model accuracy (Bacteria type): {bacteria_accuracy * 100:.2f}%')

        #plots
        plot_all_clusters(train_docs, test_docs, cluster_centers, display_names, true_bacteria_labels,predicted_bacteria_labels)
        plot_gram_type_classification(train_docs, interval_min, interval_max)
        plot_confusion_matrix(true_bacteria_labels, predicted_bacteria_labels, display_names)
        plot_cluster_center_calculation(train_docs, cluster_centers, display_names)

    #model 2 using random forest classification
    elif modelnb == '2':
        feature_names = np.array(
            ['mean_transmission', 'skewness', 'kurtosis', 'cross_distance_fft', 'std_transmission'])

        normalized_docs = normalize_data_model2(docs)
        filtered_docs = [doc for doc in normalized_docs if
                         doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']
        chunked_docs = chunk_time_series(filtered_docs, num_chunks=6)
        augmented_docs = augment_data(chunked_docs, noise_level=0.05, shift_max=10)

        train_docs, test_docs = split_data(augmented_docs)

        #extract features for Gram type classification (without Gram type and shape type as feature)
        train_features, train_gram_labels, train_bacteria_labels, train_shape_labels = extract_features_model2(
            train_docs,
            use_gram_type_as_feature=False,
            use_shape_type_as_feature=False)
        test_features, test_gram_labels, test_bacteria_labels, test_shape_labels = extract_features_model2(test_docs,
                                                                                                           use_gram_type_as_feature=False,
                                                                                                           use_shape_type_as_feature=False)

        #encode labels
        label_encoder_bacteria = LabelEncoder()
        encoded_train_bacteria_labels = label_encoder_bacteria.fit_transform(train_bacteria_labels)
        encoded_test_bacteria_labels = label_encoder_bacteria.transform(test_bacteria_labels)

        label_encoder_shape = LabelEncoder()
        encoded_train_shape_labels = label_encoder_shape.fit_transform(train_shape_labels)
        encoded_test_shape_labels = label_encoder_shape.transform(test_shape_labels)

        #split data for Gram type classification
        X_train_full, X_test, y_train_full, y_test = train_features, test_features, train_gram_labels, test_gram_labels
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.3,
                                                              random_state=42)

        #split data for Shape type classification (with Gram type as feature)
        train_features_shape, _, _, train_shape_labels_shape = extract_features_model2(train_docs,
                                                                                       use_gram_type_as_feature=False,
                                                                                       use_shape_type_as_feature=False)
        test_features_shape, _, _, test_shape_labels_shape = extract_features_model2(test_docs,
                                                                                     use_gram_type_as_feature=False,
                                                                                     use_shape_type_as_feature=False)
        X_train_shape_full, X_test_shape, y_train_shape_full, y_test_shape = train_features_shape, test_features_shape, encoded_train_shape_labels, encoded_test_shape_labels
        X_train_shape, X_valid_shape, y_train_shape, y_valid_shape = train_test_split(X_train_shape_full,
                                                                                      y_train_shape_full, test_size=0.3,
                                                                                      random_state=42)

        #split data for Bacteria type classification (with Gram type as feature but without shape type)
        train_features_bacteria, _, train_bacteria_labels_bacteria, _ = extract_features_model2(train_docs,
                                                                                                use_gram_type_as_feature=False,
                                                                                                use_shape_type_as_feature=False)
        test_features_bacteria, _, test_bacteria_labels_bacteria, _ = extract_features_model2(test_docs,
                                                                                              use_gram_type_as_feature=False,
                                                                                              use_shape_type_as_feature=False)
        encoded_train_bacteria_labels_bacteria = label_encoder_bacteria.fit_transform(train_bacteria_labels_bacteria)
        encoded_test_bacteria_labels_bacteria = label_encoder_bacteria.transform(test_bacteria_labels_bacteria)
        X_train_bacteria_full, X_test_bacteria, y_train_bacteria_full, y_test_bacteria = train_features_bacteria, test_features_bacteria, encoded_train_bacteria_labels_bacteria, encoded_test_bacteria_labels_bacteria
        X_train_bacteria, X_valid_bacteria, y_train_bacteria, y_valid_bacteria = train_test_split(X_train_bacteria_full,
                                                                                                  y_train_bacteria_full,
                                                                                                  test_size=0.3,
                                                                                                  random_state=42)

        #plot feature correlation matrix
        feature_names_with_gram = np.append(feature_names, 'gram_type')
        X_train_df_with_gram = pd.DataFrame(X_train_bacteria, columns=feature_names_with_gram)
        plot_feature_correlations(X_train_df_with_gram)

        #optimize and train Gram type classifier
        objective = ObjectiveRFModel(X_train, X_valid, y_train, y_valid, X_train_bacteria, X_valid_bacteria,
                                     y_train_bacteria, y_valid_bacteria, X_train_shape, X_valid_shape, y_train_shape,
                                     y_valid_shape)
        study_gram = optuna.create_study(direction='maximize')
        study_gram.optimize(objective.objective_rf_gram, n_trials=20)

        print(f"Best trial (Gram type): {study_gram.best_trial.value}")
        print("Best hyperparameters (Gram type): ", study_gram.best_trial.params)

        median_params_gram = objective.get_median_best_params(study_gram)
        rf_gram = RandomForestClassifier(**median_params_gram, random_state=42)
        rf_gram.fit(X_train, y_train)

        train_accuracy_gram = rf_gram.score(X_train, y_train)
        valid_accuracy_gram = rf_gram.score(X_valid, y_valid)
        test_accuracy_gram = rf_gram.score(X_test, y_test)
        print(f'Training accuracy for Gram type classifier: {train_accuracy_gram:.2f}')
        print(f'Validation accuracy for Gram type classifier: {valid_accuracy_gram:.2f}')
        print(f'Test accuracy for Gram type classifier: {test_accuracy_gram:.2f}')

        y_pred_gram = rf_gram.predict(X_test)
        print("Gram Type Classification Report:")
        print(classification_report(y_test, y_pred_gram, zero_division=0))

        #optimize and train Shape type classifier
        study_shape = optuna.create_study(direction='maximize')
        study_shape.optimize(objective.objective_rf_shape, n_trials=20)

        print(f"Best trial (Shape type): {study_shape.best_trial.value}")
        print("Best hyperparameters (Shape type): ", study_shape.best_trial.params)

        median_params_shape = objective.get_median_best_params(study_shape)
        rf_shape = RandomForestClassifier(**median_params_shape, random_state=42)
        rf_shape.fit(X_train_shape, y_train_shape)

        train_accuracy_shape = rf_shape.score(X_train_shape, y_train_shape)
        valid_accuracy_shape = rf_shape.score(X_valid_shape, y_valid_shape)
        test_accuracy_shape = rf_shape.score(X_test_shape, y_test_shape)
        print(f'Training accuracy for Shape type classifier: {train_accuracy_shape:.2f}')
        print(f'Validation accuracy for Shape type classifier: {valid_accuracy_shape:.2f}')
        print(f'Test accuracy for Shape type classifier: {test_accuracy_shape:.2f}')

        y_pred_shape = rf_shape.predict(X_test_shape)
        print("Shape Type Classification Report:")
        print(classification_report(y_test_shape, y_pred_shape, zero_division=0))

        #optimize and train Bacteria type classifier
        study_bacteria = optuna.create_study(direction='maximize')
        study_bacteria.optimize(objective.objective_rf_bacteria, n_trials=20)

        print(f"Best trial (Bacteria type): {study_bacteria.best_trial.value}")
        print("Best hyperparameters (Bacteria type): ", study_bacteria.best_trial.params)

        median_params_bacteria = objective.get_median_best_params(study_bacteria)
        rf_bacteria = RandomForestClassifier(**median_params_bacteria, random_state=42)
        rf_bacteria.fit(X_train_bacteria, y_train_bacteria)
        train_accuracy_bacteria = rf_bacteria.score(X_train_bacteria, y_train_bacteria)
        valid_accuracy_bacteria = rf_bacteria.score(X_valid_bacteria, y_valid_bacteria)
        test_accuracy_bacteria = rf_bacteria.score(X_test_bacteria, y_test_bacteria)
        print(f'Training accuracy for Bacteria type classifier: {train_accuracy_bacteria:.2f}')
        print(f'Validation accuracy for Bacteria type classifier: {valid_accuracy_bacteria:.2f}')
        print(f'Test accuracy for Bacteria type classifier: {test_accuracy_bacteria:.2f}')

        y_pred_bacteria = rf_bacteria.predict(X_test_bacteria)
        print("Bacteria Type Classification Report:")
        print(classification_report(y_test_bacteria, y_pred_bacteria, target_names=label_encoder_bacteria.classes_,
                                    zero_division=0))

        #plot validation curves and confusion matrices
        #param_range = np.linspace(100, 1000, 10, dtype=int)
        #plot_validation_curve(rf_gram, "Validation Curve (RandomForest Gram Type)", X_train, y_train,
                              #param_name="n_estimators", param_range=param_range, cv=5)
        #plot_validation_curve(rf_bacteria, "Validation Curve (RandomForest Bacteria Type)", X_train_bacteria,
                              #y_train_bacteria, param_name="n_estimators", param_range=param_range, cv=5)
        #plot_validation_curve(rf_shape, "Validation Curve (RandomForest Shape Type)", X_train_shape, y_train_shape,
                              #param_name="n_estimators", param_range=param_range, cv=5)

        plot_confusion_matrix3(y_test_bacteria, y_pred_bacteria, display_labels=label_encoder_bacteria.classes_)
        y_scores_gram = rf_gram.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_scores_gram)
        plot_precision_recall_curve(y_test, y_scores_gram)


    elif modelnb == '3':

        feature_names = np.array(

            ['skewness', 'kurtosis', 'cross_distance_fft', 'std_transmission', 'mean'])

        normalized_docs = normalize_data_model2(docs)

        filtered_docs = [doc for doc in normalized_docs if

                         doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']

        chunked_docs = chunk_time_series(filtered_docs, num_chunks=6)

        augmented_docs = augment_data(chunked_docs, noise_level=0.05, shift_max=10)

        train_docs, val_docs, test_docs = stratified_split_data(augmented_docs)

        print_bacteria_counts(train_docs, val_docs, test_docs)

        # extract features for Gram type classification (without Gram type and shape type as feature)

        train_features, train_gram_labels, train_bacteria_labels, train_shape_labels = extract_features_model2(

            train_docs, use_gram_type_as_feature=False, use_shape_type_as_feature=False)

        val_features, val_gram_labels, val_bacteria_labels, val_shape_labels = extract_features_model2(

            val_docs, use_gram_type_as_feature=False, use_shape_type_as_feature=False)

        test_features, test_gram_labels, test_bacteria_labels, test_shape_labels = extract_features_model2(

            test_docs, use_gram_type_as_feature=False, use_shape_type_as_feature=False)

        # encode labels

        label_encoder_bacteria = LabelEncoder()

        encoded_train_bacteria_labels = label_encoder_bacteria.fit_transform(train_bacteria_labels)

        encoded_val_bacteria_labels = label_encoder_bacteria.transform(val_bacteria_labels)

        encoded_test_bacteria_labels = label_encoder_bacteria.transform(test_bacteria_labels)

        label_encoder_shape = LabelEncoder()

        encoded_train_shape_labels = label_encoder_shape.fit_transform(train_shape_labels)

        encoded_val_shape_labels = label_encoder_shape.transform(val_shape_labels)

        encoded_test_shape_labels = label_encoder_shape.transform(test_shape_labels)

        # split data for Gram type classification

        X_train, X_valid, y_train, y_valid = train_features, val_features, train_gram_labels, val_gram_labels

        X_test = test_features

        y_test = test_gram_labels

        # split data for Shape type classification (with Gram type as feature)

        train_features_shape, _, _, train_shape_labels_shape = extract_features_model2(train_docs,

                                                                                       use_gram_type_as_feature=True,

                                                                                       use_shape_type_as_feature=False)

        val_features_shape, _, _, val_shape_labels_shape = extract_features_model2(val_docs,

                                                                                   use_gram_type_as_feature=True,

                                                                                   use_shape_type_as_feature=False)

        test_features_shape, _, _, test_shape_labels_shape = extract_features_model2(test_docs,

                                                                                     use_gram_type_as_feature=True,

                                                                                     use_shape_type_as_feature=False)

        X_train_shape, X_valid_shape, y_train_shape, y_valid_shape = train_features_shape, val_features_shape, encoded_train_shape_labels, encoded_val_shape_labels

        X_test_shape, y_test_shape = test_features_shape, encoded_test_shape_labels

        # split data for Bacteria type classification (with Gram type as feature but without shape type)

        train_features_bacteria, _, train_bacteria_labels_bacteria, _ = extract_features_model2(train_docs,

                                                                                                use_gram_type_as_feature=False,

                                                                                                use_shape_type_as_feature=False)

        val_features_bacteria, _, val_bacteria_labels_bacteria, _ = extract_features_model2(val_docs,

                                                                                            use_gram_type_as_feature=False,

                                                                                            use_shape_type_as_feature=False)

        test_features_bacteria, _, test_bacteria_labels_bacteria, _ = extract_features_model2(test_docs,

                                                                                              use_gram_type_as_feature=False,

                                                                                              use_shape_type_as_feature=False)

        encoded_train_bacteria_labels_bacteria = label_encoder_bacteria.fit_transform(train_bacteria_labels_bacteria)

        encoded_val_bacteria_labels_bacteria = label_encoder_bacteria.transform(val_bacteria_labels_bacteria)

        encoded_test_bacteria_labels_bacteria = label_encoder_bacteria.transform(test_bacteria_labels_bacteria)

        X_train_bacteria, X_valid_bacteria, y_train_bacteria, y_valid_bacteria = train_features_bacteria, val_features_bacteria, encoded_train_bacteria_labels_bacteria, encoded_val_bacteria_labels_bacteria

        X_test_bacteria, y_test_bacteria = test_features_bacteria, encoded_test_bacteria_labels_bacteria

        feature_names_with_gram = np.append(feature_names, 'gram_type')

        X_train_df_with_gram = pd.DataFrame(X_train_bacteria, columns=feature_names)

        plot_feature_correlations(X_train_df_with_gram)

        # optimize and train Gram type classifier

        objective = ObjectiveXGBModel(X_train, X_valid, X_test, y_train, y_valid, y_test,

                                      X_train_bacteria, X_valid_bacteria, X_test_bacteria,

                                      y_train_bacteria, y_valid_bacteria, y_test_bacteria,

                                      X_train_shape, X_valid_shape, X_test_shape, y_train_shape, y_valid_shape,

                                      y_test_shape)

        study_gram = optuna.create_study(direction='maximize')

        study_gram.optimize(objective.objective_xgb_gram, n_trials=5)

        print(f"Best trial (Gram type): {study_gram.best_trial.value}")

        print("Best hyperparameters (Gram type): ", study_gram.best_trial.params)

        best_params_gram = study_gram.best_trial.params

        xgb_gram = XGBClassifier(**best_params_gram, use_label_encoder=False, eval_metric='logloss')

        xgb_gram.set_params(early_stopping_rounds=10)

        xgb_gram.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=True)

        train_accuracy_gram = xgb_gram.score(X_train, y_train)

        valid_accuracy_gram = xgb_gram.score(X_valid, y_valid)

        test_accuracy_gram = xgb_gram.score(X_test, y_test)

        print(f'Training accuracy for Gram type classifier: {train_accuracy_gram:.2f}')

        print(f'Validation accuracy for Gram type classifier: {valid_accuracy_gram:.2f}')

        print(f'Test accuracy for Gram type classifier: {test_accuracy_gram:.2f}')

        y_pred_gram = xgb_gram.predict(X_test)

        print("Gram Type Classification Report:")

        print(classification_report(y_test, y_pred_gram, zero_division=0))

        # optimize and train Shape type classifier

        study_shape = optuna.create_study(direction='maximize')

        study_shape.optimize(objective.objective_xgb_shape, n_trials=5)

        print(f"Best trial (Shape type): {study_shape.best_trial.value}")

        print("Best hyperparameters (Shape type): ", study_shape.best_trial.params)

        best_params_shape = study_shape.best_trial.params

        xgb_shape = XGBClassifier(**best_params_shape, use_label_encoder=False, eval_metric='logloss')

        xgb_shape.set_params(early_stopping_rounds=10)

        xgb_shape.fit(X_train_shape, y_train_shape, eval_set=[(X_valid_shape, y_valid_shape)], verbose=True)

        train_accuracy_shape = xgb_shape.score(X_train_shape, y_train_shape)

        valid_accuracy_shape = xgb_shape.score(X_valid_shape, y_valid_shape)

        test_accuracy_shape = xgb_shape.score(X_test_shape, y_test_shape)

        print(f'Training accuracy for Shape type classifier: {train_accuracy_shape:.2f}')

        print(f'Validation accuracy for Shape type classifier: {valid_accuracy_shape:.2f}')

        print(f'Test accuracy for Shape type classifier: {test_accuracy_shape:.2f}')

        y_pred_shape = xgb_shape.predict(X_test_shape)

        print("Shape Type Classification Report:")

        print(classification_report(y_test_shape, y_pred_shape, zero_division=0))

        # optimize and train Bacteria type classifier

        study_bacteria = optuna.create_study(direction='maximize')

        study_bacteria.optimize(objective.objective_xgb_bacteria, n_trials=50)

        print(f"Best trial (Bacteria type): {study_bacteria.best_trial.value}")

        print("Best hyperparameters (Bacteria type): ", study_bacteria.best_trial.params)

        best_params_bacteria = study_bacteria.best_trial.params

        xgb_bacteria = XGBClassifier(**best_params_bacteria, use_label_encoder=False, eval_metric='mlogloss',

                                     objective='multi:softmax', num_class=len(np.unique(y_train_bacteria)))

        xgb_bacteria.set_params(early_stopping_rounds=10)

        xgb_bacteria.fit(X_train_bacteria, y_train_bacteria, eval_set=[(X_valid_bacteria, y_valid_bacteria)],

                         verbose=True)

        train_accuracy_bacteria = xgb_bacteria.score(X_train_bacteria, y_train_bacteria)

        valid_accuracy_bacteria = xgb_bacteria.score(X_valid_bacteria, y_valid_bacteria)

        test_accuracy_bacteria = xgb_bacteria.score(X_test_bacteria, y_test_bacteria)

        print(f'Training accuracy for Bacteria type classifier: {train_accuracy_bacteria:.5f}')

        print(f'Validation accuracy for Bacteria type classifier: {valid_accuracy_bacteria:.5f}')

        print(f'Test accuracy for Bacteria type classifier: {test_accuracy_bacteria:.5f}')

        y_pred_bacteria = xgb_bacteria.predict(X_test_bacteria)

        print("Bacteria Type Classification Report:")

        print(classification_report(y_test_bacteria, y_pred_bacteria, target_names=label_encoder_bacteria.classes_,

                                    zero_division=0))

        # plot

        plot_feature_importance(xgb_bacteria, feature_names)

        param_range = np.logspace(-2, 0, 5)

        plot_validation_curve(xgb_gram, "Validation Curve (XGBoost Gram Type)", X_train, y_train, param_name="gamma",

                              param_range=param_range, cv=5)

        plot_validation_curve(xgb_shape, "Validation Curve (XGBoost Shape Type)", X_train_shape, y_train_shape,

                              param_name="gamma", param_range=param_range, cv=5)

        plot_validation_curve(xgb_bacteria, "Validation Curve (XGBoost Bacteria Type)", X_train_bacteria,

                              y_train_bacteria,

                              param_name="gamma", param_range=param_range, cv=5)

        plot_confusion_matrix3(y_test_bacteria, y_pred_bacteria, display_labels=label_encoder_bacteria.classes_)

        y_scores_gram = xgb_gram.predict_proba(X_test)[:, 1]

        plot_roc_curve(y_test, y_scores_gram)

        plot_precision_recall_curve(y_test, y_scores_gram)
    elif modelnb == '4':
        print("Running model 4...")

        panel_file = 'processed_panel_data.pkl'
        ts_features_file = 'ts_features.pkl'
        #temp_dir = '/home/suissa/temp_scratch/2130011'
        normalized_docs = normalize_data_model2(docs)
        filtered_docs = [doc for doc in normalized_docs if
                         doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']

        augmented_docs = augment_data(filtered_docs, noise_level=0.05, shift_max=10)
        if not os.path.exists(panel_file):
            chunk_size = 100
            chunk_files = process_in_chunks(augmented_docs, chunk_size)
            panel_data = pd.concat([pd.read_pickle(file) for file in chunk_files], ignore_index=True)
            panel_data.to_pickle(panel_file)
        else:
            panel_data = pd.read_pickle(panel_file)
            print(f"Loaded panel data from {panel_file}")

        panel_data = panel_data[(panel_data['unique_id'] != 'bs134_p15_2_shift') & 
                                (panel_data['unique_id'] != 'coli_p23_0_shift') & 
                                (panel_data['unique_id'] != 'coli_p23_0_noise') & 
                                (panel_data['unique_id'] != 'ns139_p8_1_noise')]
        panel_data = ensure_correct_data_types(panel_data)

        bacteria_labels = extract_bacteria_labels(augmented_docs)
        existing_ts_features = pd.DataFrame()
        if os.path.exists(ts_features_file):
            with open(ts_features_file, 'rb') as f:
                existing_ts_features = pickle.load(f)
                print(f"Loaded existing ts_features from {ts_features_file}")
        temp_dir = '/home/suissa/temp_scratch/2130011'

        try:
            combined_ts_features = compute_missing_features(augmented_docs, temp_dir, freq=1, n_jobs=mp.cpu_count())
            save_combined_features(combined_ts_features, ts_features_file)
            print(f"Updated ts_features saved to {ts_features_file}")
        except Exception as e:
            print(f"Error calculating tsfeatures: {e}")
        correct_existing_features(ts_features_file)
        with open(ts_features_file, 'rb') as f:
            combined_ts_features = pickle.load(f)

        combined_ts_features = combined_ts_features.rename(columns=lambda x: f"ts_{x}")
        combined_ts_features['bacteria'] = combined_ts_features['ts_unique_id'].apply(lambda x: x.split('_')[0])

        print("First few rows of the ts_features DataFrame:")
        print(combined_ts_features.head())

        combined_features = combined_ts_features.copy()
        label_encoder = LabelEncoder()
        combined_features['encoded_bacteria'] = label_encoder.fit_transform(combined_features['bacteria'])
        combined_features = pd.get_dummies(combined_features, columns=['ts_unique_id'], drop_first=True)
# Convert categorical columns to numerical codes for XGBoost
        combined_features['ts_ts_unique_id'] = combined_features['ts_ts_unique_id'].astype('category')
        combined_features['ts_bacteria'] = combined_features['ts_bacteria'].astype('category')
        combined_features['ts_ts_unique_id'] = combined_features['ts_ts_unique_id'].cat.codes
        combined_features['ts_bacteria'] = combined_features['ts_bacteria'].cat.codes


        train_df, val_df, test_df = stratified_split_data_ts(combined_features, test_size=0.2, val_size=0.2)

        X_train = train_df.drop(columns=['bacteria', 'encoded_bacteria', 'ts_ts_unique_id', 'ts_bacteria'])
        y_train = train_df['encoded_bacteria']
        X_val = val_df.drop(columns=['bacteria', 'encoded_bacteria', 'ts_ts_unique_id', 'ts_bacteria'])
        y_val = val_df['encoded_bacteria']
        X_test = test_df.drop(columns=['bacteria', 'encoded_bacteria', 'ts_ts_unique_id', 'ts_bacteria'])
        y_test = test_df['encoded_bacteria']
        # Ensure all features are numeric
        combined_features = combined_features.apply(pd.to_numeric, errors='coerce')
        non_numeric_columns = combined_features.columns[combined_features.dtypes == 'object']
        if len(non_numeric_columns) > 0:
            print(f"Non-numeric columns found: {non_numeric_columns}")
            combined_features = combined_features.drop(columns=non_numeric_columns)

        # Ensure all features are numeric
        combined_features = combined_features.apply(pd.to_numeric, errors='coerce')
        combined_features = combined_features.dropna(axis=1, how='any')
        class ObjectiveXGBModel:
            def __init__(self, X_train, X_valid, X_test, y_train, y_valid, y_test):
                self.X_train = X_train
                self.X_valid = X_valid
                self.X_test = X_test
                self.y_train = y_train
                self.y_valid = y_valid
                self.y_test = y_test

            def objective_xgb_bacteria(self, trial):
                param = {
                    'objective': 'multi:softmax',
                    'num_class': len(np.unique(self.y_train)),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 7),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                    'gamma': trial.suggest_float('gamma', 0.1, 10.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
                    'enable_categorical': True  # Enable categorical data handling
                }
                xgb_bacteria = XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
                xgb_bacteria.fit(self.X_train, self.y_train, eval_set=[(self.X_valid, self.y_valid)])
                accuracy = xgb_bacteria.score(self.X_valid, self.y_valid)
                return accuracy
       
        objective = ObjectiveXGBModel(X_train, X_val, X_test, y_train, y_val, y_test)
        study_bacteria = optuna.create_study(direction='maximize')
        study_bacteria.optimize(objective.objective_xgb_bacteria, n_trials=50)

        best_params_bacteria = study_bacteria.best_trial.params
        xgb_bacteria = XGBClassifier(**best_params_bacteria, use_label_encoder=False, eval_metric='mlogloss',
                                     objective='multi:softmax', num_class=len(np.unique(y_train)), enable_categorical=True)
        xgb_bacteria.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        # Evaluate the model
        train_accuracy_bacteria = xgb_bacteria.score(X_train, y_train)
        valid_accuracy_bacteria = xgb_bacteria.score(X_val, y_val)
        test_accuracy_bacteria = xgb_bacteria.score(X_test, y_test)
        print(f'Training accuracy for Bacteria type classifier: {train_accuracy_bacteria:.5f}')
        print(f'Validation accuracy for Bacteria type classifier: {valid_accuracy_bacteria:.5f}')
        print(f'Test accuracy for Bacteria type classifier: {test_accuracy_bacteria:.5f}')

        y_pred_bacteria = xgb_bacteria.predict(X_test)
        #print("Bacteria Type Classification Report:")
        #print(classification_report(y_test, y_pred_bacteria, target_names=LabelEncoder().inverse_transform(y_test)))

        # Plot feature correlation
        #plt.figure(figsize=(12, 10))
        #correlation_matrix = combined_features.corr()
        #sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        #plt.title('Feature Correlation Matrix')
        #plt.show()

        best_model = xgb_bacteria

        # Debugging feature importances
        feature_importances = best_model.feature_importances_
        features = X_train.columns
        print(f"Length of features: {len(features)}")
        print(f"Length of feature_importances: {len(feature_importances)}")
    
        if len(features) != len(feature_importances):
            raise ValueError("Mismatch in lengths of features and feature importances")

        # Plot feature importance
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(importance_df)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.savefig('example_plot.png')
        plt.close()
        # Plot confusion matrix
        #plot_confusion_matrix3(y_test, y_pred_bacteria, display_labels=LabelEncoder().inverse_transform(y_test))
    elif modelnb == '5':
        filtered_docs = [doc for doc in docs if

                         doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']
        features, labels, label_encoder = preprocess_data(filtered_docs)

        # Stratified split data
        train_features, val_features, test_features, train_labels, val_labels, test_labels = stratified_split_data_NN(features, labels)

        # Objective function for Optuna
        def objective(trial):
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            num_neurons = trial.suggest_categorical('num_neurons', [128, 256, 512])
            # Create DataLoaders
            train_loader = create_dataloader(train_features, train_labels, batch_size=batch_size)
            val_loader = create_dataloader(val_features, val_labels, batch_size=batch_size)
            test_loader = create_dataloader(test_features, test_labels, batch_size=batch_size)
            # Define model
            input_length = train_features.shape[1]
            num_classes = len(label_encoder.classes_)
            model = EnhancedDeeperCNN(input_length, num_classes, num_layers=num_layers, num_neurons=num_neurons).to(device)  # Move model to device
            # Train model
            train_model(model, train_loader, val_loader, epochs=64, learning_rate=learning_rate)
            # Evaluate model
            val_loss, val_accuracy, _ = evaluate_model(model, val_loader)
            return val_loss

        # Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        # Print best trial
        trial = study.best_trial
        print(f'Best trial: Value: {trial.value}')
        print(f'Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
        # Train the best model
        best_params = trial.params
        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']
        num_layers = best_params['num_layers']
        num_neurons = best_params['num_neurons']
        # Create DataLoaders
        train_loader = create_dataloader(train_features, train_labels, batch_size=batch_size)
        val_loader = create_dataloader(val_features, val_labels, batch_size=batch_size)
        test_loader = create_dataloader(test_features, test_labels, batch_size=batch_size)
        # Define model
        input_length = train_features.shape[1]
        num_classes = len(label_encoder.classes_)
        model = EnhancedDeeperCNN(input_length, num_classes, num_layers=num_layers, num_neurons=num_neurons).to(device)  # Move model to device
        summary(model, (1, 1000)) 
        # Train model with best hyperparameters
        train_model(model, train_loader, val_loader, epochs=64, learning_rate=learning_rate)
    
        # Evaluate model on the test set
        test_loss, test_accuracy, test_classification_report = evaluate_model(model, test_loader)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        print('Classification Report:')
        print(test_classification_report)

    elif modelnb == '6':
        features, labels, label_encoder = preprocess_data_lmgru(docs)
        train_features, val_features, test_features, train_labels, val_labels, test_labels = stratified_split_data_NN(features, labels)
        train_loader = create_dataloader_lmgru(train_features, train_labels, batch_size=128)
        val_loader = create_dataloader_lmgru(val_features, val_labels, batch_size=128)
        test_loader = create_dataloader_lmgru(test_features, test_labels, batch_size=128)

        input_length = 1
        num_classes = len(label_encoder.classes_)
        model = LMGRU(input_length, hidden_size=128, num_layers=2, num_classes=num_classes, dropout=0.5).to(device)

        train_lmgru_model(model, train_loader, val_loader, epochs=150, learning_rate=0.001)
        test_loss, test_accuracy = evaluate_model_lmgru(model, test_loader)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    elif modelnb == '7':
        # Load and preprocess data
        features, labels, label_encoder = preprocess_data_transformer(docs)
        train_features, val_features, test_features, train_labels, val_labels, test_labels = stratified_split_data_NN(features, labels)
        train_loader = create_dataloader(train_features, train_labels, batch_size=128)
        val_loader = create_dataloader(val_features, val_labels, batch_size=128)
        test_loader = create_dataloader(test_features, test_labels, batch_size=128)
        input_length = features.shape[1]  # Should be 1000
        num_classes = len(label_encoder.classes_)
        model = Transformer(input_length, num_classes).to(device)  # Changed input_size to input_length
        train_transformer_model(model, train_loader, val_loader, epochs=150, learning_rate=0.001)
        test_loss, test_accuracy = evaluate_model_transformer(model, test_loader)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    elif modelnb=='8':
        filtered_docs = [doc for doc in docs if
                         doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']

        main_U(filtered_docs)
    elif modelnb=='9':
        filtered_docs = [doc for doc in docs if doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']
        predictions = main_nix(filtered_docs)
        print(predictions)
        
    