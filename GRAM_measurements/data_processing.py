"""
Data processing functions for the Gram and bacteria classification models.

This module contains functions for:
- Normalizing data
- Splitting data
- Augmenting data
- Chunking time series data

Functions:
    normalize_data_model1(docs)
    normalize_data_model2(docs)
    chunk_time_series(docs, num_chunks)
    augment_data(docs, noise_level, shift_max)
    split_data(docs)
"""
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler

#fonction to normalize data for model 1
def normalize_data_model1(docs):
    normalized_docs = []
    for doc in docs:
        transmission = np.array(doc['data']['transmission'])
        norm_factor = doc['normalization_factor']
        transmission_normalized = transmission / norm_factor
        normalized_docs.append({
            'transmission_normalized': transmission_normalized,
            'label': doc['gram_type'],  #gram classification label
            'bacteria': doc['bacteria'],  #strain
            'name': doc['name']  #name for filtering
        })
    return normalized_docs

#function to normale data for model 2 and 3
def normalize_data_model2(docs):
    """Normalize data for model 2 using the provided normalization factor."""
    normalized_docs = []
    for doc in docs:
        transmission = np.array(doc['data']['transmission'])
        norm_factor = doc['normalization_factor']
        transmission_normalized = transmission / norm_factor
        normalized_docs.append({
            'transmission_normalized': transmission_normalized,
            'label': doc['gram_type'],
            'shape': doc['shape_type'],  # 0 if bacillus, 1 if coccus
            'name': doc['name'],
            'bacteria': doc['bacteria']
        })
    return normalized_docs

#define a function that assures the presence of eah bateria family in each of the training, testing and validation sets
def stratified_split_data(docs, test_size=0.1, val_size=0.2):
    """Ensure each set has all bacteria types."""
    bacteria_types = list(set(doc['bacteria'] for doc in docs))
    train_docs, test_docs, val_docs = [], [], []
    for btype in bacteria_types:
        btype_docs = [doc for doc in docs if doc['bacteria'] == btype]
        train_split, temp_split = train_test_split(btype_docs, test_size=test_size + val_size, random_state=42, shuffle=True)
        val_split, test_split = train_test_split(temp_split, test_size=test_size / (test_size + val_size), random_state=42, shuffle=True)
        train_docs.extend(train_split)
        val_docs.extend(val_split)
        test_docs.extend(test_split)
    print(f"Number of training samples: {len(train_docs)}")
    print(f"Number of validation samples: {len(val_docs)}")
    print(f"Number of test samples: {len(test_docs)}")
    return train_docs, val_docs, test_docs


def stratified_split_data_ts(df, test_size=0.1, val_size=0.2):
    """Ensure each set has all bacteria types."""

    bacteria_types = df['bacteria'].unique()
    train_docs, test_docs, val_docs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for btype in bacteria_types:
        btype_docs = df[df['bacteria'] == btype]
        train_split, temp_split = train_test_split(btype_docs, test_size=test_size + val_size, random_state=42,
                                                   shuffle=True)
        val_split, test_split = train_test_split(temp_split, test_size=test_size / (test_size + val_size),
                                                 random_state=42, shuffle=True)
        train_docs = pd.concat([train_docs, train_split])
        val_docs = pd.concat([val_docs, val_split])
        test_docs = pd.concat([test_docs, test_split])

    print(f"Number of training samples: {len(train_docs)}")
    print(f"Number of validation samples: {len(val_docs)}")
    print(f"Number of test samples: {len(test_docs)}")

    return train_docs, val_docs, test_docs

def split_data(docs):
    train_docs = []
    test_docs = []
    bacteria_families = set([doc['bacteria'] for doc in docs])
    for family in bacteria_families:
        family_docs = [doc for doc in docs if doc['bacteria'] == family]
        train_family_docs, test_family_docs = train_test_split(family_docs, test_size=0.2, random_state=42)
        train_docs.extend(train_family_docs)
        test_docs.extend(test_family_docs)
    return train_docs, test_docs

#funciton to split each transmision by a shift, max set at 10 and add also manually add noise
#noise_level : standard deviation of the gaussian noise to be added
#noise reduces overffitting by making your data to generralize on noisy data and learn the underlying patterns in your data instead of fitting each points
# Function to augment data without leakage
def augment_data(docs, noise_level=0.05, shift_max=10):
    augmented_docs = []
    for doc in docs:
        transmission = np.array(doc['transmission_normalized'])
        original_shape = transmission.shape
        for _ in range(2):  # Duplicate each document twice
            # Add noise
            noisy_transmission = transmission + np.random.normal(0, noise_level, len(transmission))
            if noisy_transmission.shape != original_shape:
                raise ValueError(f"Shape mismatch after adding noise: original {original_shape}, noisy {noisy_transmission.shape}")
            augmented_docs.append({
                'transmission_normalized': noisy_transmission,
                'label': doc['label'],
                'shape': doc['shape'],
                'bacteria': doc['bacteria'],
                'name': doc['name'] + '_noise'
            })
            # Shift time series
            shift = np.random.randint(-shift_max, shift_max)
            shifted_transmission = np.roll(transmission, shift)
            if shifted_transmission.shape != original_shape:
                raise ValueError(f"Shape mismatch after shifting: original {original_shape}, shifted {shifted_transmission.shape}")
            augmented_docs.append({
                'transmission_normalized': shifted_transmission,
                'label': doc['label'],
                'shape': doc['shape'],
                'bacteria': doc['bacteria'],
                'name': doc['name'] + '_shift'
            })
    return augmented_docs



#function to separate each time series in 6 chunks
def chunk_time_series(docs, num_chunks=6):
    chunked_docs = []
    for doc in docs:
        transmission = np.array(doc['transmission_normalized'])
        original_length = len(transmission)
        chunk_size = original_length // num_chunks
        for i in range(num_chunks):
            chunk = transmission[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) != chunk_size:  # Ensure chunk is of correct size
                raise ValueError(f"Chunk size mismatch: expected {chunk_size}, got {len(chunk)}")
            chunked_docs.append({
                'transmission_normalized': chunk,
                'label': doc['label'],
                'shape': doc['shape'],
                'bacteria': doc['bacteria'],
                'name': f"{doc['name']}_chunk_{i}"
            })
    return chunked_docs

def stratified_split_data_NN(features, labels, test_size=0.1, val_size=0.2):
    df = pd.DataFrame({'features': list(features), 'labels': labels})
    bacteria_types = df['labels'].unique()
    train_docs, test_docs, val_docs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for btype in bacteria_types:
        btype_docs = df[df['labels'] == btype]
        train_split, temp_split = train_test_split(btype_docs, test_size=test_size + val_size, random_state=42, shuffle=True)
        val_split, test_split = train_test_split(temp_split, test_size=test_size / (test_size + val_size), random_state=42, shuffle=True)
        train_docs = pd.concat([train_docs, train_split])
        val_docs = pd.concat([val_docs, val_split])
        test_docs = pd.concat([test_docs, test_split])

    train_features = np.array(train_docs['features'].tolist())
    val_features = np.array(val_docs['features'].tolist())
    test_features = np.array(test_docs['features'].tolist())

    train_labels = train_docs['labels'].values
    val_labels = val_docs['labels'].values
    test_labels = test_docs['labels'].values

    print(f"Number of training samples: {len(train_features)}")
    print(f"Number of validation samples: {len(val_features)}")
    print(f"Number of test samples: {len(test_features)}")

    return train_features, val_features, test_features, train_labels, val_labels, test_labels












