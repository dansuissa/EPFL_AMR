#data_processing.py

import pandas as pd
import numpy as np
from feature_extraction import *
from sklearn.model_selection import train_test_split


def stratified_split(docs, test_size=0.2, val_size=0.1):
    # Separate docs with known and unknown living_state
    known_docs = [doc for doc in docs if doc['living_state'] is not None]
    unknown_docs = [doc for doc in docs if doc['living_state'] is None]

    # Extract antibiotics quantities for stratification
    antibiotics_quantities = [doc['antibiotics_quantity'] for doc in known_docs]

    # Split known data into training+validation and test sets
    train_val_docs, test_docs = train_test_split(
        known_docs,
        test_size=test_size,
        stratify=antibiotics_quantities,
        random_state=42
    )

    # Extract antibiotics quantities for training+validation set
    train_val_antibiotics = [doc['antibiotics_quantity'] for doc in train_val_docs]

    # Split training+validation set into training and validation sets
    train_docs, val_docs = train_test_split(
        train_val_docs,
        test_size=val_size / (1 - test_size),
        stratify=train_val_antibiotics,
        random_state=42
    )

    return train_docs, val_docs, test_docs, unknown_docs
def normalize_docs(docs_analyzed, docs_meas):
    merged_docs = merge_data(docs_analyzed, docs_meas)
    normalized_docs = []

    for doc in merged_docs:
        # Extract unique ID and antibiotic quantity
        unique_id = doc['_id']
        if 'MIC' in doc:
            antibiotics_quantity = 0
        elif 'antibiodil' in doc:
            antibiotics_quantity = doc['antibiodil']
        else:
            antibiotics_quantity = None

        # Determine living state based on antibiotic quantity
        if antibiotics_quantity == 0:
            living_state = 0  # Alive
        elif antibiotics_quantity == 32:
            living_state = 1  # Dead
        else:
            living_state = None  # To be predicted

        # Initialize normalized document
        normalized_doc = {
            '_id': unique_id,
            'antibiotics_quantity': antibiotics_quantity,
            'living_state': living_state
        }

        # Process classification indices and attach data
        for classification in doc.get('classification', []):
            indices = classification['indices']
            norm_factor = classification.get('normalizing_factor', 1)
            state_type = classification['type']

            # Extract and normalize data based on state type
            data = [doc['data']['transmission'][i] / norm_factor for i in range(indices[0], indices[1])]
            normalized_doc[f'indices_{state_type}'] = indices
            normalized_doc[f'data_{state_type}'] = data

            # Include calculated features if not already included
            if state_type == 'trapping':
                for feature in ['mean', 'std', 'q25', 'q75', 'median', 'skew', 'max', 'min', 'most_prob']:
                    if feature in classification:
                        normalized_doc[f'{feature}_trap'] = classification[feature]
            else:
                features = calculate_features(data, state_type)
                normalized_doc.update(features)

        normalized_docs.append(normalized_doc)

    return normalized_docs

def merge_data(docs_analyzed, docs_meas):
    merged_docs = []
    meas_dict = {doc['_id']: doc for doc in docs_meas}

    for doc in docs_analyzed:
        _id = doc['_id']
        if _id in meas_dict:
            merged_doc = {**doc, **meas_dict[_id]}
            merged_docs.append(merged_doc)

    return merged_docs

