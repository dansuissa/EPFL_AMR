#data processing for AMP dataset
import pandas as pd
import numpy as np
from scipy.stats import skew


def normalize_docs_AMP(docs_analyzed, docs_meas):
    merged_docs = merge_data_AMP(docs_analyzed, docs_meas)
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
         
        # Set all living state to None
        living_state = None

        # Initialize normalized document
        normalized_doc = {
            '_id': unique_id,
            'antibiotics_quantity': antibiotics_quantity,
            'dead': living_state
        }

        combined_indices = None
        combined_data = []

        # Process classification indices and attach data
        for classification in doc.get('classification', []):
            indices = classification['indices']
            norm_factor = classification.get('normalizing_factor', 1)
            state_type = classification['type']

            # Combine ON and on_to_trapping states
            if state_type in ['ON', 'on_to_trapping']:
                if combined_indices is None:
                    combined_indices = indices
                else:
                    combined_indices = [min(combined_indices[0], indices[0]), min(combined_indices[1], indices[1])]

                # Append data from both states
                combined_data.extend([doc['data']['transmission'][i] / norm_factor for i in range(indices[0], indices[1])])
            else:
                # Process other states normally
                data = [doc['data']['transmission'][i] / norm_factor for i in range(indices[0], indices[1])]
                normalized_doc[f'indices_{state_type}'] = indices
                normalized_doc[f'data_{state_type}'] = data

                # Include calculated features if not already included
                if state_type == 'trapping':
                    for feature in ['mean', 'std', 'q25', 'q75', 'median', 'skew', 'max', 'min']:
                        if feature in classification:
                            normalized_doc[f'{feature}_trapping'] = classification[feature]
                else:
                    features = calculate_features_AMP(data, state_type)
                    normalized_doc.update(features)

        # If combined ON and on_to_trapping data exists, process it
        if combined_indices:
            normalized_doc['indices_combined_ON_on_to_trapping'] = combined_indices
            normalized_doc['data_combined_ON_on_to_trapping'] = combined_data
            features = calculate_features_AMP(combined_data, 'combined_ON_on_to_trapping')
            normalized_doc.update(features)

        # Drop the specified columns
        for column in ['_id', 'indices_OFF', 'data_OFF', 'indices_trapping', 'data_trapping', 'indices_combined_ON_on_to_trapping', 'data_combined_ON_on_to_trapping']:
            normalized_doc.pop(column, None)

        normalized_docs.append(normalized_doc)

    return pd.DataFrame(normalized_docs)


#def prepare_features_AMP(docs):
    
def merge_data_AMP(docs_analyzed, docs_meas):
    merged_docs = []
    meas_dict = {doc['_id']: doc for doc in docs_meas}

    for doc in docs_analyzed:
        _id = doc['_id']
        if _id in meas_dict:
            merged_doc = {**doc, **meas_dict[_id]}
            merged_docs.append(merged_doc)

    return merged_docs

def calculate_features_AMP(data, state):
    if state=='combined_ON_on_to_trapping':
        if len(data) == 0:
            # Handle empty data case
            return {
                f'mean_ON': np.nan,
                f'std_ON': np.nan,
                f'q25_ON': np.nan,
                f'q75_ON': np.nan,
                f'median_ON': np.nan,
                f'skew_ON': np.nan,
                f'max_ON': np.nan,
                f'min_ON': np.nan
            }
        features = {
            f'mean_ON': np.mean(data),
            f'std_ON': np.std(data),
            f'q25_ON': np.percentile(data, 25),
            f'q75_ON': np.percentile(data, 75),
            f'median_ON': np.median(data),
            f'skew_ON': skew(data),
            f'max_ON': np.max(data),
            f'min_ON': np.min(data)
        }
        return features

    if len(data) == 0:
        # Handle empty data case
        return {
            f'mean_{state}': np.nan,
            f'std_{state}': np.nan,
            f'q25_{state}': np.nan,
            f'q75_{state}': np.nan,
            f'median_{state}': np.nan,
            f'skew_{state}': np.nan,
            f'max_{state}': np.nan,
            f'min_{state}': np.nan
        }
    features = {
        f'mean_{state}': np.mean(data),
        f'std_{state}': np.std(data),
        f'q25_{state}': np.percentile(data, 25),
        f'q75_{state}': np.percentile(data, 75),
        f'median_{state}': np.median(data),
        f'skew_{state}': skew(data),
        f'max_{state}': np.max(data),
        f'min_{state}': np.min(data)
    }
    return features

