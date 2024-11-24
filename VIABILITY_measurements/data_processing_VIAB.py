# data processing for VIABILITY dataset
import numpy as np
import pandas as pd
from scipy import stats

def extract_features_VIAB(docs):
    # Initialize a list to hold the processed feature dictionaries
    processed_docs = []
    
    # Iterate over each document in the list
    for data in docs:
        features = {}
        
        # Extract features from the 'classification' list in each document
        for state in data.get('classification', []):
            state_type = state['type']
            features[f'mean_{state_type}'] = state['mu']
            features[f'std_{state_type}'] = state['std']
            features[f'q25_{state_type}'] = state['q25']
            features[f'q75_{state_type}'] = state['q75']
            features[f'median_{state_type}'] = state['median']
            features[f'skew_{state_type}'] = state['skew']
            features[f'max_{state_type}'] = state['max']
            features[f'min_{state_type}'] = state['min']
        
        # Handle 'death' target variable
        features['dead'] = 1 if 'dead' in data else 0
        
        # Append the processed features to the list
        processed_docs.append(features)
    
    return pd.DataFrame(processed_docs)