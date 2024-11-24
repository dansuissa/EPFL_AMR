import numpy as np
from scipy.stats import skew
import pandas as pd
def active_learning_query(model, X_pool, n_queries=10):
    probabilities = model.predict_proba(X_pool)
    uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)
    query_indices = np.argsort(uncertainties)[-n_queries:]
    return query_indices

def remove_highly_correlated_features(features, model, threshold=0.9):
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Retain the most important feature in each highly correlated pair
    feature_importances = pd.Series(model.feature_importances_, index=features.columns)
    to_drop = [col for col in to_drop if col not in feature_importances.nlargest(len(to_drop)).index]

    return features.drop(columns=to_drop)


def remove_highly_correlated_features_v2(features, model, threshold=0.9):
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()

    for column in upper.columns:
        high_corr = upper[column][upper[column] > threshold]
        for correlated_feature in high_corr.index:
            if correlated_feature not in to_drop:
                if model.feature_importances_[features.columns.get_loc(correlated_feature)] < \
                        model.feature_importances_[features.columns.get_loc(column)]:
                    to_drop.add(correlated_feature)
                else:
                    to_drop.add(column)
                    break

    return features.drop(columns=to_drop)
def prepare_features(docs):
    features = []
    targets = []
    for doc in docs:
        features.append({
            'antibiotics_quantity': doc['antibiotics_quantity'],
            'mean_OFF': doc.get('mean_OFF', np.nan),
            'std_OFF': doc.get('std_OFF', np.nan),
            'q25_OFF': doc.get('q25_OFF', np.nan),
            'q75_OFF': doc.get('q75_OFF', np.nan),
            'median_OFF': doc.get('median_OFF', np.nan),
            'skew_OFF': doc.get('skew_OFF', np.nan),
            'max_OFF': doc.get('max_OFF', np.nan),
            'min_OFF': doc.get('min_OFF', np.nan),
            'most_prob_OFF': doc.get('most_prob_OFF', np.nan),
            'mean_ON': doc.get('mean_ON', np.nan),
            'std_ON': doc.get('std_ON', np.nan),
            'q25_ON': doc.get('q25_ON', np.nan),
            'q75_ON': doc.get('q75_ON', np.nan),
            'median_ON': doc.get('median_ON', np.nan),
            'skew_ON': doc.get('skew_ON', np.nan),
            'max_ON': doc.get('max_ON', np.nan),
            'min_ON': doc.get('min_ON', np.nan),
            'most_prob_ON': doc.get('most_prob_ON', np.nan),
            'mean_trap': doc.get('mean_trap', np.nan),
            'std_trap': doc.get('std_trap', np.nan),
            'q25_trap': doc.get('q25_trap', np.nan),
            'q75_trap': doc.get('q75_trap', np.nan),
            'median_trap': doc.get('median_trap', np.nan),
            'skew_trap': doc.get('skew_trap', np.nan),
            'max_trap': doc.get('max_trap', np.nan),
            'min_trap': doc.get('min_trap', np.nan),
            'most_prob_trap': doc.get('most_prob_trap', np.nan),
            'mean_on_to_trapping': doc.get('mean_on_to_trapping', np.nan),
            'std_on_to_trapping': doc.get('std_on_to_trapping', np.nan),
            'q25_on_to_trapping': doc.get('q25_on_to_trapping', np.nan),
            'q75_on_to_trapping': doc.get('q75_on_to_trapping', np.nan),
            'median_on_to_trapping': doc.get('median_on_to_trapping', np.nan),
            'skew_on_to_trapping': doc.get('skew_on_to_trapping', np.nan),
            'max_on_to_trapping': doc.get('max_on_to_trapping', np.nan),
            'min_on_to_trapping': doc.get('min_on_to_trapping', np.nan),
            'most_prob_on_to_trapping': doc.get('most_prob_on_to_trapping', np.nan)
        })
        targets.append(doc['living_state'])

    return pd.DataFrame(features), targets

def calculate_features(data, state):
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
            f'min_{state}': np.nan,
            f'most_prob_{state}': np.nan
        }
    features = {
        f'mean_{state}': np.mean(data),
        f'std_{state}': np.std(data),
        f'q25_{state}': np.percentile(data, 25),
        f'q75_{state}': np.percentile(data, 75),
        f'median_{state}': np.median(data),
        f'skew_{state}': skew(data),
        f'max_{state}': np.max(data),
        f'min_{state}': np.min(data),
        f'most_prob_{state}': max(set(data), key=data.count)  # Most frequent value
    }
    return features

