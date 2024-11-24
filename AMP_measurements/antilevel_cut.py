import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def stratified_split(docs, test_size=0.1, val_size=0.2):
    train_val_docs, test_docs = train_test_split(
        docs, test_size=test_size, stratify=[doc['antibiotics_quantity'] for doc in docs if doc['antibiotics_quantity'] is not None], random_state=42)
    
    train_val_antibiotics = [doc['antibiotics_quantity'] for doc in train_val_docs if doc['antibiotics_quantity'] is not None]
    train_docs, val_docs = train_test_split(
        train_val_docs, test_size=val_size / (1 - test_size), stratify=train_val_antibiotics, random_state=42)

    return train_docs, val_docs, test_docs
def merge_data(docs_analyzed, docs_meas):
    merged_docs = []
    meas_dict = {str(doc['_id']): doc for doc in docs_meas}

    print("Merging data...")
    for doc in docs_analyzed:
        _id = str(doc['_id'])
        if _id in meas_dict:
            doc_meas = meas_dict[_id]
            if 'classification' in doc and any(cls['type'] == 'trapping' for cls in doc['classification']):
                print(f"Doc {_id} has trapping classification.")
                merged_doc = {**doc, **doc_meas}
                merged_docs.append(merged_doc)
            else:
                print(f"Doc {_id} has no trapping classification.")
        else:
            print(f"Doc {_id} not found in meas_dict.")
    
    return merged_docs

def calculate_features(data):
    if len(data) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            #'q25': np.nan,
            'q75': np.nan,
            'median': np.nan,
            'skew': np.nan,
            #'max': np.nan,
            #'min': np.nan
        }
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        #'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'median': np.median(data),
        'skew': pd.Series(data).skew(),
        #'max': np.max(data),
        #'min': np.min(data)
    }

def augment_trapping_data(normalized_doc, segments=10):
    trapping_data = normalized_doc['data_trapping']
    segment_length = len(trapping_data) // segments
    augmented_docs = []

    for i in range(segments):
        segment_data = trapping_data[i * segment_length: (i + 1) * segment_length]
        features = calculate_features(segment_data)
        
        augmented_doc = {
            '_id': f"{normalized_doc['_id']}_{i+1}",
            'antibiotics_quantity': normalized_doc['antibiotics_quantity'],
            'mean_trap': features['mean'],
            'std_trap': features['std'],
            #'max_trap': features['max']
        }
        augmented_docs.append(augmented_doc)

    return augmented_docs

def normalize_docs(docs_analyzed, docs_meas):
    merged_docs = merge_data(docs_analyzed, docs_meas)
    print(f"Merged docs count: {len(merged_docs)}")
    normalized_docs = []

    for doc in merged_docs:
        unique_id = doc['_id']
        if 'MIC' in doc:
            antibiotics_quantity = 0
        elif 'antibiodil' in doc:
            antibiotics_quantity = doc['antibiodil']
        else:
            antibiotics_quantity = None

        if antibiotics_quantity is None:
            print(f"Doc {unique_id} has no antibiotic quantity.")
            continue

        for classification in doc.get('classification', []):
            if classification['type'] == 'trapping':
                indices = classification['indices']
                norm_factor = classification.get('normalizing_factor', 1)
                data = [doc['data']['transmission'][i] / norm_factor for i in range(indices[0], indices[1])]
                normalized_doc = {
                    '_id': unique_id,
                    'antibiotics_quantity': antibiotics_quantity,
                    'data_trapping': data
                }
                augmented_docs = augment_trapping_data(normalized_doc)
                normalized_docs.extend(augmented_docs)

    print(f"Normalized docs count: {len(normalized_docs)}")
    return normalized_docs



def prepare_features(docs):
    X = []
    y = []

    for doc in docs:
        features = {
            'mean_trap': doc.get('mean_trap'),
            'std_trap': doc.get('std_trap'),
            #'max_trap': doc.get('max_trap')
        }

        if doc['antibiotics_quantity'] is not None:
            X.append(features)
            y.append(doc['antibiotics_quantity'])

    return pd.DataFrame(X), pd.Series(y)

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)

def build_xgboost_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'scale_pos_weight': [1, 2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_xgb_model = grid_search.best_estimator_
    
    return best_xgb_model

if __name__ == '__main__':
    print("Loading data from pickle file...")
    with open('data_analysed.pkl', 'rb') as f:
        docs_analyzed = pickle.load(f)
    with open('data_meas.pkl', 'rb') as f:
        docs_meas = pickle.load(f)
    print("Data loaded successfully.")
    
    print(f"Docs Analyzed sample: {docs_analyzed[:1]}")
    print(f"Docs Meas sample: {docs_meas[:1]}")
    
    norm_docs = normalize_docs(docs_analyzed, docs_meas)
    
    if len(norm_docs) == 0:
        print("No documents left after normalization.")
    else:
        np.random.shuffle(norm_docs)

        train_docs, val_docs, test_docs = stratified_split(norm_docs, test_size=0.2, val_size=0.1)

        X_train, y_train = prepare_features(train_docs)
        X_val, y_val = prepare_features(val_docs)
        X_test, y_test = prepare_features(test_docs)

        #X_train = remove_highly_correlated_features(X_train)
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]

        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_test_imputed = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        antibiotic_mapping = {0: 0, 2: 1, 4: 2, 16: 3, 32: 4}
        reverse_mapping = {v: k for k, v in antibiotic_mapping.items()}
        
        y_train_mapped = y_train.map(antibiotic_mapping)
        y_val_mapped = y_val.map(antibiotic_mapping)
        y_test_mapped = y_test.map(antibiotic_mapping)

        y_train_mapped = y_train_mapped.dropna()
        y_val_mapped = y_val_mapped.dropna()
        y_test_mapped = y_test_mapped.dropna()

        X_train_scaled = X_train_scaled[y_train_mapped.index]
        X_val_scaled = X_val_scaled[y_val_mapped.index]
        X_test_scaled = X_test_scaled[y_test_mapped.index]

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_mapped)

        xgb_model = build_xgboost_model(X_train_balanced, y_train_balanced)
        
        val_preds = xgb_model.predict(X_val_scaled)
        val_preds_original = pd.Series(val_preds).map(reverse_mapping)
        val_accuracy = accuracy_score(y_val_mapped.map(reverse_mapping), val_preds_original)
        print(f"Validation Accuracy: {val_accuracy:.2f}")
        print("Validation Classification Report:")
        print(classification_report(y_val_mapped.map(reverse_mapping), val_preds_original, target_names=['0', '2', '4', '16', '32']))

        test_preds = xgb_model.predict(X_test_scaled)
        test_preds_original = pd.Series(test_preds).map(reverse_mapping)
        test_accuracy = accuracy_score(y_test_mapped.map(reverse_mapping), test_preds_original)
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print("Test Classification Report:")
        print(classification_report(y_test_mapped.map(reverse_mapping), test_preds_original, target_names=['0', '2', '4', '16', '32']))
    
        with open('xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

        print("Model saved successfully.")
