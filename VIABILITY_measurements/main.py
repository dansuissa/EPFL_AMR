from data_processing_AMP import *
from data_processing_VIAB import *
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the selected features manually
selected_features = [
    'min_trapping', 'max_trapping', 'mean_trapping', 'std_trapping'
]

# Data Augmentation Function
def augment_data(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Optionally, add some random noise to the data for further augmentation
    noise_factor = 0.05
    X_res += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_res.shape)
    
    return X_res, y_res

# Objective function for Optuna
def objective(trial):
    # Define the hyperparameters to optimize
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # Create the XGBoost model
    model = XGBClassifier(**param)
    
    # Train the model with augmented and normalized data
    model.fit(X_train_VIAB_aug[selected_features], y_train_VIAB_aug)
    
    # Predict on the validation set using the selected features
    y_pred = model.predict(X_test_VIAB_norm[selected_features])
    
    # Return the ROC AUC score for evaluation
    return roc_auc_score(y_test_VIAB, model.predict_proba(X_test_VIAB_norm[selected_features])[:, 1])

if __name__ == '__main__':
    
    print("Loading data from pickle files...")
    
    # Loading viability
    with open('docs_analysed.pkl', 'rb') as f:
        docs_analyzed_VIAB = pickle.load(f)
    with open('docs_meas.pkl', 'rb') as f:
        docs_meas_VIAB = pickle.load(f)
    # Loading AMP
    with open('docs_analysed_AMP.pkl', 'rb') as f:
        docs_analyzed_AMP = pickle.load(f)
    with open('docs_meas_AMP.pkl', 'rb') as f:
        docs_meas_AMP = pickle.load(f)
    print("Data loaded successfully.")
    print(docs_analyzed_AMP[0], docs_meas_AMP[0])
    norm_docs_VIAB = extract_features_VIAB(docs_analyzed_VIAB)
    print('first dataset', norm_docs_VIAB.iloc[0])
    norm_docs_AMP = normalize_docs_AMP(docs_analyzed_AMP, docs_meas_AMP)
    print('second dataset', norm_docs_AMP.iloc[0])
    
    X_VIAB = norm_docs_VIAB.drop(columns=['dead'])
    y_VIAB = norm_docs_VIAB['dead']
    
    # Perform a stratified split to maintain the same ratio of classes in both training and testing sets
    X_train_VIAB, X_test_VIAB, y_train_VIAB, y_test_VIAB = train_test_split(
        X_VIAB, y_VIAB, test_size=0.2, stratify=y_VIAB, random_state=42
    )
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_VIAB_norm = scaler.fit_transform(X_train_VIAB[selected_features])
    X_test_VIAB_norm = scaler.transform(X_test_VIAB[selected_features])
    X_norm_docs_AMP = scaler.transform(norm_docs_AMP[selected_features])

    # Convert back to DataFrame for easier handling
    X_train_VIAB_norm = pd.DataFrame(X_train_VIAB_norm, columns=selected_features)
    X_test_VIAB_norm = pd.DataFrame(X_test_VIAB_norm, columns=selected_features)
    X_norm_docs_AMP = pd.DataFrame(X_norm_docs_AMP, columns=selected_features)
    
    print("Training set class distribution:\n", y_train_VIAB.value_counts())
    print("Test set class distribution:\n", y_test_VIAB.value_counts())
    
    # Data Augmentation
    X_train_VIAB_aug, y_train_VIAB_aug = augment_data(X_train_VIAB_norm, y_train_VIAB)
    
    # Use Optuna to find the best hyperparameters for XGBoost
    sampler = TPESampler(seed=42)  # Use TPE for efficient sampling
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=150, timeout=600)  # Adjust n_trials and timeout as needed
    
    print("Best hyperparameters:", study.best_params)
    
    # Train the best model with the found hyperparameters and augmented data
    best_model = XGBClassifier(**study.best_params)
    best_model.fit(X_train_VIAB_aug[selected_features], y_train_VIAB_aug)
    
    # Evaluate the best model
    y_pred_VIAB = best_model.predict(X_test_VIAB_norm[selected_features])
    print("Accuracy:", accuracy_score(y_test_VIAB, y_pred_VIAB))
    print("Classification Report:\n", classification_report(y_test_VIAB, y_pred_VIAB))
    
    # Predict the 'dead' status for the second dataset
    predicted_dead = best_model.predict(X_norm_docs_AMP[selected_features])
    
    # Add the predicted dead status back to the DataFrame
    norm_docs_AMP['predicted_dead'] = predicted_dead

    # Combine with antibiotics_quantity
    antibiotics_quantity = norm_docs_AMP['antibiotics_quantity']
    norm_docs_AMP['antibiotics_quantity'] = antibiotics_quantity
    print(norm_docs_AMP)
    # Group by antibiotics_quantity and count the number of 0s and 1s
    death_counts = norm_docs_AMP.groupby(['antibiotics_quantity', 'predicted_dead']).size().unstack(fill_value=0)
    death_counts.plot(kind='bar', stacked=False, color=['blue', 'yellow'])
    plt.xlabel('Antibiotics Quantity')
    plt.ylabel('Number of Samples')
    plt.title('Predicted Death vs Alive by Antibiotics Quantity')
    plt.legend(['Alive (0)', 'Dead (1)'])
    plt.savefig('bar_plot.png')  
    plt.close()
