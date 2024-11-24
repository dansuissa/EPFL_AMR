import pandas as pd
import pickle
import numpy as np
from data_processing import normalize_docs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def stratified_split(docs, test_size=0.2, val_size=0.2):
    train_val_docs, test_docs = train_test_split(
        docs, test_size=test_size, stratify=[doc['antibiotics_quantity'] for doc in docs if doc['antibiotics_quantity'] is not None], random_state=42)
    
    train_val_antibiotics = [doc['antibiotics_quantity'] for doc in train_val_docs if doc['antibiotics_quantity'] is not None]
    train_docs, val_docs = train_test_split(
        train_val_docs, test_size=val_size / (1 - test_size), stratify=train_val_antibiotics, random_state=42)

    return train_docs, val_docs, test_docs

def prepare_features(docs):
    X = []
    y = []

    for doc in docs:
        features = {
            'mean_trap': doc.get('mean_trap'),
            'std_trap': doc.get('std_trap'),
            'max_trap': doc.get('max_trap'),
            'most_prob_trap': doc.get('most_prob_trap'),
            'mean_on_to_trapping': doc.get('mean_on_to_trapping'),
            'std_on_to_trapping': doc.get('std_on_to_trapping'),
            'q25_on_to_trapping': doc.get('q25_on_to_trapping'),
            'min_on_to_trapping': doc.get('min_on_to_trapping'),
            'max_on_to_trapping': doc.get('max_on_to_trapping'),
            'most_prob_on_to_trapping': doc.get('most_prob_on_to_trapping')
        }

        if doc['antibiotics_quantity'] is not None:
            X.append(features)
            y.append(doc['antibiotics_quantity'])

    return pd.DataFrame(X), pd.Series(y)

def build_xgboost_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')

    # Hyperparameter tuning for XGBoost
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

def build_generator(latent_dim, n_features):
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(n_features, activation='tanh')
    ])
    return model

def build_discriminator(n_features):
    model = Sequential([
        Dense(512, input_dim=n_features),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    z = Input(shape=(latent_dim,))
    generated_features = generator(z)
    discriminator.trainable = False
    validity = discriminator(generated_features)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return combined

def train_gan(generator, discriminator, combined, X_train, epochs, batch_size=128, sample_interval=50):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_features = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_features = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_features, valid)
        d_loss_fake = discriminator.train_on_batch(generated_features, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

def augment_data_with_gan(generator, X_train, n_samples):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    synthetic_samples = generator.predict(noise)
    X_train_augmented = np.vstack([X_train, synthetic_samples])
    return X_train_augmented

if __name__ == '__main__':
    print("Loading data from pickle file...")
    with open('data_analysed.pkl', 'rb') as f:
        docs_analyzed = pickle.load(f)
    with open('data_meas.pkl', 'rb') as f:
        docs_meas = pickle.load(f)
    print("Data loaded successfully.")
    
    norm_docs = normalize_docs(docs_analyzed, docs_meas)
    train_docs, val_docs, test_docs = stratified_split(norm_docs, test_size=0.2, val_size=0.1)

    X_train, y_train = prepare_features(train_docs)
    X_val, y_val = prepare_features(val_docs)
    X_test, y_test = prepare_features(test_docs)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Map antibiotic levels to a range of integers
    antibiotic_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 16: 4, 32: 5}
    reverse_mapping = {v: k for k, v in antibiotic_mapping.items()}
    
    y_train_mapped = y_train.map(antibiotic_mapping)
    y_val_mapped = y_val.map(antibiotic_mapping)
    y_test_mapped = y_test.map(antibiotic_mapping)

    # Check for NaN values in the target variable and remove them
    y_train_mapped = y_train_mapped.dropna()
    y_val_mapped = y_val_mapped.dropna()
    y_test_mapped = y_test_mapped.dropna()

    # Align X and y after dropping NaNs
    X_train_scaled = X_train_scaled[y_train_mapped.index]
    X_val_scaled = X_val_scaled[y_val_mapped.index]
    X_test_scaled = X_test_scaled[y_test_mapped.index]

    # Parameters for GAN
    latent_dim = 100
    n_features = X_train_scaled.shape[1]
    generator = build_generator(latent_dim, n_features)
    discriminator = build_discriminator(n_features)
    combined = build_gan(generator, discriminator)

    # Train the GAN
    train_gan(generator, discriminator, combined, X_train_scaled, epochs=10000, batch_size=32, sample_interval=1000)

    # Augment data using the trained GAN
    X_train_augmented = augment_data_with_gan(generator, X_train_scaled, n_samples=len(X_train_scaled) // 2)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_augmented, y_train_mapped)

    # XGBoost Model
    xgb_model = build_xgboost_model(X_train_balanced, y_train_balanced)
    
    # Evaluate on validation set
    val_preds = xgb_model.predict(X_val_scaled)
    val_preds_original = pd.Series(val_preds).map(reverse_mapping)
    val_accuracy = accuracy_score(y_val_mapped.map(reverse_mapping), val_preds_original)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print("Validation Classification Report:")
    print(classification_report(y_val_mapped.map(reverse_mapping), val_preds_original, target_names=['0', '1', '2', '4', '16', '32']))

    # Evaluate on test set
    test_preds = xgb_model.predict(X_test_scaled)
    test_preds_original = pd.Series(test_preds).map(reverse_mapping)
    test_accuracy = accuracy_score(y_test_mapped.map(reverse_mapping), test_preds_original)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print("Test Classification Report:")
    print(classification_report(y_test_mapped.map(reverse_mapping), test_preds_original, target_names=['0', '1', '2', '4', '16', '32']))
    
    # Save the model
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    print("Model saved successfully.")
