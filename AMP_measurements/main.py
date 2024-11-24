import pandas as pd
import pickle
import numpy as np
from data_processing import normalize_docs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# Custom loss function with increased penalty for critical levels
def custom_loss_function(y_true, y_pred):
    weights = tf.where(y_true == 2, 2.0, 1.0)  # Penalize more for 2 µg/mL
    weights = tf.where(y_true == 4, 2.0, weights)  # Penalize more for 4 µg/mL
    loss = SparseCategoricalCrossentropy()(y_true, y_pred)
    return tf.reduce_mean(loss * weights)

# Custom GAN for data augmentation
class GAN:
    def __init__(self, input_dim, noise_dim):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=self.noise_dim))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.input_dim, activation='tanh'))
        return model

    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=self.input_dim))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = models.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, X_train, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            # Train discriminator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            gen_data = self.generator.predict(noise)
            real_data = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            X = np.vstack((real_data, gen_data))
            y = np.ones((2 * batch_size, 1))
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(X, y)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            y_gen = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, y_gen)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]')

    def generate(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        gen_data = self.generator.predict(noise)
        return gen_data

def stratified_split_with_antibiotics(docs, test_size=0.2, val_size=0.1):
    known_docs = [doc for doc in docs if doc['living_state'] is not None]
    unknown_docs = [doc for doc in docs if doc['living_state'] is None]

    antibiotics_quantities = [doc['antibiotics_quantity'] for doc in known_docs]
    if len(known_docs) > 0:
        train_val_docs, test_docs = train_test_split(
            known_docs, test_size=test_size, stratify=antibiotics_quantities, random_state=42)
    else:
        train_val_docs = []
        test_docs = []

    train_val_antibiotics = [doc['antibiotics_quantity'] for doc in train_val_docs]
    if len(train_val_docs) > 0:
        train_docs, val_docs = train_test_split(
            train_val_docs, test_size=val_size / (1 - test_size), stratify=train_val_antibiotics, random_state=42)
    else:
        train_docs = []
        val_docs = []

    return train_docs, val_docs, test_docs, unknown_docs

def prepare_features_with_antibiotics(docs):
    X = []
    y = []
    antibiotics = []

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

        X.append(features)
        antibiotics.append(doc['antibiotics_quantity'])
        if doc['living_state'] is not None:
            y.append(doc['living_state'])

    return pd.DataFrame(X), pd.Series(y) if y else None, antibiotics

def build_simple_transformer_model(input_shape):
    input_layer = layers.Input(shape=(input_shape,))
    x = layers.Dense(512, activation='relu')(input_layer)
    x = layers.Dense(256, activation='relu')(x)
    output_layer = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

def build_voting_ensemble(X_train, y_train):
    knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Hyperparameter tuning for XGBoost
    param_grid = {
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__n_estimators': [100, 200, 300]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_xgb_model = grid_search.best_estimator_
    
    # Fit individual models
    knn_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # Create a weighted voting classifier
    ensemble_model = VotingClassifier(estimators=[
        ('knn', knn_model),
        ('svm', svm_model),
        ('xgb', best_xgb_model)
    ], voting='soft', weights=[1, 2, 3])
    
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model

if __name__ == '__main__':
    print("Loading data from pickle file...")
    with open('data_analysed.pkl', 'rb') as f:
        docs_analyzed = pickle.load(f)
    with open('data_meas.pkl', 'rb') as f:
        docs_meas = pickle.load(f)
    print("Data loaded successfully.")
    
    norm_docs = normalize_docs(docs_analyzed, docs_meas)
    train_docs, val_docs, test_docs, unknown_docs = stratified_split_with_antibiotics(norm_docs, test_size=0.2, val_size=0.1)

    X_train, y_train, train_antibiotics = prepare_features_with_antibiotics(train_docs)
    X_val, y_val, val_antibiotics = prepare_features_with_antibiotics(val_docs)
    X_test, y_test, test_antibiotics = prepare_features_with_antibiotics(test_docs)
    X_unknown, _, unknown_antibiotics = prepare_features_with_antibiotics(unknown_docs)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    X_unknown_imputed = imputer.transform(X_unknown)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_unknown_scaled = scaler.transform(X_unknown_imputed)

    # Data augmentation using GAN
    gan = GAN(input_dim=X_train_scaled.shape[1], noise_dim=100)
    gan.train(X_train_scaled, epochs=5000, batch_size=32)
    X_train_augmented = gan.generate(num_samples=1000)
    y_train_augmented = np.random.choice(y_train.unique(), size=1000, replace=True)
    
    X_train_combined = np.vstack((X_train_scaled, X_train_augmented))
    y_train_combined = np.concatenate((y_train.values, y_train_augmented))
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train_combined)
    
    # Transformer model
    transformer_model = build_simple_transformer_model(X_train_smote.shape[1])
    optimizer = optimizers.Adam(learning_rate=5e-5)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    transformer_model.compile(optimizer=optimizer, loss=custom_loss_function, metrics=['accuracy'])
    transformer_model.fit(
        X_train_smote,
        y_train_smote,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping]
    )

    # Ensemble Model
    ensemble_model = build_voting_ensemble(X_train_smote, y_train_smote)

    y_unknown_pred_transformer = transformer_model.predict(X_unknown_scaled)
    y_unknown_pred_transformer = np.argmax(y_unknown_pred_transformer, axis=1)

    y_unknown_pred_ensemble = ensemble_model.predict(X_unknown_scaled)
    
    # Combining the predictions using weighted voting
    y_unknown_pred = (0.4 * y_unknown_pred_transformer + 0.6 * y_unknown_pred_ensemble).astype(int)

    for doc, pred in zip(unknown_docs, y_unknown_pred):
        doc['living_state'] = int(pred)

    # Calculate and print the percentages of dead bacteria for each antibiotic level
    print("\nCombined Model Predictions for Each Antibiotic Level:")
    antibiotic_levels = set(doc['antibiotics_quantity'] for doc in unknown_docs)
    for level in sorted(antibiotic_levels):
        total = sum(1 for doc in unknown_docs if doc['antibiotics_quantity'] == level)
        dead = sum(1 for doc in unknown_docs if doc['antibiotics_quantity'] == level and doc['living_state'] == 1)
        dead_percentage = (dead / total) * 100 if total > 0 else 0
        print(f"Antibiotic Level {level} µg/mL: {dead_percentage:.2f}% Dead")
