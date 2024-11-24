from Z_HELPERS.DM_communication import get_docs_VM
import numpy as np
from scipy.fftpack import fft, ifft
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from collections import Counter


def compute_cross_distance_fft(transmission):
    T = len(transmission)
    f = fft(transmission)
    power_spectrum = np.abs(f) ** 2
    autocorrelation = ifft(power_spectrum).real
    cross_distances = np.sqrt(np.sum((transmission - autocorrelation[:T]) ** 2))
    return np.min(cross_distances)


def normalize_data(docs, sequence_length=1000):
    normalized_docs = []
    for doc in docs:
        transmission = np.array(doc['data']['transmission'])
        norm_factor = doc['normalization_factor']
        transmission_normalized = transmission / norm_factor

        if len(transmission_normalized) > sequence_length:
            transmission_normalized = transmission_normalized[:sequence_length]
        else:
            transmission_normalized = np.pad(transmission_normalized,
                                             (0, max(0, sequence_length - len(transmission_normalized))), 'constant')

        cross_distance = compute_cross_distance_fft(transmission_normalized)

        normalized_docs.append({
            'transmission_normalized': transmission_normalized,
            'cross_distance': cross_distance,
            'label': doc['gram_type'],
            'bacteria': doc['bacteria'],
            'name': doc['name']
        })
    return normalized_docs


def chunk_time_series(docs, num_chunks=6):
    chunked_docs = []
    for doc in docs:
        transmission = np.array(doc['transmission_normalized'])
        original_length = len(transmission)
        chunk_size = original_length // num_chunks
        for i in range(num_chunks):
            chunk = transmission[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) != chunk_size:
                raise ValueError(f"Chunk size mismatch: expected {chunk_size}, got {len(chunk)}")
            chunked_docs.append({
                'transmission_normalized': chunk,
                'cross_distance': compute_cross_distance_fft(chunk),
                'label': doc['label'],
                'bacteria': doc['bacteria'],
                'name': f"{doc['name']}_chunk_{i}"
            })
    return chunked_docs


def augment_data(docs, noise_level=0.05, shift_max=10):
    augmented_docs = []
    for doc in docs:
        transmission = np.array(doc['transmission_normalized'])
        original_shape = transmission.shape
        for _ in range(2):
            noisy_transmission = transmission + np.random.normal(0, noise_level, len(transmission))
            if noisy_transmission.shape != original_shape:
                raise ValueError(
                    f"Shape mismatch after adding noise: original {original_shape}, noisy {noisy_transmission.shape}")
            augmented_docs.append({
                'transmission_normalized': noisy_transmission,
                'cross_distance': compute_cross_distance_fft(noisy_transmission),
                'label': doc['label'],
                'bacteria': doc['bacteria'],
                'name': doc['name'] + '_noise'
            })
            shift = np.random.randint(-shift_max, shift_max)
            shifted_transmission = np.roll(transmission, shift)
            if shifted_transmission.shape != original_shape:
                raise ValueError(
                    f"Shape mismatch after shifting: original {original_shape}, shifted {shifted_transmission.shape}")
            augmented_docs.append({
                'transmission_normalized': shifted_transmission,
                'cross_distance': compute_cross_distance_fft(shifted_transmission),
                'label': doc['label'],
                'bacteria': doc['bacteria'],
                'name': doc['name'] + '_shift'
            })
    return augmented_docs


def stratified_split_data(docs, test_size=0.2):
    bacteria_types = list(set(doc['bacteria'] for doc in docs))
    train_docs, test_docs = [], []
    for btype in bacteria_types:
        btype_docs = [doc for doc in docs if doc['bacteria'] == btype]
        train_split, test_split = train_test_split(btype_docs, test_size=test_size, random_state=42, shuffle=True)
        train_docs.extend(train_split)
        test_docs.extend(test_split)
    return train_docs, test_docs


def print_bacteria_distribution(docs, set_name):
    bacteria_count = Counter(doc['bacteria'] for doc in docs)
    print(f"{set_name} set bacteria distribution:")
    for bacteria, count in bacteria_count.items():
        print(f"{bacteria}: {count}")


def objective_xgb_bacteria(trial):
    param = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_train_bacteria)),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    xgb = XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []

    for train_index, valid_index in skf.split(X_train_bacteria, y_train_bacteria):
        X_train_fold, X_valid_fold = X_train_bacteria[train_index], X_train_bacteria[valid_index]
        y_train_fold, y_valid_fold = y_train_bacteria[train_index], y_train_bacteria[valid_index]

        xgb.fit(X_train_fold, y_train_fold)
        preds = xgb.predict(X_valid_fold)
        accuracy = accuracy_score(y_valid_fold, preds)
        accuracies.append(accuracy)

    return np.mean(accuracies)
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


if __name__ == '__main__':
    docs = get_docs_VM(db_name='Optical_Trapping_ML_DB', coll_name='tseries_Gram_analysis', query={})
    modelnb = input("Type the number of the model you want to use: ")

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

    if modelnb == '1':
        feature_names = np.array(
            ['transmission_normalized', 'cross_distance_fft', 'gram_type'])
        normalized_docs = normalize_data(docs, sequence_length=1000)
        filtered_docs = [doc for doc in normalized_docs if
                         doc['name'] not in problematic_names and doc['bacteria'] != 'pp6']

        # Chunk and augment data
        chunked_docs = chunk_time_series(filtered_docs, num_chunks=6)
        augmented_docs = augment_data(chunked_docs, noise_level=0.05, shift_max=10)

        train_docs, test_docs = stratified_split_data(augmented_docs)

        # Print bacteria distribution in each set
        print_bacteria_distribution(train_docs, "Training")
        print_bacteria_distribution(test_docs, "Testing")

        X_train_bacteria = np.array(
            [np.concatenate((doc['transmission_normalized'], [doc['cross_distance'], doc['label']])) for doc in
             train_docs])
        y_train_bacteria = np.array([doc['bacteria'] for doc in train_docs])
        X_test_bacteria = np.array(
            [np.concatenate((doc['transmission_normalized'], [doc['cross_distance'], doc['label']])) for doc in
             test_docs])
        y_test_bacteria = np.array([doc['bacteria'] for doc in test_docs])

        label_encoder = LabelEncoder()
        y_train_bacteria = label_encoder.fit_transform(y_train_bacteria)
        y_test_bacteria = label_encoder.transform(y_test_bacteria)

        study_bacteria = optuna.create_study(direction='maximize')
        study_bacteria.optimize(objective_xgb_bacteria, n_trials=50)

        print(f"Best trial (Bacteria type): {study_bacteria.best_trial.value}")
        print("Best hyperparameters (Bacteria type): ", study_bacteria.best_trial.params)

        # Train the final model with the best parameters for Bacteria type classification
        best_params_bacteria = study_bacteria.best_trial.params
        xgb_bacteria = XGBClassifier(**best_params_bacteria, use_label_encoder=False, eval_metric='mlogloss',
                                     objective='multi:softmax', num_class=len(np.unique(y_train_bacteria)))
        xgb_bacteria.fit(X_train_bacteria, y_train_bacteria)

        # Calculate and print training accuracy for Bacteria type classifier
        train_accuracy_bacteria = xgb_bacteria.score(X_train_bacteria, y_train_bacteria)
        print(f'Training accuracy for Bacteria type classifier: {train_accuracy_bacteria:.2f}')

        y_pred_bacteria = xgb_bacteria.predict(X_test_bacteria)
        print("Bacteria Type Classification Report:")
        print(classification_report(y_test_bacteria, y_pred_bacteria, target_names=label_encoder.classes_,
                                    zero_division=0))

        # Cross-validation for Bacteria type classifier
        bacteria_cv_scores = cross_val_score(xgb_bacteria, X_train_bacteria, y_train_bacteria, cv=5, scoring='accuracy')
        print(
            f'Bacteria type classifier cross-validation accuracy: {bacteria_cv_scores.mean():.2f} ± {bacteria_cv_scores.std():.2f}')

        # Check for overfitting/underfitting
        test_accuracy_bacteria = xgb_bacteria.score(X_test_bacteria, y_test_bacteria)
        print(f'Test accuracy for Bacteria type classifier: {test_accuracy_bacteria:.2f}')
        plot_feature_importance(xgb_bacteria, feature_names)
        if train_accuracy_bacteria > test_accuracy_bacteria:
            print("Potential overfitting detected in Bacteria type classifier.")
        elif train_accuracy_bacteria < test_accuracy_bacteria:
            print("Potential underfitting detected in Bacteria type classifier.")

        # Plot confusion matrix


        cm = confusion_matrix(y_test_bacteria, y_pred_bacteria)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix for Bacteria Type Classification')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
