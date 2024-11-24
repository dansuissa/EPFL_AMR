from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

from feature_extraction import extract_features_model2
from data_processing import split_data
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd



"""
Model training

Functions:
    drop_redundant_feature(X, feature_names)
    repeated_random_splitting_evaluation(docs, iterations=)
    baseline_model_model1(normalized_transmission, interval_min, interval_max)
    predict_bacteria_family_model1(normalized_transmission, cluster_centers)
"""

#function to drop redundent feature
def drop_redundant_feature(X, feature_names):
    #convert X to DataFrame for easy manipulation
    X_df = pd.DataFrame(X, columns=feature_names)
    #drop the less useful highly correlated feature
    feature_to_drop = 'min_transmission'  # Adjust based on your findings
    X_df = X_df.drop(columns=[feature_to_drop])
    #return the updated values and feature names
    updated_feature_names = [f for f in feature_names if f != feature_to_drop]
    return X_df.values, updated_feature_names




#function not used anymore but might be useful later for other datasets.
def repeated_random_splitting_evaluation(docs, iterations=2):
    """
    performs repeated random shuffling and cross-validation to evaluate model performance
    this function takes a dataset, shuffles it multiple times, splits it into training and testing sets,
    extracts features, trains RandomForest classifiers for Gram type, shape type, and bacteria type classification,
    and evaluates the classifiers using cross-validation.
    iterations: number of times to shuffle and evaluate the dataset.
    """
    gram_accuracies = []
    shape_accuracies = []
    bacteria_accuracies = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i in range(iterations):
        print(f"Iteration: {i + 1}/{iterations}")
        shuffled_docs = shuffle(docs, random_state=i)
        train_docs, test_docs = split_data(shuffled_docs)
        train_features, train_gram_labels, train_bacteria_labels, train_shape_labels = extract_features_model2(
            train_docs, use_shape_type_as_feature=True)
        test_features, test_gram_labels, test_bacteria_labels, test_shape_labels = extract_features_model2(test_docs,
                                                                                                           use_shape_type_as_feature=True)

        label_encoder_bacteria = LabelEncoder()
        encoded_train_bacteria_labels = label_encoder_bacteria.fit_transform(train_bacteria_labels)
        encoded_test_bacteria_labels = label_encoder_bacteria.transform(test_bacteria_labels)

        label_encoder_gram = LabelEncoder()
        encoded_train_gram_labels = label_encoder_gram.fit_transform(train_gram_labels)
        encoded_test_gram_labels = label_encoder_gram.transform(test_gram_labels)

        label_encoder_shape = LabelEncoder()
        encoded_train_shape_labels = label_encoder_shape.fit_transform(train_shape_labels)
        encoded_test_shape_labels = label_encoder_shape.transform(test_shape_labels)

        #For Gram type classification, use features without shape type
        X_train_gram = train_features
        X_test_gram = test_features
        y_train_gram = train_gram_labels
        y_test_gram = test_gram_labels

        #For Shape type classification
        X_train_shape = train_features
        X_test_shape = test_features
        y_train_shape = encoded_train_shape_labels
        y_test_shape = encoded_test_shape_labels

        #For Bacteria type classification, use features with Gram type only
        X_train_bacteria = np.hstack((train_features, encoded_train_gram_labels.reshape(-1, 1)))
        X_test_bacteria = np.hstack((test_features, encoded_test_gram_labels.reshape(-1, 1)))
        y_train_bacteria = encoded_train_bacteria_labels
        y_test_bacteria = encoded_test_bacteria_labels

        #Gram type classifier
        gram_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=3, max_features='sqrt',
            min_samples_split=20, min_samples_leaf=10, max_samples=0.8
        )
        gram_classifier.fit(X_train_gram, y_train_gram)
        gram_cv_scores = cross_val_score(gram_classifier, X_train_gram, y_train_gram, cv=skf, scoring='accuracy')
        gram_accuracies.append(gram_cv_scores.mean())
        train_accuracy_gram = gram_classifier.score(X_train_gram, y_train_gram)
        print(f"Training accuracy for Gram type classifier in iteration {i + 1}: {train_accuracy_gram:.2f}")

        #Shape type classifier
        shape_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=3, max_features='sqrt',
            min_samples_split=20, min_samples_leaf=10, max_samples=0.8
        )
        shape_classifier.fit(X_train_shape, y_train_shape)
        shape_cv_scores = cross_val_score(shape_classifier, X_train_shape, y_train_shape, cv=skf, scoring='accuracy')
        shape_accuracies.append(shape_cv_scores.mean())
        train_accuracy_shape = shape_classifier.score(X_train_shape, y_train_shape)
        print(f"Training accuracy for Shape type classifier in iteration {i + 1}: {train_accuracy_shape:.2f}")

        #Bacteria type classifier
        bacteria_classifier = RandomForestClassifier(
            n_estimators=300, random_state=42, max_depth=4, max_features='sqrt',
            min_samples_split=20, min_samples_leaf=10, max_samples=0.8
        )
        bacteria_classifier.fit(X_train_bacteria, y_train_bacteria)
        bacteria_cv_scores = cross_val_score(bacteria_classifier, X_train_bacteria, y_train_bacteria, cv=skf,
                                             scoring='accuracy')
        bacteria_accuracies.append(bacteria_cv_scores.mean())
        train_accuracy_bacteria = bacteria_classifier.score(X_train_bacteria, y_train_bacteria)
        print(f"Training accuracy for Bacteria type classifier in iteration {i + 1}: {train_accuracy_bacteria:.2f}")

    return gram_accuracies, shape_accuracies, bacteria_accuracies


#model 1 based on theory will deduce the gram type using the mean normalized transmission
def baseline_model_model1(normalized_transmission, interval_min, interval_max):
    mean_value = np.mean(normalized_transmission)
    if interval_min >= mean_value:
        return 1  #Gram-positive
    elif mean_value >= interval_max:
        return 0  #Gram-negative
    else:
        if interval_max - mean_value >= mean_value - interval_min:
            return 1
        else:
            return 0

#predicts the bacteria strain by calculating the nearest cluster and assinig the prediction to the family it represents
def predict_bacteria_family_model1(normalized_transmission, cluster_centers):
    mean_transmission = np.mean(normalized_transmission)
    std_transmission = np.std(normalized_transmission)
    min_distance = float('inf')
    predicted_bacteria = None
    for bacteria, center in cluster_centers.items():
        distance = np.sqrt((mean_transmission - center[0]) ** 2 + (std_transmission - center[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            predicted_bacteria = bacteria
    return predicted_bacteria

