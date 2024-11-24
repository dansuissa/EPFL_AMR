import pickle
from data_processing import *
from visu import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score,log_loss
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':
    print("Loading data from pickle file...")
    with open('data_analysed.pkl', 'rb') as f:
        docs_analyzed = pickle.load(f)
    with open('data_meas.pkl', 'rb') as f:
        docs_meas = pickle.load(f)
    print("Data loaded successfully.")
    norm_docs = normalize_docs(docs_analyzed, docs_meas)
    train_docs, val_docs, test_docs, unknown_docs = stratified_split(norm_docs)

    # Prepare features for training, validation, and test sets
    X_train, y_train = prepare_features(train_docs)
    X_val, y_val = prepare_features(val_docs)
    X_test, y_test = prepare_features(test_docs)
    X_unknown, _ = prepare_features(unknown_docs)

    # Train the initial model
    model_initial = RandomForestClassifier(random_state=42)
    model_initial.fit(X_train, y_train)

    # Validate the initial model
    y_val_pred_initial = model_initial.predict(X_val)
    val_accuracy_initial = accuracy_score(y_val, y_val_pred_initial)
    print(f'Initial Validation Accuracy: {val_accuracy_initial}')

    # Predict living_state for unknown_docs with the initial model
    y_unknown_pred_initial = model_initial.predict(X_unknown)

    # Update unknown_docs with predicted living_state
    for doc, pred in zip(unknown_docs, y_unknown_pred_initial):
        doc['living_state'] = pred

    # Optionally, evaluate the initial model on the test set
    y_test_pred_initial = model_initial.predict(X_test)
    test_accuracy_initial = accuracy_score(y_test, y_test_pred_initial)
    print(f'Initial Test Accuracy: {test_accuracy_initial}')

    # Plot feature importances for the initial model
    plot_feature_importances(model_initial, X_train)

    # Plot feature correlations for the initial feature set
    plot_feature_correlations(X_train)

    # Remove highly correlated features
    X_train_reduced = remove_highly_correlated_features(X_train, model_initial)
    X_val_reduced = X_val[X_train_reduced.columns]
    X_test_reduced = X_test[X_train_reduced.columns]
    X_unknown_reduced = X_unknown[X_train_reduced.columns]

    # Train a new model on the reduced feature set
    model_reduced = RandomForestClassifier(random_state=42)
    model_reduced.fit(X_train_reduced, y_train)

    # Validate the new model
    y_val_pred_reduced = model_reduced.predict(X_val_reduced)
    val_accuracy_reduced = accuracy_score(y_val, y_val_pred_reduced)
    print(f'Reduced Validation Accuracy: {val_accuracy_reduced}')

    # Predict living_state for unknown_docs with the reduced model
    y_unknown_pred_reduced = model_reduced.predict(X_unknown_reduced)

    # Update unknown_docs with predicted living_state
    for doc, pred in zip(unknown_docs, y_unknown_pred_reduced):
        doc['living_state'] = pred

    # Optionally, evaluate the reduced model on the test set
    y_test_pred_reduced = model_reduced.predict(X_test_reduced)
    test_accuracy_reduced = accuracy_score(y_test, y_test_pred_reduced)
    print(f'Reduced Test Accuracy: {test_accuracy_reduced}')

    # Plot feature importances for the reduced model
    plot_feature_importances(model_reduced, X_train_reduced)

    # Plot feature correlations for the reduced feature set
    plot_feature_correlations(X_train_reduced)

    # Remove additional highly correlated features
    X_train_reduced_v2 = remove_highly_correlated_features_v2(X_train_reduced, model_reduced)
    X_val_reduced_v2 = X_val[X_train_reduced_v2.columns]
    X_test_reduced_v2 = X_test[X_train_reduced_v2.columns]
    X_unknown_reduced_v2 = X_unknown[X_train_reduced_v2.columns]

    # Train a new model on the further reduced feature set
    model_reduced_v2 = RandomForestClassifier(random_state=42)
    model_reduced_v2.fit(X_train_reduced_v2, y_train)

    # Validate the new model
    y_val_pred_reduced_v2 = model_reduced_v2.predict(X_val_reduced_v2)
    val_accuracy_reduced_v2 = accuracy_score(y_val, y_val_pred_reduced_v2)
    print(f'Further Reduced Validation Accuracy: {val_accuracy_reduced_v2}')

    # Predict living_state for unknown_docs with the further reduced model
    y_unknown_pred_reduced_v2 = model_reduced_v2.predict(X_unknown_reduced_v2)

    # Update unknown_docs with predicted living_state
    for doc, pred in zip(unknown_docs, y_unknown_pred_reduced_v2):
        doc['living_state'] = pred

    # Optionally, evaluate the further reduced model on the test set
    y_test_pred_reduced_v2 = model_reduced_v2.predict(X_test_reduced_v2)
    test_accuracy_reduced_v2 = accuracy_score(y_test, y_test_pred_reduced_v2)
    print(f'Further Reduced Test Accuracy: {test_accuracy_reduced_v2}')

    # Plot feature importances for the further reduced model
    plot_feature_importances(model_reduced_v2, X_train_reduced_v2)

    # Plot feature correlations for the further reduced feature set
    plot_feature_correlations(X_train_reduced_v2)

    # Cross-validation on the further reduced model
    cv_scores = cross_val_score(model_reduced_v2, X_train_reduced_v2, y_train, cv=5)
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Cross-Validation Mean Score: {np.mean(cv_scores)}')
    # Precision, Recall, and F1 Score for the test set
    precision = precision_score(y_test, y_test_pred_reduced_v2)
    recall = recall_score(y_test, y_test_pred_reduced_v2)
    f1 = f1_score(y_test, y_test_pred_reduced_v2)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    # ROC and AUC for the test set
    y_test_proba = model_reduced_v2.predict_proba(X_test_reduced_v2)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    print(f'Test AUC: {roc_auc}')

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    from sklearn.metrics import precision_recall_curve

    # Precision-Recall Curve for the test set
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    from sklearn.utils import resample

    # Bootstrapping for the Further Reduced Model
    n_iterations = 1000
    n_size = int(len(X_train_reduced_v2) * 0.5)
    bootstrap_scores = []
    y_train = pd.Series(y_train)
    for i in range(n_iterations):
        X_sample, y_sample = resample(X_train_reduced_v2, y_train, n_samples=n_size)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_sample, y_sample)
        y_pred_sample = model.predict(X_test_reduced_v2)
        score = accuracy_score(y_test, y_pred_sample)
        bootstrap_scores.append(score)

    bootstrap_mean = np.mean(bootstrap_scores)
    bootstrap_std = np.std(bootstrap_scores)
    print(f'Bootstrapped Accuracy Mean: {bootstrap_mean}, Std: {bootstrap_std}')

    from sklearn.model_selection import StratifiedKFold

    # Stratified k-Fold Cross-Validation for the Further Reduced Model
    skf = StratifiedKFold(n_splits=5)
    cv_scores = []

    for train_index, test_index in skf.split(X_train_reduced_v2, y_train):
        X_train_fold, X_val_fold = X_train_reduced_v2.iloc[train_index], X_train_reduced_v2.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred_fold)
        cv_scores.append(score)

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f'k-Fold Cross-Validation Mean Score: {cv_mean}, Std: {cv_std}')

    # Plot relationships between features and living_state
    sns.boxplot(x='living_state', y='antibiotics_quantity',
                data=pd.concat([X_train_reduced_v2, pd.Series(y_train, name='living_state')], axis=1))
    plt.title('Antibiotics Quantity vs. Living State')
    plt.show()

    sns.boxplot(x='living_state', y='min_trap',
                data=pd.concat([X_train_reduced_v2, pd.Series(y_train, name='living_state')], axis=1))
    plt.title('Min Trap vs. Living State')
    plt.show()

    plot_classifications_by_antibiotic_level_with_counts(unknown_docs + train_docs + val_docs + test_docs)
    plot_classifications_by_antibiotic_level(unknown_docs + train_docs + val_docs + test_docs, model_reduced_v2,
                                             X_train_reduced_v2.columns)

    # Perform active learning query
    query_indices = active_learning_query(model_reduced_v2, X_unknown_reduced_v2)
    queried_samples = [unknown_docs[i] for i in query_indices]

    print("Queried samples for manual labeling:", query_indices)


    # Simulate manual labeling (In reality, this would be done by domain experts)
    def get_manual_labels(samples):
        manual_labels = []
        for sample in samples:
            # Here we simulate the manual labeling process.
            # In a real scenario, you would replace this with actual manual labeling.
            if sample['antibiotics_quantity'] > 16:
                manual_labels.append(1)  # Assume high antibiotic levels lead to death
            else:
                manual_labels.append(0)  # Assume low antibiotic levels lead to alive
        return manual_labels


    # Get manual labels for the queried samples
    new_labels = get_manual_labels(queried_samples)

    # Update the queried samples with the new labels
    for sample, label in zip(queried_samples, new_labels):
        sample['living_state'] = label

    # Add newly labeled samples to the training set
    train_docs += queried_samples

    # Prepare the combined features and target
    X_combined, y_combined = prepare_features(train_docs + val_docs + test_docs)

    # Ensure X_unknown_reduced_v2 uses the same feature names as X_combined
    X_unknown_reduced_v2 = X_unknown_reduced_v2.reindex(columns=X_combined.columns, fill_value=0)
    X_test_reduced_v2 = X_test_reduced_v2.reindex(columns=X_combined.columns, fill_value=0)

    # Retrain the model on the combined dataset
    model_combined = RandomForestClassifier(random_state=42)
    model_combined.fit(X_combined, y_combined)

    # Validate the new model
    y_val_pred_combined = model_combined.predict(X_val)
    val_accuracy_combined = accuracy_score(y_val, y_val_pred_combined)
    plot_predictions_vs_actual(y_val, y_val_pred_combined, 'Validation Set Predictions vs Actual')
    print(f'Combined Validation Accuracy after Active Learning: {val_accuracy_combined}')

    # Predict living_state for unknown_docs with the new combined model
    y_unknown_pred_combined = model_combined.predict(X_unknown_reduced_v2)

    # Update unknown_docs with predicted living_state
    for doc, pred in zip(unknown_docs, y_unknown_pred_combined):
        doc['living_state'] = pred

    # Check overfitting by comparing training and validation accuracy
    y_train_pred = model_combined.predict(X_combined)
    train_accuracy_combined = accuracy_score(y_combined, y_train_pred)
    print(f'Combined Training Accuracy after Active Learning: {train_accuracy_combined}')


    # Optionally, evaluate the new combined model on the test set
    y_test_pred_combined = model_combined.predict(X_test_reduced_v2)
    test_accuracy_combined = accuracy_score(y_test, y_test_pred_combined)
    print(f'Combined Test Accuracy after Active Learning: {test_accuracy_combined}')
    plot_predictions_vs_actual(y_test, y_test_pred_combined, 'Test Set Predictions vs Actual')
    # Plot feature importances for the combined model
    plot_feature_importances(model_combined, X_combined)

    # Plot feature correlations for the combined feature set
    plot_feature_correlations(X_combined)

    # Cross-validation on the combined model
    cv_scores_combined = cross_val_score(model_combined, X_combined, y_combined, cv=5)
    print(f'Cross-Validation Scores after Active Learning: {cv_scores_combined}')
    print(f'Cross-Validation Mean Score after Active Learning: {np.mean(cv_scores_combined)}')

    # Plot classifications by antibiotic level with counts
    plot_classifications_by_antibiotic_level_with_counts(unknown_docs + train_docs + val_docs + test_docs)

    # Plot classifications by antibiotic level for the combined model
    plot_classifications_by_antibiotic_level(unknown_docs + train_docs + val_docs + test_docs, model_combined,
                                             X_combined.columns)

