import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from neuralforecast.losses.pytorch import HuberLoss
from collections import Counter
import seaborn as sns

def normalize_data_model2_nix(docs):
    """Normalize data for model 2 using the provided normalization factor."""
    normalized_docs = []
    for doc in docs:
        transmission = np.array(doc['data']['transmission'])
        norm_factor = doc['normalization_factor']
        transmission_normalized = transmission / norm_factor
        normalized_docs.append({
            'transmission_normalized': transmission_normalized,
            'bacteria': doc['bacteria']
        })
    return normalized_docs

def preprocess_time_series_tft_nix(docs):
    """Preprocess the time series data for the TFT model, ensuring continuity for each sample."""
    normalized_docs = normalize_data_model2_nix(docs)
    records = []
    
    for doc in normalized_docs:
        transmission_normalized = doc['transmission_normalized']
        num_seconds = len(transmission_normalized)  # Get the number of seconds for this transmission
        
        # Create a starting datetime
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start at midnight on January 1, 2023
        
        # Create time indices based on the length of the transmission normalized data
        for idx in range(num_seconds):
            records.append({
                'unique_id': doc['bacteria'],  # Each doc is associated with a unique bacteria type
                'ds': start_time + datetime.timedelta(seconds=idx),  # Increment seconds
                'y': transmission_normalized[idx]  # Use this value for regression
            })
    
    return pd.DataFrame(records)


def stratified_split_data(data_df, test_size=0.1, val_size=0.2):
    """Ensure each set has all bacteria types."""
    bacteria_types = data_df['unique_id'].unique()
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Shuffle bacteria types
    np.random.shuffle(bacteria_types)

    for btype in bacteria_types:
        btype_data = data_df[data_df['unique_id'] == btype]
        
        # Split into train and temp (for val and test)
        train_split, temp_split = train_test_split(btype_data, test_size=test_size + val_size, random_state=42, shuffle=False)
        # Split temp into val and test
        val_split, test_split = train_test_split(temp_split, test_size=test_size / (test_size + val_size), random_state=42, shuffle=False)
        
        train_df = pd.concat([train_df, train_split])
        val_df = pd.concat([val_df, val_split])
        test_df = pd.concat([test_df, test_split])

    # Check counts of unique bacteria types in each set
    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")
    print(f"Number of test samples: {len(test_df)}")

    print("\nBacteria counts in Training Set:")
    print(train_df['unique_id'].value_counts())
    
    print("\nBacteria counts in Validation Set:")
    print(val_df['unique_id'].value_counts())
    
    print("\nBacteria counts in Test Set:")
    print(test_df['unique_id'].value_counts())

    # Plot one instance of bacteria for continuity check
    example_bacteria = train_df['unique_id'].unique()[0]
    example_data = train_df[train_df['unique_id'] == example_bacteria]
    
    plt.figure(figsize=(12, 6))
    plt.plot(example_data['ds'], example_data['y'], label=f'Bacteria: {example_bacteria}')
    plt.title(f'Time Series for {example_bacteria}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Transmission')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'time_series_example_{example_bacteria}.png')  # Save plot to file
    plt.close()

    return train_df, val_df, test_df

def check_class_imbalance(y):
    """Check for class imbalance in the target variable."""
    counter = Counter(y)
    print("Class distribution:", counter)
    return counter

def plot_prediction_example(val_df, predictions, unique_id):
    """Plot the actual vs predicted time series for a specific unique_id and save the plot."""
    
    # Print the structure of val_df and predictions to understand their contents
    print("Validation DataFrame (val_df) Columns:", val_df.columns)
    print("Predictions DataFrame Columns:", predictions.columns)

    # Print the first few rows of both DataFrames
    print("First few rows of val_df:")
    print(val_df.head())
    print("First few rows of predictions:")
    print(predictions.head())

    # Print data types of each column in both DataFrames
    print("Data types in val_df:")
    print(val_df.dtypes)
    print("Data types in predictions:")
    print(predictions.dtypes)

    # Attempt to filter the data for the specified unique_id
    actual = val_df[val_df['unique_id'] == unique_id]
    print(f"Actual Data for unique_id '{unique_id}':")
    print(actual)

    # Check if there are predictions for the specified unique_id
    # Since predictions do not have 'unique_id', we will need to find the relevant rows differently
    pred = predictions  # Currently, we can't filter by unique_id since it doesn't exist
    print(f"Predictions for unique_id '{unique_id}':")
    print(pred)

    # Check if the DataFrames are empty
    if actual.empty:
        print(f"No actual data available for unique_id: {unique_id}.")
    if pred.empty:
        print(f"No predictions available for unique_id: {unique_id}.")

    # Proceed to plot if we have actual data and predictions
    if not actual.empty and not pred.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(actual['ds'], actual['y'], label='Actual', color='blue')
        plt.plot(pred['ds'], pred['TFT'], label='Predicted', color='orange', linestyle='--')
        plt.title(f'Actual vs Predicted for {unique_id}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Transmission')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'prediction_example_{unique_id}.png')  # Save plot to file
        plt.close()
    else:
        print("No data available to plot for unique_id:", unique_id)



def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot confusion matrix and save the figure."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Save plot to file
    plt.close()

def plot_feature_importance(classifier, feature_names):
    """Plot feature importance and save the figure."""
    feature_importances = classifier.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')  # Save plot to file
    plt.close()


def main_nix(docs):
    # Step 1: Preprocess the input data
    data_df = preprocess_time_series_tft_nix(docs)

    # Step 2: Stratified split the data into training, validation, and test sets
    train_df, val_df, test_df = stratified_split_data(data_df, test_size=0.1, val_size=0.2)

    # Step 3: Initialize the TFT model
    h = 12  # Forecast horizon
    input_size = 1  # We are forecasting one variable, 'y'
    hidden_size = 20  # Number of hidden units
    max_steps = 100  # Set max_steps for training

    # Use HuberLoss for the model
    tft_model = NeuralForecast(models=[TFT(h=h, input_size=input_size, hidden_size=hidden_size, loss=HuberLoss(), max_steps=max_steps)], freq='S')



    # Step 4: Fit the model
    try:
        tft_model.fit(train_df)
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return

    # Step 5: Make predictions on the validation set
    try:
        predictions = tft_model.predict(val_df)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    print('NN finished, let’s move on to classification')

    # Plot an example of the predictions
    unique_id_example = val_df['unique_id'].unique()[0]  # Choose the first unique ID for plotting
    plot_prediction_example(val_df, predictions, unique_id_example)

    # Step 6: Prepare data for multi-class classification
    prediction_map = predictions.groupby('unique_id')['TFT'].apply(list).reset_index(name='predicted_values')

    # Join the predictions with the validation DataFrame to extract features
    feature_df = val_df.merge(prediction_map, on='unique_id', how='left')

    # Calculate statistical features
    feature_df['mean_y'] = feature_df.groupby('unique_id')['y'].transform('mean')
    feature_df['std_y'] = feature_df.groupby('unique_id')['y'].transform('std')
    feature_df['min_y'] = feature_df.groupby('unique_id')['y'].transform('min')
    feature_df['max_y'] = feature_df.groupby('unique_id')['y'].transform('max')

    # Calculate features from predicted values
    feature_df['mean_pred'] = feature_df['predicted_values'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    feature_df['std_pred'] = feature_df['predicted_values'].apply(lambda x: np.std(x) if len(x) > 0 else 0)
    feature_df['min_pred'] = feature_df['predicted_values'].apply(lambda x: np.min(x) if len(x) > 0 else 0)
    feature_df['max_pred'] = feature_df['predicted_values'].apply(lambda x: np.max(x) if len(x) > 0 else 0)

    # Encode the labels for classification
    le = LabelEncoder()
    feature_df['bacteria_class'] = le.fit_transform(feature_df['unique_id'])

    # Prepare the final features for the classifier
    X = feature_df[['mean_y', 'std_y', 'min_y', 'max_y', 'mean_pred', 'std_pred', 'min_pred', 'max_pred']]
    y_class = feature_df['bacteria_class']

    # Step 7: Split into training and validation sets for classification
    X_train, X_val, y_train, y_val = train_test_split(X, y_class, test_size=0.2, random_state=42, shuffle=True)

    print('Data processing finished, let’s move on to the RF')
    classifier = RandomForestClassifier(random_state=42)
    
    # Step 8: Cross-validation for classifier
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)  # 5-fold cross-validation
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores)}")

    classifier.fit(X_train, y_train)

    # Step 9: Make predictions for the classification task
    y_class_pred = classifier.predict(X_val)

    # Step 10: Evaluate classification performance
    print(f"Classification Accuracy: {accuracy_score(y_val, y_class_pred)}")
    print(classification_report(y_val, y_class_pred, target_names=le.classes_))

    # Step 11: Plot confusion matrix for test set
    plot_confusion_matrix(y_val, y_class_pred, target_names=le.classes_)

    # Step 12: Feature importance
    plot_feature_importance(classifier, X.columns)

    # Step 13: Prepare features for the test set
    test_prediction_map = predictions.groupby('unique_id')['TFT'].apply(list).reset_index(name='predicted_values')
    test_feature_df = test_df.merge(test_prediction_map, on='unique_id', how='left')

    # Calculate statistical features for the test set
    test_feature_df['mean_y'] = test_feature_df.groupby('unique_id')['y'].transform('mean')
    test_feature_df['std_y'] = test_feature_df.groupby('unique_id')['y'].transform('std')
    test_feature_df['min_y'] = test_feature_df.groupby('unique_id')['y'].transform('min')
    test_feature_df['max_y'] = test_feature_df.groupby('unique_id')['y'].transform('max')

    # Calculate features from predicted values for the test set
    test_feature_df['mean_pred'] = test_feature_df['predicted_values'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    test_feature_df['std_pred'] = test_feature_df['predicted_values'].apply(lambda x: np.std(x) if len(x) > 0 else 0)
    test_feature_df['min_pred'] = test_feature_df['predicted_values'].apply(lambda x: np.min(x) if len(x) > 0 else 0)
    test_feature_df['max_pred'] = test_feature_df['predicted_values'].apply(lambda x: np.max(x) if len(x) > 0 else 0)

    # Check if 'bacteria_class' exists and create it for the test set
    if 'bacteria_class' not in test_feature_df.columns:
        test_feature_df['bacteria_class'] = le.transform(test_feature_df['unique_id'])

    # Prepare features for testing
    X_test = test_feature_df[['mean_y', 'std_y', 'min_y', 'max_y', 'mean_pred', 'std_pred', 'min_pred', 'max_pred']]
    y_test_class = test_feature_df['bacteria_class']

    # Step 13: Make predictions for the test set
    y_test_class_pred = classifier.predict(X_test)

    # Step 14: Evaluate test performance
    print(f"Test Classification Accuracy: {accuracy_score(y_test_class, y_test_class_pred)}")
    print(classification_report(y_test_class, y_test_class_pred, target_names=le.classes_))

# Usage Example
# Assuming `docs` is your input data structured correctly
# main_nix(docs)
