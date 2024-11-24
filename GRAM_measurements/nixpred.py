import pandas as pd
from neuralforecast import NeuralForecast, models
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

# Function to prepare the data for the model
def prepare_data(docs):
    # Extract the time series data
    data = docs[0]['data']

    # Debug: Print type and keys of 'data'
    print("Type of 'data':", type(data))
    print("Keys of 'data':", data.keys() if isinstance(data, dict) else "Not a dictionary")

    # Check for expected keys and extract the time series values
    if isinstance(data, dict) and 'time' in data and 'transmission' in data:
        time_values = data['time']
        transmission_values = data['transmission']
        if isinstance(time_values, list) or isinstance(time_values, np.ndarray):
            if isinstance(transmission_values, list) or isinstance(transmission_values, np.ndarray):
                # Convert time_values to integers
                time_values = (np.array(time_values) * 1e6).astype(int)  # Convert to microseconds for finer granularity

                # Creating a DataFrame for the time series data
                df = pd.DataFrame({
                    'unique_id': 'series_1',  # Add a constant unique_id
                    'ds': time_values,  # Use the converted 'time' values
                    'y': transmission_values  # Use the 'transmission' values
                })
            else:
                raise ValueError("The 'transmission' field is not a list or numpy array")
        else:
            raise ValueError("The 'time' field is not a list or numpy array")
    else:
        raise ValueError("The 'data' field does not contain the expected keys or is not structured as expected")

    return df

# Function to calculate Mean Absolute Error
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Function to plot actual vs predicted values
def plot_results(train_df, test_df, forecast):
    plt.figure(figsize=(14, 7))
    plt.plot(train_df['ds'], train_df['y'], label='Train')
    plt.plot(test_df['ds'], test_df['y'], label='Actual')
    plt.plot(test_df['ds'], forecast, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Transmission')
    plt.legend()
    plt.savefig('plot_results.png')  # Save plot to file
    plt.close()

if __name__ == '__main__':
    print("Loading data from pickle file...")
    with open('data.pkl', 'rb') as f:
        docs = pickle.load(f)

    # Print the keys to understand the data structure
    print(docs[0].keys())  # Should output the keys of the dictionary

    # Prepare the data
    df = prepare_data(docs)
    print(df.head())  # Print the first few rows to verify the data

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.75)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Force torch to use CPU
    torch.cuda.is_available = lambda : False

    # Initialize the TFT model with a smaller hidden size
    tft_model = models.TFT(input_size=1, hidden_size=8, h=12, max_steps=50, batch_size=16)  # Reduced hidden size

    # Correctly initialize the NeuralForecast object
    model = NeuralForecast(models=[tft_model], freq=1)  # Set freq parameter to 1 for integer step size

    # Fit the model on the training data using CPU
    trainer = pl.Trainer(max_epochs=50, gpus=0, gradient_clip_val=0.5)  # Set gpus=0 to use CPU
    trainer.fit(model, train_df)

    # Make predictions on the test data
    forecast_df = model.predict(test_df)
    print(forecast_df)

    # Calculate and print the Mean Absolute Error
    y_true = test_df['y'].values
    y_pred = forecast_df['TFT'].values  # Adjusted to use the correct column name
    mae = calculate_mae(y_true, y_pred)
    print(f'Mean Absolute Error: {mae}')

    # Plot the results
    plot_results(train_df, test_df, y_pred)
