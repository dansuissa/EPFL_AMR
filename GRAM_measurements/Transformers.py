import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from data_processing import *  # Ensure this module contains necessary functions
from feature_extraction import *  # Ensure this module contains necessary functions

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Positional Encoding for Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:, :seq_len, :].to(x.device)

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_size, num_classes, num_heads=8, num_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, 128)  # Increased embedding size
        self.positional_encoding = PositionalEncoding(d_model=128, max_len=1000)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, sequence_length, 128)
        x = x.unsqueeze(1)  # Add sequence length dimension
        
        pos_enc = self.positional_encoding(x)  # (1, seq_len, 128)
        x = x + pos_enc  # Add positional encoding
        
        x = self.transformer_layers(x.permute(1, 0, 2))  # (sequence_length, batch_size, 128)
        x = x.mean(dim=0)  # Mean pooling
        x = self.fc(x)  # (batch_size, num_classes)
        return x

# Dataset Class
class BacteriaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label

# DataLoader Creation
def create_dataloader(features, labels, batch_size=32, shuffle=True):
    dataset = BacteriaDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Improved Normalization Function
def normalize_data_for_transformer(docs):
    """Normalize data for the transformer model."""
    normalized_docs = []
    for doc in docs:
        transmission = np.array(doc['data']['transmission'])
        norm_factor = doc['normalization_factor']
        transmission_normalized = transmission / norm_factor
        
        # Clipping values to avoid extremes
        transmission_normalized = np.clip(transmission_normalized, 0, 1)
        
        normalized_docs.append({
            'transmission_normalized': transmission_normalized,
            'label': doc['gram_type'],
            'shape': doc['shape_type'],
            'name': doc['name'],
            'bacteria': doc['bacteria']
        })
    
    return normalized_docs

# Preprocessing Data for Transformer
def preprocess_data_transformer(docs):
    normalized_docs = normalize_data_for_transformer(docs)
    augmented_docs = preprocess_time_series_lmgru(normalized_docs)  # Use LMGRU preprocessing

    raw_signals = [doc['transmission_normalized'] for doc in augmented_docs]
    bacteria_labels = [doc['bacteria'] for doc in augmented_docs]

    label_encoder_bacteria = LabelEncoder()
    encoded_bacteria_labels = label_encoder_bacteria.fit_transform(bacteria_labels)

    return np.array(raw_signals), np.array(encoded_bacteria_labels), label_encoder_bacteria

# Training Function
def train_transformer_model(model, train_loader, val_loader, epochs=150, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        val_loss, val_accuracy = evaluate_model_transformer(model, val_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Evaluation Function
def evaluate_model_transformer(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return average_loss, accuracy

# Preprocessing Time Series using LMGRU
def preprocess_time_series_lmgru(docs, num_chunks=6, max_length=1000):
    chunked_docs = chunk_time_series(docs)  # Ensure this function exists
    augmented_docs = augment_data(chunked_docs, noise_level=0.05, shift_max=10)  # Ensure this function exists

    for doc in augmented_docs:
        transmission = np.array(doc['transmission_normalized'])
        if len(transmission) < max_length:
            padded_transmission = np.pad(transmission, (0, max_length - len(transmission)), 'constant')
        else:
            padded_transmission = transmission[:max_length]
        doc['transmission_normalized'] = padded_transmission

    return augmented_docs