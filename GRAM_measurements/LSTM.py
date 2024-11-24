#implementation of LMGRU

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import AdamW, RMSprop, Adam
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils import clip_grad_norm_
from data_processing import *
from feature_extraction import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LMGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LMGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Initialize Linear layers with the correct input sizes
        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        for t in range(x.size(1)):
            x_t = x[:, t, :]

            # Concatenate x_t and h correctly
            combined = torch.cat((x_t, h), dim=1)
            


            z_t = torch.sigmoid(self.Wz(combined))
            r_t = torch.sigmoid(self.Wr(combined))
            combined_reset = torch.cat((x_t, r_t * c), dim=1)
            c_tilde = torch.tanh(self.Wc(combined_reset))
            c = z_t * c + (1 - z_t) * c_tilde
            o_t = torch.sigmoid(self.Wo(combined))
            h = o_t * torch.tanh(c)

        out = self.fc(self.dropout(h))
        return out


class BacteriaDatasetLMGRU(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label

def create_dataloader_lmgru(features, labels, batch_size=32, shuffle=True):
    if features.ndim == 2:
        features = features[:, :, np.newaxis]  # Add a new axis to match the required input dimensions
    dataset = BacteriaDatasetLMGRU(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader




def train_lmgru_model(model, train_loader, val_loader, epochs=150, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Ensure inputs are 3-dimensional
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(-1)
            
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        val_loss, val_accuracy = evaluate_model_lmgru(model, val_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

def evaluate_model_lmgru(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Ensure inputs are 3-dimensional
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(-1)
            
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    average_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return average_loss, accuracy


def preprocess_time_series_lmgru(docs, num_chunks=6, max_length=1000):
    chunked_docs = chunk_time_series(docs)
    augmented_docs = augment_data(chunked_docs, noise_level=0.05, shift_max=10)

    for doc in augmented_docs:
        transmission = np.array(doc['transmission_normalized'])
        if len(transmission) < max_length:
            padded_transmission = np.pad(transmission, (0, max_length - len(transmission)), 'constant')
        else:
            padded_transmission = transmission[:max_length]
        doc['transmission_normalized'] = padded_transmission

    return augmented_docs

def preprocess_data_lmgru(docs):
    normalized_docs = normalize_data_model2(docs)
    augmented_docs = preprocess_time_series_lmgru(normalized_docs)
    raw_signals = [doc['transmission_normalized'] for doc in augmented_docs]
    bacteria_labels = [doc['bacteria'] for doc in augmented_docs]

    label_encoder_bacteria = LabelEncoder()
    encoded_bacteria_labels = label_encoder_bacteria.fit_transform(bacteria_labels)

    return np.array(raw_signals), np.array(encoded_bacteria_labels), label_encoder_bacteria
