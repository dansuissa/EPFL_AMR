import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from data_processing import *
from feature_extraction import *

# Dataset Class
class BacteriaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Keep as single-label for the entire sequence
        return sample, label

def preprocess_data_U(docs, max_length=1000):
    normalized_docs = normalize_data_model2(docs)
    raw_signals = []
    bacteria_labels = []

    for doc in normalized_docs:
        transmission = np.array(doc['transmission_normalized'])
        
        if len(transmission) < max_length:
            padded_transmission = np.pad(transmission, (0, max_length - len(transmission)), 'constant')
        else:
            padded_transmission = transmission[:max_length]
        
        raw_signals.append(padded_transmission)
        bacteria_labels.append(doc['label'])  # Assuming this is the class label

    raw_signals = np.array(raw_signals)
    bacteria_labels = np.array(bacteria_labels)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(bacteria_labels)

    # Convert to tensor
    sequence_labels = torch.tensor(encoded_labels, dtype=torch.long)  # Shape will be [num_samples]

    return raw_signals, sequence_labels, label_encoder


class UNetModel(nn.Module):
    def __init__(self, num_classes):
        super(UNetModel, self).__init__()
        self.encoder1 = self.conv_block(1, 16)  # Input channels: 1, Output channels: 16
        self.encoder2 = self.conv_block(16, 32)  # Output channels: 32
        self.encoder3 = self.conv_block(32, 64)  # Output channels: 64
        self.encoder4 = self.conv_block(64, 128)  # Output channels: 128

        self.middle = self.conv_block(128, 256)  # Bottleneck layer

        # Decoder structure for upsampling and concatenation
        self.decoder3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)  # 256 input channels
        self.decoder2 = nn.ConvTranspose1d(128 + 64, 64, kernel_size=2, stride=2)  # Expecting 128 + 64 input channels
        self.decoder1 = nn.ConvTranspose1d(64 + 32, 32, kernel_size=2, stride=2)  # Expecting 64 + 32 input channels
        
        # Final classification layer
        self.decoder0 = nn.Conv1d(32, num_classes, kernel_size=1)  # Output num_classes directly

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        # Forward pass through the network
        x1 = self.encoder1(x)  # x1 shape: [batch_size, 16, new_length]
        x2 = self.encoder2(x1)  # x2 shape: [batch_size, 32, new_length]
        x3 = self.encoder3(x2)  # x3 shape: [batch_size, 64, new_length]
        x4 = self.encoder4(x3)  # x4 shape: [batch_size, 128, new_length]
        x5 = self.middle(x4)  # x5 shape: [batch_size, 256, new_length]

        # Start decoding
        x6 = self.decoder3(x5)  # x6 shape: [batch_size, 128, new_length*2]
        x6 = torch.cat((x6, x4), dim=1)  # Concatenate encoder output x4 with x6

        x7 = self.decoder2(x6)  # x7 shape: [batch_size, 64, new_length*4]
        x7 = torch.cat((x7, x3), dim=1)  # Concatenate encoder output x3 with x7

        x8 = self.decoder1(x7)  # x8 shape: [batch_size, 32, new_length*8]
        x8 = torch.cat((x8, x2), dim=1)  # Concatenate encoder output x2 with x8

        x9 = self.decoder0(x8)  # x9 shape: [batch_size, num_classes, new_length*16]
        
        # Global average pooling to reduce to [batch_size, num_classes]
        output = nn.functional.adaptive_avg_pool1d(x9, 1).view(x9.size(0), -1)
        
        return output



# Training Loop
def train_model_U(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 1)  # Change dimensions for loss calculation
            loss = criterion(outputs, labels)  
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_accuracy = evaluate_model_U(model, val_loader)
        print(f"Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Evaluation Function
def evaluate_model_U(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 1)  # Change dimensions for loss calculation
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return average_loss, accuracy

# Main Function
def main_U(docs):
    features, labels, label_encoder = preprocess_data_U(docs, max_length=1000)

    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, stratify=labels)

    # Create DataLoaders
    train_dataset = BacteriaDataset(X_train, y_train)
    val_dataset = BacteriaDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Parameters
    num_classes = len(label_encoder.classes_)

    # Create and train model
    unet_model = UNetModel(num_classes)
    train_model_U(unet_model, train_loader, val_loader)

    # Evaluate model
    test_loss, test_accuracy = evaluate_model_U(unet_model, val_loader)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Assuming `docs` is defined elsewhere
# main(docs)
