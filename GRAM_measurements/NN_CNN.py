import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW, RMSprop, Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.nn.utils import clip_grad_norm_
from data_processing import *
from feature_extraction import *
from sklearn.metrics import classification_report
import optuna

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BacteriaDataset class
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

# DataLoader
def create_dataloader(features, labels, batch_size=32, shuffle=True):
    dataset = BacteriaDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Enhanced Deeper CNN Model
class EnhancedDeeperCNN(nn.Module):
    def __init__(self, input_length, num_classes, num_layers=4, num_neurons=512):
        super(EnhancedDeeperCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * (input_length // 16), num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * (x.shape[2]))  # Flatten for fully connected layer
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training Loop
def train_model(model, train_loader, val_loader, epochs=64, learning_rate=0.001, weight_decay=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(device)  # Add channel dimension for Conv1D and move to device
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        val_loss, val_accuracy, _ = evaluate_model(model, val_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.unsqueeze(1).to(device)  # Add channel dimension for Conv1D and move to device
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    classification_rep = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(model.fc2.out_features)])
    return average_loss, accuracy, classification_rep

# Preprocess Time Series Function with Enhanced Augmentation
def preprocess_time_series(docs, num_chunks=6, max_length=1000):
    chunked_docs = chunk_time_series(docs, num_chunks)
    augmented_docs = augment_data(chunked_docs, noise_level=0.05, shift_max=10)

    for doc in augmented_docs:
        transmission = np.array(doc['transmission_normalized'])
        if len(transmission) < max_length:
            padded_transmission = np.pad(transmission, (0, max_length - len(transmission)), 'constant')
        else:
            padded_transmission = transmission[:max_length]
        doc['transmission_normalized'] = padded_transmission

    return augmented_docs

# Preprocess Data Function
def preprocess_data(docs):
    # Normalize data
    normalized_docs = normalize_data_model2(docs)

    # Augment data
    augmented_docs = preprocess_time_series(normalized_docs)

    # Extract raw time series
    raw_signals = [doc['transmission_normalized'] for doc in augmented_docs]
    bacteria_labels = [doc['bacteria'] for doc in augmented_docs]

    # Encode bacteria labels
    label_encoder_bacteria = LabelEncoder()
    encoded_bacteria_labels = label_encoder_bacteria.fit_transform(bacteria_labels)

    return np.array(raw_signals), np.array(encoded_bacteria_labels), label_encoder_bacteria

