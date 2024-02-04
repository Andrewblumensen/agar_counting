import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from models.model import ColonyCounter, TransferLearningColonyCounter
from tqdm import tqdm
import numpy as np
from data.make_dataset import BacteriaDataset
import statistics
import wandb
import datetime

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Initialize wandb
wandb.login(key="ca8b8153dd94925f61e5df4e4fc0bf7b8b234ecc")
wandb.init(project='MT_agar1', name="Restnet-18 big DS Retrain")

# Read config from JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

num_epochs = config['num_epochs']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
weight_decay = config['weightdecay']
model_name = config['model']
loss_f = config['loss']
val_split = config['val_split']
dataset_size = config['dataset']

wandb.config.update(config, allow_val_change=True)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalize
])

# Set paths to your image and annotation folders

if dataset_size == "big":
    image_folder = '../data/raw/images_big'
    annotation_folder = '../data/raw/annotations_big'
elif dataset_size == "small":
    image_folder = '../data/raw/images'
    annotation_folder = '../data/raw/annotations'
    
# Create dataset
dataset = BacteriaDataset(image_folder, annotation_folder, transform=transform)

# Split dataset into train and validation sets
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(val_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]

# Define samplers for train and validation sets
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create data loaders for train and validation sets
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Initialize model, loss function, and optimizer
if model_name == "resnet18":
    model = TransferLearningColonyCounter().to(device)
elif model_name == "cnn":
    model = ColonyCounter().to(device)

if loss_f == "l1":
    criterion = nn.L1Loss()
elif loss_f == "l2":
    criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a function to calculate the average distance between true labels and predictions
def calculate_avg_distance(labels, predictions):
    distances = torch.abs(labels - predictions.view(-1))
    avg_distance = torch.mean(distances)
    return avg_distance.item()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    with tqdm(total=len(train_loader)) as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.update(1)
        
        avg_loss = total_loss / len(train_loader)
        print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')
        
        # Log metrics to wandb
        wandb.log({'Train_loss': avg_loss})

    # Validation loop
    model.eval()
    all_distances = []
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(images)
            val_loss = criterion(outputs, labels.float().unsqueeze(1))
            total_val_loss += val_loss.item()
            distances = [calculate_avg_distance(labels, outputs) for labels, outputs in zip(labels, outputs)]
            all_distances.extend(distances)
        avg_val_loss = total_val_loss / len(val_loader)
        median_distance = statistics.median(all_distances)
        print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss}, Median Distance: {median_distance}')

        # Log metrics to wandb
        wandb.log({'Val_loss': avg_val_loss, 'median_distance': median_distance})

# Save the trained model with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f'{model_name}_model_{timestamp}.pth'
model_path = os.path.join('../models', model_name)
torch.save(model.state_dict(), model_path)

# Finish wandb run
wandb.finish()
