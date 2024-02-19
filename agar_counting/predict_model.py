import os
import json
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from models.model import ColonyCounter, TransferLearningColonyCounter, TLold
from data.make_dataset import TestDataset
import random
from pytorch_forecasting.metrics import QuantileLoss

# Define transforms for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalize
])

# Set paths to the test image and annotation folders
test_image_folder = '../data/raw/images_test'
test_annotation_folder = '../data/raw/annotations_test'

# Create dataset for inference
test_dataset = TestDataset(test_image_folder, test_annotation_folder, transform=transform)


model = TLold()    
model.load_state_dict(torch.load('../models/ensemble/Restnet-18_ensemble_13.pth'))
model.eval()

# Define a function to calculate the absolute error between true labels and predictions
def calculate_absolute_error(true_label, prediction):
    return abs(true_label - prediction)

# Initialize lists to store absolute errors and predicted counts
absolute_errors = []
predicted_counts = []
true_labels = []

# Define the number of samples to plot
n_samples = 20

# Choose n_samples randomly
random_indices = random.sample(range(len(test_dataset)), n_samples)

# Run inference on the randomly chosen samples and plot the results
for i in random_indices:
    image, annotation_path = test_dataset[i]
    
    output = model(image.unsqueeze(0))
    predicted_count = output.item()
    
    # Load corresponding annotation file to get true label
    with open(annotation_path) as f:
        annotation = json.load(f)
        true_label = annotation['colonies_number']
    
    absolute_error = calculate_absolute_error(true_label, predicted_count)
    absolute_errors.append(absolute_error)
    predicted_counts.append(predicted_count)
    true_labels.append(true_label)
    
    # Plot image with true and predicted numbers in the title
    plt.figure(figsize=(4, 4))
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(f"True: {true_label}, Predicted: {predicted_count}")
    plt.axis('off')
    plt.show()


