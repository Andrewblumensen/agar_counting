import os
import json
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from models.model import ColonyCounter, TransferLearningColonyCounter
from data.make_dataset import TestDataset
import random

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



# Set paths to the test image and annotation folders
cal_image_folder = '../data/raw/images_cal'
cal_annotation_folder = '../data/raw/annotations_cal'

# Create dataset for inference
cal_dataset = TestDataset(cal_image_folder, cal_annotation_folder, transform=transform)




model = model = TransferLearningColonyCounter(class_num = 2)   
model.load_state_dict(torch.load('../models/special/Restnet-18 small pinball7_correct2.pth'))
model.eval()

# Define a function to calculate the absolute error between true labels and predictions
def calculate_absolute_error(true_label, prediction):
    return abs(true_label - prediction)

# Initialize lists to store absolute errors and predicted counts
absolute_errors = []
predicted_counts = []


# Define the number of samples to plot
n_cal = 2000




"""
correct = 0
cal_upper = []
cal_lower = []
cal_labels = []

# Choose n_samples randomly
random_indices = random.sample(range(len(cal_dataset)), n_cal)

# Run inference on the randomly chosen samples and plot the results
for i in random_indices:
    image, annotation_path = cal_dataset[i]
    
    outputs = model(image.unsqueeze(0))
    print(outputs)
    predicted_counts.append(outputs.tolist())  # Append the list of quantile guesses
    
    # Load corresponding annotation file to get true label
    with open(annotation_path) as f:
        annotation = json.load(f)
        true_label = annotation['colonies_number']
    if true_label == -1:
        continue  # Skip examples with true label -1
    cal_labels.append(true_label)
    cal_lower.append(outputs[:,0].item())
    cal_upper.append(outputs[:,1].item())
    
    if true_label >= outputs[:,0] and true_label <= outputs[:,1]:
        correct = correct+1
    # Plot image with true and predicted numbers in the title
    
    plt.figure(figsize=(4, 4))
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(f"True: {true_label}, Predicted: {outputs}")
    plt.axis('off')
    plt.show()
    print(correct/n_cal)
    

alpha = 0.1 # 1-alpha is the desired coverage

# Get scores. cal_upper.shape[0] == cal_lower.shape[0] == n
cal_scores = np.maximum(np.array(cal_labels)-np.array(cal_upper), np.array(cal_lower)-np.array(cal_labels))
# Get the score quantile
qhat = np.quantile(cal_scores, np.ceil((n_cal+1)*(1-alpha))/n_cal, interpolation='higher')
"""

#%%
import pickle
"""
# Pickle qhat
with open('../data/processed/qhat.pkl', 'wb') as f:
    pickle.dump(qhat, f)
"""

# Load qhat from the pickle file
with open('../data/processed/qhat.pkl', 'rb') as f:
    qhat = pickle.load(f)

#%%

val_upper = []
val_lower = []
val_labels = []

correctval1 = 0
correctval2 = 0
n_val = 149

# Choose n_samples randomly
random_indices = random.sample(range(len(test_dataset)), n_val)

# Run inference on the randomly chosen samples and plot the results
for i in random_indices:
    image, annotation_path = test_dataset[i]
    
    outputs = model(image.unsqueeze(0))
    
    up = outputs[:,1].item() + qhat
    low = outputs[:,0].item() - qhat
    
    print(i)
    
    # Load corresponding annotation file to get true label
    with open(annotation_path) as f:
        annotation = json.load(f)
        true_label = annotation['colonies_number']
    if true_label == -1:
        continue  # Skip examples with true label -1  
    
    if true_label >= outputs[:,0].item() and true_label <= outputs[:,1].item():
        correctval1 = correctval1+1
        
    if true_label >= low and true_label <= up:
        correctval2 = correctval2+1
    # Plot image with true and predicted numbers in the title
    
    plt.figure(figsize=(4, 4))
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(f"True: {true_label}, OI: {outputs}, NI: {low}  {up}")
    plt.axis('off')
    plt.show()
    

print(correctval1/n_val)
print(correctval2/n_val)

























