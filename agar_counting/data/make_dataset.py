import os
import json
from PIL import Image
import torch



# Define dataset class for inference
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_names = os.listdir(image_folder)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_folder, img_name)
        annotation_path = os.path.join(self.annotation_folder, img_name.split('.')[0] + '.json')
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, annotation_path
    
    
# Define dataset class
class BacteriaDataset():
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_names = os.listdir(image_folder)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_folder, img_name)
        annotation_path = os.path.join(self.annotation_folder, img_name.split('.')[0] + '.json')
        
        image = Image.open(img_path).convert("RGB")
        
        with open(annotation_path) as f:
            annotation = json.load(f)
            colonies_number = annotation['colonies_number']
        
        if self.transform:
            image = self.transform(image)
        
        return image, colonies_number