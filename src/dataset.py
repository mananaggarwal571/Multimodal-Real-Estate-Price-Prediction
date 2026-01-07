import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
from torchvision import transforms

class MultimodalHousingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        self.features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                        'waterfront', 'view', 'condition', 'grade', 'lat', 'long', 'house_age']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prop_id = self.df.iloc[idx]['id']
        img_name = os.path.join(self.img_dir, f"{int(prop_id)}.jpg")
        
        # Image Loading Logic
        if os.path.exists(img_name):
            try:
                image = Image.open(img_name).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), color=(0,0,0))
        else:
            image = Image.new('RGB', (224, 224), color=(0,0,0))
        
        if self.transform:
            image = self.transform(image)
            
        # Feature Extraction
        features_data = self.df.iloc[idx][self.features].values.astype(np.float32)
        tabular_data = torch.from_numpy(features_data)
        
        # Target: Log scale ensure kiya hai tune, nice!
        target = torch.tensor(self.df.iloc[idx]['price_log'], dtype=torch.float32)
        
        return image, tabular_data, target

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])