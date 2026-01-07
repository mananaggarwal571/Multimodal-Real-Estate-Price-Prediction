import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import MultimodalHousingDataset, data_transforms
from src.model import MultimodalModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    
    full_dataset = MultimodalHousingDataset("data/processed/train_ready.csv", "data/satellite_images", transform=data_transforms)
    
 
    subset_size =  len(full_dataset)
    indices = np.random.choice(len(full_dataset), subset_size, replace=False)
    dataset = Subset(full_dataset, indices)
    print(f"📉 Using Subset: {subset_size} samples for speed.")

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0) 

    model = MultimodalModel(num_tabular_features=12).to(device)
    

    for name, param in model.image_branch.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False

    
    optimizer = optim.AdamW([
        {'params': model.image_branch.layer4.parameters(), 'lr': 1e-5},
        {'params': model.tabular_branch.parameters(), 'lr': 1e-3},
        {'params': model.final_fc.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-2)

    
    num_epochs = 15
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=num_epochs)
    
    criterion = nn.HuberLoss() 

    print("🔥 Starting Fast Training...")
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        for images, tabular, targets in loop:
            images, tabular, targets = images.to(device), tabular.to(device), targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images, tabular).squeeze()
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/multimodal_house_model.pth")
    print("✅ Model Trained & Saved as models/multimodal_house_model.pth")

if __name__ == "__main__":
    train()