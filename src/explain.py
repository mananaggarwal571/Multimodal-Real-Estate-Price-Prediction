import os
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Windows environment fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. EXACT ARCHITECTURE MATCHING YOUR CHECKPOINT ---
class MultimodalModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultimodalModel, self).__init__()
        # Image Branch (ResNet18)
        self.image_branch = models.resnet18(weights=None)
        self.image_branch.fc = nn.Identity() 

        # Tabular Branch (Must end with 64 units to match 512+64=576)
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU()
        )

        # Final Fusion Layer (576 -> 128 -> 1)
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 64, 128), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.image_branch(image)
        tab_feat = self.tabular_branch(tabular)
        combined = torch.cat((img_feat, tab_feat), dim=1)
        return self.final_fc(combined)

# --- 2. HEATMAP GENERATION FUNCTION ---
def generate_heatmap(image_id):
    # Setup folders
    if not os.path.exists("reports"): os.makedirs("reports")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize & Load Model
    model = MultimodalModel(num_tabular_features=12).to(device)
    model_path = "models/multimodal_house_model.pth"
    
    if os.path.exists(model_path):
        # strict=True rakha hai kyunki ab architecture match kar di hai
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Architecture Matched! Model Loaded Successfully.")
    else:
        print(f"❌ Error: {model_path} nahi mili!")
        return

    # Image Processing
    img_path = f"data/satellite_images/{image_id}.jpg"
    if not os.path.exists(img_path):
        print(f"❌ Image {image_id}.jpg nahi mili!")
        return

    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = np.array(img_pil.resize((224, 224))) / 255.0
    img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Hook for Grad-CAM (Targeting last conv layer)
    features = []
    def hook(module, input, output): features.append(output)
    handle = model.image_branch.layer4[-1].register_forward_hook(hook)

    # Forward Pass
    with torch.no_grad():
        # Using dummy zeros for tabular data during image explanation
        _ = model(img_tensor, torch.zeros((1, 12)).to(device))
    
    # Grad-CAM Logic
    feature_map = features[0].cpu().numpy().squeeze()
    heatmap = np.mean(feature_map, axis=0)
    
    # --- THIK KARNE WALA FIX: Heatmap Sharpening ---
    heatmap = np.maximum(heatmap, 0) # ReLU
    heatmap = np.power(heatmap, 2)   # Squaring clears the noise/blurry red
    heatmap /= (np.max(heatmap) + 1e-8) # Normalize
    
    # Post-process for visualization
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    overlayed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    # Save Result
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Grad-CAM Explanation: House {image_id}")
    save_path = f"reports/gradcam_{image_id}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    handle.remove() # Cleanup
    print(f"🔥 Heatmap saved at: {save_path}")

if __name__ == "__main__":
    # Settings
    IMG_FOLDER = "data/satellite_images"
    MAX_IMAGES = 5  
    
    if os.path.exists(IMG_FOLDER):
        # Saari JPG files ki list lo
        img_list = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if img_list:
            # 20 se 50 images tak limit set karlo
            images_to_process = img_list[:MAX_IMAGES]
            print(f"🚀 Total {len(images_to_process)} images process ho rahi hain...")
            
            for i, filename in enumerate(images_to_process):
                image_id = filename.split('.')[0]
                try:
                    # Har image ke liye function call
                    generate_heatmap(image_id)
                    print(f"[{i+1}/{len(images_to_process)}] Done: {image_id}")
                except Exception as e:
                    print(f"❌ Error in {image_id}: {e}")
            
            print(f"\n Images have been processed.")
        else:
            print("No Images")
    else:
        print("wrong path ")