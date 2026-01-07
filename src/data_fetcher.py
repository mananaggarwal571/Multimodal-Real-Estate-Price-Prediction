import os
import time
import pandas as pd
import requests
from onedrivedownloader import download
from tqdm import tqdm


TRAIN_LINK = "https://onedrive.live.com/:x:/g/personal/8CF6803ADF7941C3/IQBue1q4w4TETL_7xWMGhcD_AejALtdsXTBejVUjRA9qeM8?resid=8CF6803ADF7941C3!sb85a7b6e84c34cc4bffbc5630685c0ff&ithint=file%2Cxlsx&e=kWdglC&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy84Y2Y2ODAzYWRmNzk0MWMzL0lRQnVlMXE0dzRURVRMXzd4V01HaGNEX0FlakFMdGRzWFRCZWpWVWpSQTlxZU04P2U9a1dkZ2xD"
DATA_RAW_DIR = "data/raw"
IMAGE_DIR = "data/satellite_images"
INPUT_FILE = os.path.join(DATA_RAW_DIR, "train.xlsx")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def setup_data_files():
    print("--- Step 1: Downloading Excel ---")
    if not os.path.exists(INPUT_FILE):
        try:
            download(TRAIN_LINK, filename=INPUT_FILE, force_download=True)
            print("Excel Downloaded!")
        except Exception as e:
            print(f"OneDrive Error: {e}")
    else:
        print("Excel already exists.")

def download_satellite_images():
    print("\n--- Step 2: Downloading Satellite Images (Public Esri Server) ---")
    df = pd.read_excel(INPUT_FILE)
    
    total_rows=len(df)
    for _, row in tqdm(df.iterrows(), total=total_rows):
        property_id = int(row['id'])
        save_path = os.path.join(IMAGE_DIR, f"{property_id}.jpg")

        if os.path.exists(save_path): continue

        lat, lon = row['lat'], row['long']

        
        offset = 0.002 
        bbox = f"{lon-offset},{lat-offset},{lon+offset},{lat+offset}"
        url = f"https://render.githubusercontent.com/render/dynamic/map/staticmap?center={lat},{lon}&zoom=17&size=400x400&maptype=satellite"
        
      
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={lon-0.001},{lat-0.001},{lon+0.001},{lat+0.001}&bboxSR=4326&size=400,400&format=jpg&f=image"

        try:
           
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
            else:
               
                pass 
            
            time.sleep(0.05)
        except:
            continue

if __name__ == "__main__":
    setup_data_files()
    download_satellite_images()