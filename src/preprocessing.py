import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler

def preprocess_data():
    df = pd.read_excel("data/raw/train.xlsx")
    
    
    df['price_log'] = np.log1p(df['price'])
    
    
    df['house_age'] = 2026 - df['yr_built']
    df['renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
    
    
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'grade', 'lat', 'long', 'house_age']
    
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
    
   
    image_folder = "data/satellite_images"
    downloaded_ids = [int(f.split('.')[0]) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    df_filtered = df[df['id'].isin(downloaded_ids)].copy()
    
    os.makedirs("data/processed", exist_ok=True)
    df_filtered.to_csv("data/processed/train_ready.csv", index=False)
    print(f"✅ Preprocessing Done! Records: {len(df_filtered)}")

if __name__ == "__main__":
    preprocess_data()