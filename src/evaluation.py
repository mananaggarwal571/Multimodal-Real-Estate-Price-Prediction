import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import MultimodalHousingDataset, data_transforms
from src.model import MultimodalModel

def evaluate_and_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Starting Evaluation & Prediction on {device}...")

 
    model = MultimodalModel(num_tabular_features=12).to(device)
    model_path = "models/multimodal_house_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Model Loaded Successfully")
    else:
        print("❌ Error: Model file missing! Pehle train.py chalao.")
        return

    
    csv_path = "data/processed/train_ready.csv" 
    if os.path.exists(csv_path):
        dataset = MultimodalHousingDataset(csv_path, "data/satellite_images", transform=data_transforms)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        log_actuals, log_predictions = [], []
        print("📊 Calculating Metrics...")
        with torch.no_grad():
            for images, tabular, targets in tqdm(loader, desc="Evaluating"):
                images, tabular = images.to(device), tabular.to(device)
                outputs = model(images, tabular)
                log_predictions.extend(outputs.cpu().numpy().flatten())
                log_actuals.extend(targets.cpu().numpy().flatten())

        
        actual_prices = np.expm1(np.array(log_actuals))
        predicted_prices = np.expm1(np.array(log_predictions))

      
        r2 = r2_score(actual_prices, predicted_prices)
        rmse_cost = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        rmse_log = np.sqrt(mean_squared_error(np.array(log_actuals), np.array(log_predictions)))

        print("\n" + "="*40)
        print(f"📈 R2 SCORE (Decimal): {r2:.4f}")
        print(f"📉 RMSE (Actual Cost): ${rmse_cost:,.2f}")
        print(f"📉 RMSE (Log Scale):   {rmse_log:.4f}")
        print("="*40)

      
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_prices, predicted_prices, alpha=0.4, color='teal', label='Data Points')
        lims = [0, max(actual_prices.max(), predicted_prices.max())]
        plt.plot(lims, lims, 'r--', lw=2, label='Ideal Prediction')
        plt.title(f"Evaluation: R2={r2:.4f} | RMSE=${rmse_cost:,.0f}")
        plt.xlabel("Actual Price ($)")
        plt.ylabel("Predicted Price ($)")
        plt.legend()
        os.makedirs("reports", exist_ok=True)
        plt.savefig("reports/evaluation_results.png")
        print("🎨 Plot saved in 'reports/evaluation_results.png'")
    else:
        print("⚠️ train_ready.csv missing, skipping evaluation metrics.")

    
    print("\n🚀 Generating Final predictions.csv...")
    test_excel = "data/raw/test2.xlsx"
    if os.path.exists(test_excel):
        test_df = pd.read_excel(test_excel)
        
     
        test_df['house_age'] = 2026 - test_df['yr_built']
        features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                    'waterfront', 'view', 'condition', 'grade', 'lat', 'long', 'house_age']
        
        scaler = RobustScaler()
        test_df[features] = scaler.fit_transform(test_df[features])
        test_df['price_log'] = 0 # Dummy target
        
        test_ready_path = "data/processed/test_ready.csv"
        test_df.to_csv(test_ready_path, index=False)

        test_dataset = MultimodalHousingDataset(test_ready_path, "data/satellite_images", transform=data_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        submission_results = []
        with torch.no_grad():
            for i, (image, tabular, _) in enumerate(tqdm(test_loader, desc="Predicting Test")):
                image, tabular = image.to(device), tabular.to(device)
                output = model(image, tabular)
                
                final_price = np.expm1(output.item())
                row_id = test_df.iloc[i]['id']
                submission_results.append([int(row_id), final_price])


        pd.DataFrame(submission_results, columns=['id', 'predicted_price']).to_csv("predictions.csv", index=False)
        print("✅ Done! 'predictions.csv' is ready for submission.")
    else:
        print("❌ Error: test2.xlsx missing in data/raw/")

if __name__ == "__main__":
    evaluate_and_generate()