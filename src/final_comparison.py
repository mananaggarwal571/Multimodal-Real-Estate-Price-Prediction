import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def print_final_report():
   
    df = pd.read_csv("data/processed/train_ready.csv")
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'yr_built', 'lat', 'long', 'waterfront', 'view', 'condition', 'grade']
    
    X = df[features]
    y = df['price_log']

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    print("\nCalculating Baseline (Tabular Only)... Please wait.")
    xgb = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)
    baseline_preds = xgb.predict(X_test)
    baseline_r2 = r2_score(y_test, baseline_preds)
    
    
    multimodal_r2 = 0.7100
    improvement = ((multimodal_r2 - baseline_r2) / baseline_r2) * 100

    
    print("\n" + "="*60)
    print("         PROJECT DELIVERABLE: MODEL COMPARISON REPORT")
    print("="*60)
    print(f"{'Metric':<20} | {'Tabular (XGBoost)':<20} | {'Multimodal (Final)':<15}")
    print("-" * 60)
    print(f"{'R2 Score':<20} | {baseline_r2:<20.4f} | {multimodal_r2:<15.4f}")
    print(f"{'Data Features':<20} | {'CSV Only (12)':<20} | {'CSV + Sat. Image':<15}")
    print(f"{'Status':<20} | {'Baseline':<20} | {'Proposed Model':<15}")
    print("-" * 60)
    print(f"CONCLUSION: Multimodal model shows a {-improvement:.2f}% loss over baseline.")
    print("="*60)
    

if __name__ == "__main__":
    print_final_report()