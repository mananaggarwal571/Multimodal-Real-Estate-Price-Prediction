# Multimodal House Price Prediction (Satellite Imagery + Tabular Data)

This project implements a multimodal deep learning model to predict real estate prices. It combines traditional property features (tabular data) with geographical context extracted from satellite imagery.

## 🚀 Final Performance
- **R2 Score:** 0.706
- **Target Variable:** Log-transformed price (Price_log)
- **Framework:** PyTorch, Torchvision

## 🏗️ Model Architecture
The model uses a dual-branch fusion architecture:
1. **Vision Branch:** Pre-trained **ResNet18** to extract spatial features from satellite images.
2. **Tabular Branch:** A Multi-Layer Perceptron (MLP) to process numerical features like square footage, location, and property age.
3. **Fusion:** Features from both branches are concatenated and passed through final fully connected layers for regression.

## 🛠️ Features Used
- **Tabular:** Bedrooms, Bathrooms, Sqft_living, Sqft_lot, Floors, Waterfront, View, Condition, Grade, Latitude, Longitude, House_Age.
- **Visual:** 400x400 RGB Satellite images (Esri World Imagery).

## 📁 Project Structure
- `src/analysis.py`: Exploratory Data Analysis (EDA) and price distribution visualization(for saving images locally in your machine).
- `src/data-fetcher.py`: Automated script to download the Excel dataset and fetch satellite imagery via Esri API.
- `src/dataset.py`: Custom PyTorch Dataset handling images and tabular data.
- `src/explain.py`: Implementation of Grad-CAM to visualize which parts of the satellite image (e.g., greenery, roads) influenced the price.
- `src/final_comparison.py`: Custom PyTorch Dataset handling images and tabular data.
- `src/preprocessing.py`: Handles feature engineering (house age, total sqft), outlier removal, and RobustScaler application.
- `src/model.py`: Multimodal architecture definition.
- `src/train.py`: Training script with differential learning rates and OneCycleLR.
- `src/evaluation.py`: Script to calculate R2/RMSE and generate submission file.
- `01_Analysis.ipynb`: This .ipynb contains all the analysis done for Exploratory Data Analysis(EDA)
- `final_predictions.csv`: Final output file for the test set.

## ⚙️ How to Run
1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess & Download Data:**
    Run the data fetcher and preprocessing scripts to prepare
   ```bash
   train_ready.csv
   ```
3. **Training:**
    ```bash
    python src/train.py
    ```
4. **Evaluation & Prediction:**
    ```bash
    python src/evaluation.py
    ```
5. **Grad-Cam Analysis**
     50 examples of satellite images have been stored in the reports folder which show the Grad-Cam analysis describing which part have been considered by the model for deciding the prices.
6. **Plot of prediction**
     Plot of actual prices and predicted prices has been stored in the reports folder with the name
     ```bash
       evaluation_results.png
     ```
7. **Screenshot of R2 score and RMSE Score**
   
     <img width="366" height="153" alt="Screenshot 2026-01-07 184001" src="https://github.com/user-attachments/assets/96134c02-1d0e-4576-9bbe-b5a3d1973b8b" />
     
8.  **Screenshot of Comparison between baseline model(using only csv) and my model(using both satellite images and csv file)**
   
       <img width="714" height="234" alt="Screenshot 2026-01-07 184537" src="https://github.com/user-attachments/assets/42c5cc94-2ed7-4f8a-a26a-b3a89b27084e" />

         
