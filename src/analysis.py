import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel("data/raw/train.xlsx")

# 1. Price Distribution (Humein pata hona chahiye ki ghar kitne me bik rahe hain)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=50)
plt.title('House Price Distribution')
plt.show()

# 2. Location Map (Lat/Long check)
plt.figure(figsize=(10, 8))
plt.scatter(df['long'], df['lat'], c=df['price'], cmap='viridis', s=1, alpha=0.5)
plt.colorbar(label='Price')
plt.title('Property Locations & Prices')
plt.show()