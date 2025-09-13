import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)

n_samples = 1000

soil_types = ['Loamy', 'Sandy', 'Clay', 'Peaty', 'Silt']
locations = ['Dhaka', 'Sylhet', 'Rajshahi', 'Khulna', 'Rangpur', 'Barishal', 'Chattogram']
seasons = ['Winter', 'Summer', 'Monsoon', 'Autumn']
land_use_types = ['Agricultural', 'Residential', 'Barren', 'Unused']
irrigation_options = ['Yes', 'No']
pest_disease_levels = ['None', 'Low', 'Moderate', 'High']
previous_crops = ['Rice', 'Wheat', 'Maize', 'Vegetables', 'Jute', 'Tea', 'None']

# Generate data
data = {
    'Soil_Type': np.random.choice(soil_types, n_samples),
    'Fertility_Index': np.random.randint(40, 101, n_samples),
    'Location': np.random.choice(locations, n_samples),
    'Season': np.random.choice(seasons, n_samples),
    'Average_Rainfall(mm)': np.round(np.random.uniform(50, 400, n_samples), 2),
    'Temperature(°C)': np.round(np.random.uniform(15, 40, n_samples), 2),
    'Land_Use_Type': np.random.choice(land_use_types, n_samples),
    'Market_Price': np.random.randint(20, 501, n_samples),
    'Soil_pH': np.round(np.random.uniform(4.5, 8.5, n_samples), 2),
    'Sunlight_Exposure(hours)': np.round(np.random.uniform(4, 12, n_samples), 2),
    'Irrigation_Availability': np.random.choice(irrigation_options, n_samples),
    'Pest_and_Disease_History': np.random.choice(pest_disease_levels, n_samples),
    'Previous_Crop': np.random.choice(previous_crops, n_samples),
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_crop_recommendation_dataset.csv', index=False)

print("Synthetic dataset generated and saved as 'synthetic_crop_recommendation_dataset.csv'")
