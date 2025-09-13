import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load synthetic dataset generated before
df = pd.read_csv('synthetic_crop_recommendation_dataset.csv')

# For demo: create a random target variable Crop_Suitability
crops = ['Rice', 'Wheat', 'Maize', 'Vegetables', 'Jute', 'Tea', 'Spices']
np.random.seed(42)
df['Crop_Suitability'] = np.random.choice(crops, size=len(df))

# Separate features and target
X = df.drop('Crop_Suitability', axis=1)
y = df['Crop_Suitability']

# Encode categorical variables
categorical_cols = ['Soil_Type', 'Location', 'Season', 'Land_Use_Type',
                    'Irrigation_Availability', 'Pest_and_Disease_History', 'Previous_Crop']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target variable
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# Scale numeric features
numeric_cols = ['Fertility_Index', 'Average_Rainfall(mm)', 'Temperature(°C)', 'Market_Price',
                'Soil_pH', 'Sunlight_Exposure(hours)']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(f"Training Accuracy: {model.score(X_train, y_train):.2f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.2f}")

# Function to predict crop given new input
def predict_crop(input_dict):
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Encode categorical
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Scale numeric
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    pred_encoded = model.predict(input_df)[0]
    pred_crop = target_le.inverse_transform([pred_encoded])[0]

    return pred_crop

# Example usage:
new_sample = {
    'Soil_Type': 'Loamy',
    'Fertility_Index': 65,
    'Location': 'Dhaka',
    'Season': 'Monsoon',
    'Average_Rainfall(mm)': 200,
    'Temperature(°C)': 30,
    'Land_Use_Type': 'Agricultural',
    'Market_Price': 100,
    'Soil_pH': 6.5,
    'Sunlight_Exposure(hours)': 8,
    'Irrigation_Availability': 'Yes',
    'Pest_and_Disease_History': 'Low',
    'Previous_Crop': 'Rice'
}

predicted_crop = predict_crop(new_sample)
print(f"Predicted suitable crop: {predicted_crop}")
