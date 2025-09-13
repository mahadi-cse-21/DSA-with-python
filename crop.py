import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('bangladesh_divisions_dataset.csv')

# --- Step 1: Encode categorical features ---
label_cols = [
    'Location', 'Soil_Type', 'Land_Use_Type', 'Season'
]

encoders = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Encode the target column (Crop_Suitability)
target_encoder = LabelEncoder()
data['Crop_Suitability'] = target_encoder.fit_transform(data['Crop_Suitability'])

# --- Step 2: Prepare features and target ---
feature_cols = [
    'Fertility_Index',
    'Location',
    'Soil_Type',
    'Land_Use_Type',
    'Season',
    'Average_Rainfall(mm)',
    'Temperature(°C)',
    'Price'  # ✅ Price included as a feature
]

X = data[feature_cols]
Y = data['Crop_Suitability']

# --- Step 3: Standardize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 4: Train-test split ---
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# --- Step 5: Train model ---
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# --- Step 6: Evaluate model ---
predictions = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(Y_test, predictions))
print("\nClassification Report:\n", classification_report(Y_test, predictions))

# --- Step 7: Make a new prediction ---
new_data = {
    'Location': 'Dhaka',
    'Soil_Type': 'Loamy',
    'Land_Use_Type': 'Agricultural',
    'Season': 'Summer',
    'Average_Rainfall(mm)': 300,
    'Temperature(°C)': 30,
    'Fertility_Index': 63,
    'Price': 60  # Price of Rice (example)
}

new_df = pd.DataFrame([new_data])

# Encode categorical features in new data
for col in label_cols:
    le = encoders[col]
    new_df[col] = le.transform(new_df[col])

# Ensure column order matches training
new_df = new_df[feature_cols]

# Scale and predict
new_scaled = scaler.transform(new_df)
pred = model.predict(new_scaled)
predicted_crop = target_encoder.inverse_transform(pred)

print("\nPredicted Suitable Crop:", predicted_crop[0])

# --- Step 8: Show feature importances ---
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", importance_df)
