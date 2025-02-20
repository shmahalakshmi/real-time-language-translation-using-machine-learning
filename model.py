import pandas as pd
import pickle
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("eng_-french.csv")

# Print columns to confirm structure
print("🔹 Available columns in CSV:", df.columns.tolist())
print(df.head())

# Rename columns for easier access
df.rename(columns={"English words/sentences": "english", "French words/sentences": "french"}, inplace=True)

# Feature extraction functions
def count_vowels(text):
    return len(re.findall(r'[aeiouAEIOU]', text))

def count_consonants(text):
    return len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', text))

# Add new features
df["length"] = df["english"].apply(len)
df["vowel_count"] = df["english"].apply(count_vowels)
df["consonant_count"] = df["english"].apply(count_consonants)

# Define numerical features
numerical_cols = ["length", "vowel_count", "consonant_count"]
X = df[numerical_cols]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create synthetic labels (not ideal for translation, but for demonstration)
y = [0] * len(df)  # Placeholder labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=50)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and scaler
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "model.pkl"), "wb") as model_file:
    pickle.dump(model, model_file)

with open(os.path.join(save_dir, "scaler.pkl"), "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"✅ Model and Scaler saved successfully in {save_dir}/")
