import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load your dataset
df = pd.read_csv("fetal_health.csv")  # update with your dataset path

# 2️⃣ Prepare data
X = df.drop(columns=['fetal_health'])
y = df['fetal_health'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3️⃣ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4️⃣ Train model with the current sklearn version
model = GradientBoostingClassifier(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

# 5️⃣ Save model and scaler
joblib.dump(model, "gradient_boosting_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler retrained and saved successfully with sklearn 1.7.2!")
