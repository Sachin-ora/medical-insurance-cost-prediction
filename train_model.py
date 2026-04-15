# train_stacking_model.py

import os
import pandas as pd
import joblib
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---- Load dataset ----
possible = ["Medical Insurance Predict\insurance-predict Web Application Folder\medical_insurance.csv"]
csv_path = next((p for p in possible if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("medical_insurance.csv not found. Place it in the project root (or ./data/).")

df = pd.read_csv(csv_path)
print("Loaded", csv_path, "shape:", df.shape)

if 'charges' not in df.columns:
    raise ValueError("'charges' column (target) not found in CSV.")

# ---- Features & target ----
X = df.drop(columns=['charges'])
y = df['charges']

numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# ---- Encoder compatibility ----
try:
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_encoder, categorical_features)
    ]
)

# ---- Base models ----
rf = RandomForestRegressor(n_estimators=100, random_state=42)
dt = DecisionTreeRegressor(random_state=42)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)

# ---- Stacking model ----
stacking_model = StackingRegressor(
    estimators=[
        ('rf', rf),
        ('dt', dt),
        ('xgb', xgb)
    ],
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

# ---- Full pipeline ----
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', stacking_model)
])

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training stacking ensemble model...")
pipeline.fit(X_train, y_train)

# ---- Evaluate ----
preds = pipeline.predict(X_test)

try:
    rmse = mean_squared_error(y_test, preds, squared=False)
except TypeError:
    rmse = sqrt(mean_squared_error(y_test, preds))

r2 = r2_score(y_test, preds)

print(f"Stacking Model RMSE: {rmse:.2f}  R2: {r2:.3f}")

# ---- Save model ----
model_path = "model_stacking.pkl"
joblib.dump(pipeline, model_path)

print("Saved stacking model to", model_path)