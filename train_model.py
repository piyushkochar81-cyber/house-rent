
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv("House_Rent_Dataset.csv")

# 2. Select required columns
df = df[["City", "Size", "Rent"]].copy()

# 3. Data preprocessing
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
df["Rent"] = pd.to_numeric(df["Rent"], errors="coerce")

df["City"] = df["City"].astype("string").str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=["City", "Size", "Rent"], inplace=True)

# Remove invalid records
df = df[
    (df["Size"] > 0) &
    (df["Rent"] > 0) &
    (df["City"] != "")
]

print("Total records:", len(df))

# 4. Define input and output
X = df[["City", "Size"]]
y = df["Rent"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Encode location
preprocessor = ColumnTransformer(
    transformers=[
        ("city", OneHotEncoder(handle_unknown="ignore"), ["City"]),
        ("size", "passthrough", ["Size"])
    ]
)

# 7. Create Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

# 8. Train model
print("Training Random Forest model...")

model.fit(X_train, y_train)

# 9. Evaluate model
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nModel Evaluation")
print("-------------------------")
print("RMSE:", round(rmse, 2))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 10. Save model and information
model_data = {
    "model": model,
    "cities": sorted(df["City"].unique().tolist()),
    "min_size": float(df["Size"].min()),
    "max_size": float(df["Size"].max()),
    "rmse": float(rmse),
    "total_records": len(df)
}

joblib.dump(model_data, "rent_model.pkl")

print("\nModel successfully saved as rent_model.pkl")
print("Training completed!")
