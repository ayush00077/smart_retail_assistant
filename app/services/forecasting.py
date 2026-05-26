# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# import pickle
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Load dataset
# df = pd.read_csv("/Users/ayush/Desktop/Sprint-project/data/raw/Walmart.csv")

# # Standardize column names
# df.columns = df.columns.str.lower()

# # Convert date column
# df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# # Feature engineering
# df["year"] = df["date"].dt.year
# df["month"] = df["date"].dt.month
# df["week"] = df["date"].dt.isocalendar().week

# # Features
# X = df[
#     [
#         "store",
#         "holiday_flag",
#         "temperature",
#         "fuel_price",
#         "cpi",
#         "unemployment",
#         "year",
#         "month",
#         "week"
#     ]
# ]

# # Target
# y = df["weekly_sales"]

# # Train test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# # Create model
# model = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42
# )

# # Train model
# model.fit(X_train, y_train)

# # Predictions
# predictions = model.predict(X_test)

# # Evaluate model
# # MAE
# mae = mean_absolute_error(y_test, predictions)

# # RMSE
# rmse = np.sqrt(mean_squared_error(y_test, predictions))

# # R2 Score
# r2 = r2_score(y_test, predictions)

# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")

# # Save model
# with open("../models/forecast.pkl", "wb") as f:
#     pickle.dump(model, f)

# print("Forecast model saved successfully")


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Load dataset
df = pd.read_csv("/Users/ayush/Desktop/Sprint-project/data/raw/Walmart.csv")

# Standardize column names
df.columns = df.columns.str.lower()

# Convert date column
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Feature engineering
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.isocalendar().week

# Features & Target
X = df[["store", "holiday_flag", "temperature", "fuel_price", "cpi", "unemployment", "year", "month", "week"]]
y = df["weekly_sales"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate
mae  = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2   = r2_score(y_test, predictions)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Save model pickle
with open("../models/forecast.pkl", "wb") as f:
    pickle.dump(model, f)
print("Forecast model saved successfully")

# -----------------------------
# Save predictions to MongoDB
# -----------------------------
client = MongoClient(os.getenv("MONGO_URI"))
db = client["smart_retail"]
collection = db["forecasts"]

# Build prediction records from test set
X_test_copy = X_test.copy()
X_test_copy["predicted_sales"] = predictions
X_test_copy["actual_sales"]    = y_test.values
X_test_copy["error"]           = abs(X_test_copy["actual_sales"] - X_test_copy["predicted_sales"])
X_test_copy["saved_at"]        = datetime.utcnow()

# Convert to list of dicts and insert
records = X_test_copy.to_dict(orient="records")

collection.delete_many({})           # clear old predictions before inserting fresh
collection.insert_many(records)

print(f"✅ {len(records)} forecast records saved to MongoDB → smart_retail.forecasts")
client.close()