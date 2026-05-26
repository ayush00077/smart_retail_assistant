import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Load dataset
df = pd.read_csv("/Users/ayush/Desktop/Sprint-project/data/raw/Walmart.csv")

# Standardize column names
df.columns = df.columns.str.lower()

# Standardize date column
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Select features for anomaly detection
X = df[["weekly_sales", "temperature", "fuel_price", "cpi", "unemployment"]]

# Train model
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
model.fit(X)

# Predict anomalies
df["anomaly"] = model.predict(X)
df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

# Anomaly score (lower = more anomalous)
df["anomaly_score"] = model.decision_function(X)

# Count
anomaly_count = (df["anomaly"] == "Anomaly").sum()
print(f"Total anomalies detected: {anomaly_count}")

# Save model pickle
with open("../models/anomaly.pkl", "wb") as f:
    pickle.dump(model, f)
print("Anomaly detection model saved successfully")

# -----------------------------
# Save to MongoDB
# -----------------------------
client = MongoClient(os.getenv("MONGO_URI"))
db = client["smart_retail"]
collection = db["anomalies"]

# Keep all useful columns
save_df = df[["date", "store", "weekly_sales", "temperature",
              "fuel_price", "cpi", "unemployment",
              "anomaly", "anomaly_score"]].copy()

save_df["date"]     = save_df["date"].astype(str)   # MongoDB-friendly
save_df["saved_at"] = datetime.utcnow()

records = save_df.to_dict(orient="records")

collection.delete_many({})          # clear old results
collection.insert_many(records)

print(f"✅ {len(records)} anomaly records saved to MongoDB → smart_retail.anomalies")
print(f"   → {anomaly_count} flagged as Anomaly")
print(f"   → {len(records) - anomaly_count} flagged as Normal")

client.close()