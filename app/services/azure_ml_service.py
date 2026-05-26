import requests
import os
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
API_KEY  = os.getenv("AZURE_ML_API_KEY")

def predict_sales(store: int, date: str, holiday_flag: int,
                  temperature: float, fuel_price: float,
                  cpi: float, unemployment: float,
                  day: int, month: int, year: int) -> dict:
    """
    Call Azure ML real-time endpoint for sales forecasting.
    """
    payload = {
        "Inputs": {
            "input1": [
                {
                    "Store":        store,
                    "Date":         date,        # format: "YYYY-MM-DD"
                    "Holiday_Flag": holiday_flag,
                    "Temperature":  temperature,
                    "Fuel_Price":   fuel_price,
                    "CPI":          cpi,
                    "Unemployment": unemployment,
                    "day":          day,
                    "month":        month,
                    "year":         year
                }
            ]
        },
        "GlobalParameters": {}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result       = response.json()
        output       = result["Results"]["WebServiceOutput0"][0]
        predicted    = output["Scored Labels"]

        return {
            "status":          "success",
            "predicted_sales": round(float(predicted), 2),
            "input": {
                "store": store, "date": date,
                "holiday_flag": holiday_flag,
                "temperature": temperature,
                "fuel_price": fuel_price,
                "cpi": cpi, "unemployment": unemployment,
                "day": day, "month": month, "year": year
            }
        }

    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# Quick test
# -----------------------------
# if __name__ == "__main__":
#     result = predict_sales(
#         store=1,
#         date="2024-05-24",
#         holiday_flag=0,
#         temperature=72.5,
#         fuel_price=3.45,
#         cpi=211.0,
#         unemployment=7.8,
#         day=24,
#         month=5,
#         year=2024
#     )
#     print(result)