from fastapi import APIRouter, UploadFile, File
import pandas as pd
from app.database import sales_collection

router = APIRouter()

@router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):

    # Read CSV
    df = pd.read_csv(file.file)

    # Clean column names
    df.columns = df.columns.str.lower()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Convert date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            dayfirst=True
        ).astype(str)

    # Convert dataframe to records
    records = df.to_dict("records")

    # Insert into MongoDB
    await sales_collection.insert_many(records)

    return {
        "message": "Preprocessed data uploaded successfully",
        "rows_inserted": len(records)
    }