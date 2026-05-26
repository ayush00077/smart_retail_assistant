from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI, DATABASE_NAME
print("Mongo URI:", MONGO_URI)

client = AsyncIOMotorClient(MONGO_URI)

database = client[DATABASE_NAME]

sales_collection = database["sales"]