from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env")

client = MongoClient(MONGO_URI)

db = client["logs"]  # your DB name
collection = db["predicted_logs"]  # your collection name


def get_collection():
    return collection