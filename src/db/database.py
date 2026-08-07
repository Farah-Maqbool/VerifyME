import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["verifyme"]
employees_collection = db["employees"]


def enroll_employee(employee_id, name, full_face_embedding, periocular_embedding):
    """
    Stores a new employee record with both embeddings.
    Embeddings should be numpy arrays or lists of floats.
    """
    record = {
        "employee_id": employee_id,
        "name": name,
        "full_face_embedding": list(map(float, full_face_embedding)),
        "periocular_embedding": list(map(float, periocular_embedding)),
    }
    result = employees_collection.insert_one(record)
    return result.inserted_id


def get_all_employees():
    """
    Returns all enrolled employees (for matching against).
    """
    return list(employees_collection.find())


def get_employee_by_id(employee_id):
    return employees_collection.find_one({"employee_id": employee_id})


def delete_employee(employee_id):
    result = employees_collection.delete_one({"employee_id": employee_id})
    return result.deleted_count