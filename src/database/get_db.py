import os
from dotenv import load_dotenv

from src.database.connector import Connector as Database


def get_db():
    load_dotenv()
    username = os.environ.get("DB_USERNAME")
    password = os.environ.get("DB_PASSWORD")
    host = os.environ.get("DB_HOST")
    port = int(os.environ.get("DB_PORT"))
    db = Database(username=username, password=password, host=host, port=port)
    return db
