import mysql.connector
from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker
from src.database.base import Base


class Connector:
    def __init__(
        self,
        username: str,
        password: str,
        db_name: str = "shogi_ai",
        host: str = "localhost",
        port: int = 3360,
    ):
        """
        Initializes the Database connection.

        Args:
            username (str): Username for the database.
            password (str): Password for the database.
            db_name (str): Name of the database. Default is 'shogi_ai'.
            host (str): Host address of the database. Default is 'localhost'.
            port (int): Port number for the database. Default is 3360.
        """
        self.db_name = db_name
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.db_url = f"mysql://{username}:{password}@{host}:{port}/{db_name}"

        self.base = Base
        self.engine = create_engine(self.db_url)
        self.session_maker = sessionmaker(bind=self.engine)

    def create(self):
        connection = mysql.connector.connect(
            host=self.host,
            user=self.username,
            password=self.password,
            port=self.port,
        )

        # Create a cursor object to execute SQL commands
        cursor = connection.cursor()

        # Execute SQL command to create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")

        # Close the cursor and connection
        cursor.close()
        connection.close()

        # Create tables based on metabase
        self.base.metadata.create_all(self.engine)
        self.session_maker = sessionmaker(bind=self.engine)
