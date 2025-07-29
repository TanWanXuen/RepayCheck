from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
from queue import Queue

# To load .env file
load_dotenv()

db = SQLAlchemy()
db_queue = Queue()

# To initiate database
def init_db(app, testing=False):
    if testing:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    else:
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        host = os.getenv("MYSQL_HOST")
        port = os.getenv("MYSQL_PORT")
        database = os.getenv("MYSQL_DATABASE")

        uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

        app.config["SQLALCHEMY_DATABASE_URI"] = uri
        print(app.config["SQLALCHEMY_DATABASE_URI"])
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

