from flask_sqlalchemy import SQLAlchemy

# Initialize the database object
db = SQLAlchemy()

# Import the models here (after db is initialized)
from models.models import Student, Teacher, Attendance, Class
