from flask import Flask
from models import db  # Import the db object (this won't cause a circular import)

# Initialize the Flask app
app = Flask(__name__)

# Configure the app with database details
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost/attendance_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database with the app
db.init_app(app)

# Define a simple route to test
@app.route('/')
def index():
    return "University Attendance System"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
