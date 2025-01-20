"""
Flask app entry point
"""

from flask import Flask

from api.prediction import prediction_bp


# Create the Flask app
app = Flask(__name__)
app.register_blueprint(prediction_bp)

if __name__ == '__main__':
    app.run()