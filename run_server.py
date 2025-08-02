# run_server.py
# The main entry point to start the Ferrocella API server.

from flask import Flask
from api.routes import api_bp # Import our API blueprint
import config

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    # Register the blueprint, which contains all our API endpoints
    app.register_blueprint(api_bp, url_prefix='/api')
    return app

if __name__ == '__main__':
    app = create_app()
    print(f"-> Starting Ferrocella server...")
    print(f"-> API endpoints available under http://{config.SERVER_HOST}:{config.SERVER_PORT}/api")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=True)
