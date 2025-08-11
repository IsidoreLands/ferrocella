# run_server.py
# The main entry point to start the Ferrocella API server.

from flask import Flask
from api.routes import api_bp
import config

# The app is now created at the top level
app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    # This block is for direct `python run_server.py` for debugging
    print(f"-> Starting Ferrocella server in DEBUG mode...")
    print(f"-> API available under http://{config.SERVER_HOST}:{config.SERVER_PORT}/api")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=True)
