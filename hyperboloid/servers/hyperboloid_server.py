from flask import Flask
from hyperboloid_api_routes import api_bp
import hyperboloid_config as config

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    print(f"-> Starting Ferrocella Toroid API Server...")
    print(f"-> API available under http://{config.SERVER_HOST}:{config.SERVER_PORT}/api")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=True)
