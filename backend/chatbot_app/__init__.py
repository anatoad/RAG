from flask import Flask
from flask_cors import CORS

def create_app(config_filename=None):
    app = Flask(__name__)
    cors = CORS(app)
    
    # Load configuration
    if config_filename:
        app.config.from_pyfile(config_filename)
    else:
        app.config.from_object('config.Config')
    
    # Import and register the main blueprint
    from chatbot_app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint, url_prefix="/api")
    
    return app
