import os
from flask import Flask
from .routes import main_bp
from .models import db
from .auth import auth_bp
from flask_jwt_extended import JWTManager

def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    
    # Configure SQLite database
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(base_dir, 'data', 'app.db')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Configure JWT
    app.config['JWT_SECRET_KEY'] = 'super-secret-key-music-rec'
    app.config['JWT_TOKEN_LOCATION'] = ['cookies']
    app.config['JWT_COOKIE_CSRF_PROTECT'] = False # For simplicity
    app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
    
    db.init_app(app)
    JWTManager(app)
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    
    with app.app_context():
        db.create_all()

    return app
