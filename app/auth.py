from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, set_access_cookies, unset_jwt_cookies, get_jwt_identity
import bcrypt
from .models import db, User, Playlist

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
        
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
        
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
        
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(username=username, password_hash=hashed)
    db.session.add(new_user)
    db.session.commit()
    
    # Create default playlist
    default_playlist = Playlist(name="Liked Songs", user_id=new_user.id)
    db.session.add(default_playlist)
    db.session.commit()
    
    return jsonify({"message": "User created successfully"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
        return jsonify({"error": "Invalid username or password"}), 401
        
    access_token = create_access_token(identity=str(user.id))
    resp = jsonify({'login': True, 'username': user.username})
    set_access_cookies(resp, access_token)
    return resp, 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    resp = jsonify({'logout': True})
    unset_jwt_cookies(resp)
    return resp, 200

@auth_bp.route('/me', methods=['GET'])
@jwt_required(optional=True)
def me():
    current_user_id = get_jwt_identity()
    if current_user_id:
        user = User.query.get(int(current_user_id))
        if user:
            return jsonify({"logged_in": True, "username": user.username}), 200
    return jsonify({"logged_in": False}), 200
