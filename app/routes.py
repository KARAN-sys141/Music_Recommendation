import random
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_jwt_extended import jwt_required, get_jwt_identity
from .models import db, Playlist, PlaylistSong
import pandas as pd
from .utils import load_data, get_recommendations

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    # Load data
    df = load_data()
    
    # Get top unique artists (let's say we grab artists that appear frequently, or just a random set for discovery)
    # For a deterministic "top" we could count values, but lets just grab a chunk of unique artists
    artists_series = df['artist'].value_counts().head(50).index.tolist()
    
    return render_template('artists.html', artists=artists_series)

@main_bp.route('/artist/<artist_name>')
def artist_detail(artist_name):
    df = load_data()
    # Filter songs by artist
    artist_songs = df[df['artist'].str.lower() == artist_name.lower()].copy()
    
    if artist_songs.empty:
        return "Artist not found", 404
        
    songs = artist_songs.to_dict(orient='records')
    return render_template('artist_detail.html', artist_name=artist_name, songs=songs)

@main_bp.route('/song/<track_id>')
def song_detail(track_id):
    df = load_data()
    song = df[df['track_id'] == track_id].copy()
    
    if song.empty:
        return "Song not found", 404
        
    song_dict = song.iloc[0].to_dict()
    
    # Passing the exact audio features
    features = {
        'danceability': song_dict.get('danceability', 0) * 100,
        'energy': song_dict.get('energy', 0) * 100,
        'valence': song_dict.get('valence', 0) * 100,
        'acousticness': song_dict.get('acousticness', 0) * 100,
        'liveness': song_dict.get('liveness', 0) * 100
    }
    
    return render_template('song_detail.html', song=song_dict, features=features)


@main_bp.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Invalid or empty JSON body'}), 400

    song_name = data.get('song_name', '').strip()
    artist_name = data.get('artist_name', '').strip()
    k = data.get('k', 10)
    filtering = data.get('filtering', 'content')

    if not song_name:
        return jsonify({'error': 'Please enter a song name'}), 400
        
    try:
        recs = get_recommendations(song_name, artist_name, k, filtering)
        return jsonify(recs)
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@main_bp.route('/api/search', methods=['GET'])
def api_search():
    q = request.args.get('q', '').strip().lower()
    if not q:
        return jsonify([])
        
    df = load_data()
    matches = df[df['name'].str.lower().str.contains(q, na=False) | df['artist'].str.lower().str.contains(q, na=False)].head(20)
    
    if matches.empty:
        return jsonify([])
        
    res = matches[['track_id', 'name', 'artist', 'spotify_preview_url']].fillna('').to_dict(orient='records')
    return jsonify(res)

@main_bp.route('/api/playlist/add', methods=['POST'])
@jwt_required()
def add_to_playlist():
    user_id = int(get_jwt_identity())
    data = request.get_json()
    song_id = data.get('song_id')
    song_name = data.get('song_name')
    artist_name = data.get('artist_name')
    preview_url = data.get('preview_url')
    
    if not all([song_id, song_name, artist_name]):
        return jsonify({"error": "Missing song data"}), 400
        
    playlist = Playlist.query.filter_by(user_id=user_id, name="Liked Songs").first()
    if not playlist:
        playlist = Playlist(name="Liked Songs", user_id=user_id)
        db.session.add(playlist)
        db.session.commit()
        
    if PlaylistSong.query.filter_by(playlist_id=playlist.id, song_id=song_id).first():
        return jsonify({"message": "Already in playlist"}), 200
        
    ps = PlaylistSong(playlist_id=playlist.id, song_id=song_id, song_name=song_name, artist_name=artist_name, preview_url=preview_url)
    db.session.add(ps)
    db.session.commit()
    
    return jsonify({"message": "Added to Liked Songs"}), 201

@main_bp.route('/auth')
def auth_page():
    return render_template('auth.html')

@main_bp.route('/playlists')
@jwt_required(optional=True)
def playlists():
    raw_id = get_jwt_identity()
    if not raw_id:
        return redirect(url_for('main.auth_page'))
    user_id = int(raw_id)
    playlist = Playlist.query.filter_by(user_id=user_id, name="Liked Songs").first()
    songs = []
    if playlist:
        songs = PlaylistSong.query.filter_by(playlist_id=playlist.id).order_by(PlaylistSong.added_at.desc()).all()
    return render_template('playlists.html', songs=songs)


