
from flask import Flask, request, jsonify, render_template
import pandas as pd
from scipy.sparse import load_npz
from numpy import load
from content_based_filtering import recommend as content_recommendation
from collaborative_filtering import collaborative_recommendation
from hybrid_recommendation import HybridRecommendatonSystem as hrs

app = Flask(__name__)

_songs_data = None
_filtered_data = None
_transformed_data = None
_collab_matrix = None
_track_ids = None
_interaction_matrix = None
_transformed_hybrid_data = None

def load_data():
    global _songs_data
    if _songs_data is None:
        _songs_data = pd.read_csv("data/cleaned_data.csv")
    return _songs_data

def load_filtered_data():
    global _filtered_data
    if _filtered_data is None:
        _filtered_data = pd.read_csv("data/collab_filtered_data.csv")
    return _filtered_data

def load_transformed_data():
    global _transformed_data
    if _transformed_data is None:
        _transformed_data = load_npz("data/transformed_data.npz")
    return _transformed_data

def load_collab_matrix():
    global _collab_matrix
    if _collab_matrix is None:
        _collab_matrix = load_npz("data/interaction_matrix.npz")
    return _collab_matrix

def load_track_ids():
    global _track_ids
    if _track_ids is None:
        _track_ids = load("data/track_ids.npy", allow_pickle=True)
    return _track_ids

def load_interaction_matrix():
    global _interaction_matrix
    if _interaction_matrix is None:
        _interaction_matrix = load_npz("data/interaction_matrix.npz")
    return _interaction_matrix

def load_transformed_hybrid_data():
    global _transformed_hybrid_data
    if _transformed_hybrid_data is None:
        _transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")
    return _transformed_hybrid_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # data = request.get_json()
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Invalid or empty JSON body'}), 400

    song_name = data.get('song_name', '').strip()
    artist_name = data.get('artist_name', '').strip()
    k = data.get('k', 10)
    filtering = data.get('filtering', 'content')

    songs_data = load_data()
    filtered_data = load_filtered_data()
    transformed_data = load_transformed_data()
    collab_matrix = load_collab_matrix()
    track_ids = load_track_ids()
    interaction_matrix = load_interaction_matrix()
    transformed_hybrid_data = load_transformed_hybrid_data()

    if not song_name:
        return jsonify({'error': 'Please enter a song name'}), 400

    song_name_lower = song_name.lower()
    artist_name_lower = artist_name.lower()

    try:
        if filtering == 'content':
            if artist_name:
                match = songs_data[
                    (songs_data['name'].str.lower() == song_name_lower) &
                    (songs_data['artist'].str.lower() == artist_name_lower)
                ]
            else:
                match = songs_data[
                    (songs_data['name'].str.lower() == song_name_lower)
                ]

            if match.empty:
                return jsonify([])

            current_song = match.iloc[0:1]
            recommendations = content_recommendation(song_name_lower, songs_data, transformed_data, k)

            recommendations = recommendations[~(
                (recommendations['name'].str.lower() == current_song['name'].values[0].lower()) &
                (recommendations['artist'].str.lower() == current_song['artist'].values[0].lower())
            )]

            recommendations = pd.concat([current_song, recommendations]).reset_index(drop=True)
            return jsonify(recommendations.fillna('').to_dict(orient='records'))

        elif filtering == 'collab':
            if artist_name:
                match = filtered_data[
                    (filtered_data['name'].str.lower() == song_name_lower) &
                    (filtered_data['artist'].str.lower() == artist_name_lower)
                ]
            else:
                match = filtered_data[
                    (filtered_data['name'].str.lower() == song_name_lower)
                ]

            if match.empty:
                return jsonify([])

            current_song = match.iloc[0:1]
            selected_artist = current_song['artist'].values[0]

            recommendations = collaborative_recommendation(
                song_name_lower,
                selected_artist.lower(),
                track_ids,
                filtered_data,
                collab_matrix,
                k
            )

            recommendations = recommendations[~(
                (recommendations['name'].str.lower() == current_song['name'].values[0].lower()) &
                (recommendations['artist'].str.lower() == current_song['artist'].values[0].lower())
            )]

            recommendations = pd.concat([current_song, recommendations]).reset_index(drop=True)
            return jsonify(recommendations.fillna('').to_dict(orient='records'))

        elif filtering == 'hybrid':
            if artist_name:
                match = filtered_data[
                    (filtered_data['name'].str.lower() == song_name_lower) &
                    (filtered_data['artist'].str.lower() == artist_name_lower)
                ]
            else:
                match = filtered_data[
                    (filtered_data['name'].str.lower() == song_name_lower)
                ]

            if match.empty:
                return jsonify([])

            current_song = match.iloc[0:1]
            selected_name = current_song['name'].values[0]
            selected_artist = current_song['artist'].values[0]

            recommender = hrs(
                song_name=selected_name,
                artist_name=selected_artist,
                number_of_recommendations=k,
                weight_content_based=0.3,
                weight_collaborative=0.7,
                songs_data=filtered_data,
                transformed_matrix=transformed_hybrid_data,
                track_ids=track_ids,
                interaction_matrix=interaction_matrix
            )

            recommendations = recommender.give_recommendations()

            recommendations = recommendations[~(
                (recommendations['name'] == selected_name) &
                (recommendations['artist'] == selected_artist)
            )]

            recommendations = pd.concat([current_song, recommendations]).reset_index(drop=True)
            return jsonify(recommendations.fillna('').to_dict(orient='records'))

        else:
            return jsonify({'error': 'Invalid filtering method selected'}), 400

    except Exception as e:
        return jsonify({'error': f'Error...: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
