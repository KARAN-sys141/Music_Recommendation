import pandas as pd
from scipy.sparse import load_npz
from numpy import load
from src.models.content_based_filtering import recommend as content_recommendation
from src.models.collaborative_filtering import collaborative_recommendation
from src.models.hybrid_recommendation import HybridRecommendatonSystem as hrs

_songs_data = None
_filtered_data = None
_transformed_data = None
_collab_matrix = None
_track_ids = None
_interaction_matrix = None
_transformed_hybrid_data = None

import json
_offline_predictions = None

def load_offline_predictions():
    global _offline_predictions
    if _offline_predictions is None:
        try:
            with open('offline_predictions.json', 'r', encoding='utf-8') as f:
                _offline_predictions = json.load(f)
        except Exception:
            _offline_predictions = {}
    return _offline_predictions

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

def get_recommendations(song_name, artist_name, k, filtering):
    song_name_lower = song_name.lower().strip()
    artist_name_lower = artist_name.lower().strip() if artist_name else ''
    
    songs_data = load_data()
    filtered_data = load_filtered_data()
    transformed_data = load_transformed_data()
    collab_matrix = load_collab_matrix()
    track_ids = load_track_ids()
    interaction_matrix = load_interaction_matrix()
    transformed_hybrid_data = load_transformed_hybrid_data()

    if filtering == 'content':
        if artist_name_lower:
            match = songs_data[
                (songs_data['name'].str.lower() == song_name_lower) &
                (songs_data['artist'].str.lower() == artist_name_lower)
            ]
        else:
            match = songs_data[
                (songs_data['name'].str.lower() == song_name_lower)
            ]

        if match.empty:
            return []

        current_song = match.iloc[0:1]
        recommendations = content_recommendation(song_name_lower, songs_data, transformed_data, k)

        recommendations = recommendations[~(
            (recommendations['name'].str.lower() == current_song['name'].values[0].lower()) &
            (recommendations['artist'].str.lower() == current_song['artist'].values[0].lower())
        )]

        recommendations = pd.concat([current_song, recommendations]).reset_index(drop=True)
        return recommendations.fillna('').to_dict(orient='records')

    elif filtering == 'collab':
        if artist_name_lower:
            match = filtered_data[
                (filtered_data['name'].str.lower() == song_name_lower) &
                (filtered_data['artist'].str.lower() == artist_name_lower)
            ]
        else:
            match = filtered_data[
                (filtered_data['name'].str.lower() == song_name_lower)
            ]

        if match.empty:
            raise ValueError("This song lacks user interaction data required for Collaborative Filtering.")

        current_song = match.iloc[0:1].fillna('').to_dict(orient='records')[0]
        track_id_str = str(current_song['track_id'])

        # --- OLD ON-THE-FLY LOGIC (COMMENTED OUT) ---
        # current_song_df = match.iloc[0:1]
        # selected_artist = current_song_df['artist'].values[0]
        # recommendations = collaborative_recommendation(
        #     song_name_lower, selected_artist.lower(), track_ids, filtered_data, collab_matrix, k
        # )
        # recommendations = recommendations[~(
        #     (recommendations['name'].str.lower() == current_song_df['name'].values[0].lower()) &
        #     (recommendations['artist'].str.lower() == current_song_df['artist'].values[0].lower())
        # )]
        # recommendations = pd.concat([current_song_df, recommendations]).reset_index(drop=True)
        # return recommendations.fillna('').to_dict(orient='records')

        predictions = load_offline_predictions()
        if track_id_str in predictions and 'collab' in predictions[track_id_str]:
            recs = predictions[track_id_str]['collab'][:k]
            return [current_song] + recs
        else:
            return [current_song]

    elif filtering == 'hybrid':
        if artist_name_lower:
            match = filtered_data[
                (filtered_data['name'].str.lower() == song_name_lower) &
                (filtered_data['artist'].str.lower() == artist_name_lower)
            ]
        else:
            match = filtered_data[
                (filtered_data['name'].str.lower() == song_name_lower)
            ]

        if match.empty:
            raise ValueError("This song lacks user interaction data required for the Hybrid Model.")

        current_song = match.iloc[0:1].fillna('').to_dict(orient='records')[0]
        track_id_str = str(current_song['track_id'])

        # --- OLD ON-THE-FLY LOGIC (COMMENTED OUT) ---
        # current_song_df = match.iloc[0:1]
        # selected_name = current_song_df['name'].values[0]
        # selected_artist = current_song_df['artist'].values[0]
        # recommender = hrs(
        #     song_name=selected_name, artist_name=selected_artist, number_of_recommendations=k,
        #     weight_content_based=0.3, weight_collaborative=0.7, songs_data=filtered_data,
        #     transformed_matrix=transformed_hybrid_data, track_ids=track_ids, interaction_matrix=interaction_matrix
        # )
        # recommendations = recommender.give_recommendations()
        # recommendations = recommendations[~(
        #     (recommendations['name'] == selected_name) &
        #     (recommendations['artist'] == selected_artist)
        # )]
        # recommendations = pd.concat([current_song_df, recommendations]).reset_index(drop=True)
        # return recommendations.fillna('').to_dict(orient='records')

        predictions = load_offline_predictions()
        if track_id_str in predictions and 'hybrid' in predictions[track_id_str]:
            recs = predictions[track_id_str]['hybrid'][:k]
            return [current_song] + recs
        else:
            return [current_song]

    return []
