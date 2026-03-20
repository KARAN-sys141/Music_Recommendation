import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

track_ids_save_path = 'data/track_ids.npy'
filtered_data_save_path = 'data/collab_filtered_data.csv'
interaction_matrix_save_path = 'data/interaction_matrix.npz'

songs_data_path = 'data/cleaned_data.csv'
user_listening_history_data_path = 'data/User Listening History.csv'

def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_path: str):
    filtered_data = songs_data[songs_data['track_id'].isin(track_ids)]
    filtered_data.reset_index(drop=True, inplace=True)
    filtered_data.to_csv(save_path, index=False)
    return filtered_data

def save_sparse_matrix(matrix: csr_matrix, file_path: str):
    save_npz(file_path, matrix)

def create_interaction_matrix(history_data: dd.DataFrame, track_ids_path: str, matrix_path: str) -> csr_matrix:
    df = history_data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)
    df = df.categorize(columns=['user_id', 'track_id'])

    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes

    track_ids = df['track_id'].cat.categories.values
    np.save(track_ids_path, track_ids, allow_pickle=True)

    df = df.assign(user_idx=user_mapping, track_idx=track_mapping)
    interaction_df = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    interaction_df = interaction_df.compute()

    row_indices = interaction_df['track_idx'].values
    col_indices = interaction_df['user_idx'].values
    values = interaction_df['playcount'].values

    n_tracks = row_indices.max() + 1
    n_users = col_indices.max() + 1

    matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    save_sparse_matrix(matrix, matrix_path)

    return matrix

def collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k=5):
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    song_row = songs_data[
        (songs_data['name'].str.lower() == song_name) &
        (songs_data['artist'].str.lower() == artist_name)
    ]

    if song_row.empty:
        raise ValueError("Song not found in the dataset")

    input_track_id = song_row['track_id'].iloc[0]
    # ind = np.where(track_ids == input_track_id)[0].item()
    inds = np.where(track_ids == input_track_id)[0]
    if len(inds) == 0:
        return pd.DataFrame()  # ya jsonify({'error': 'Track ID not found'}), 404
    ind = inds[0]

    input_array = interaction_matrix[ind]

    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    recommendation_indices = [i for i in recommendation_indices if i != ind][:k]

    recommended_ids = track_ids[recommendation_indices]
    top_scores = similarity_scores.ravel()[recommendation_indices]

    scores_df = pd.DataFrame({'track_id': recommended_ids.tolist(), 'score': top_scores})
    top_k_songs = (
        songs_data[songs_data['track_id'].isin(recommended_ids)]
        .merge(scores_df, on='track_id')
        .sort_values(by='score', ascending=False)
        .drop(columns=['score'])
        .reset_index(drop=True)
    )

    return top_k_songs

def main():
    user_data = dd.read_csv(user_listening_history_data_path)
    unique_track_ids = user_data['track_id'].unique().compute().tolist()

    songs_data = pd.read_csv(songs_data_path)
    filtered_songs = filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    interaction_matrix = create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)

    print("Ready to recommend songs")

if __name__ == '__main__':
    main()
