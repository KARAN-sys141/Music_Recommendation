import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommendatonSystem:
    def __init__(self, song_name, artist_name, number_of_recommendations, weight_content_based, weight_collaborative, songs_data, transformed_matrix, interaction_matrix, track_ids):
        self.number_of_recommendations = number_of_recommendations
        self.song_name = song_name.lower()
        self.artist_name = artist_name.lower()
        self.weight_content_based = weight_content_based
        self.weight_collaborative = weight_collaborative
        self.songs_data = songs_data.reset_index(drop=True)
        self.transformed_matrix = transformed_matrix
        self.interaction_matrix = interaction_matrix
        self.track_ids = track_ids
        self.track_id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}

    def calculate_content_based_similarities(self, song_name, artist_name, songs_data, transformed_matrix):
        song_row = songs_data.loc[
            (songs_data['name'].str.lower() == song_name) &
            (songs_data['artist'].str.lower() == artist_name)
        ]
        if song_row.empty:
            raise ValueError("Song not found for content based filtering")
        song_index = song_row.index[0]
        input_vector = transformed_matrix[song_index].toarray().reshape(1, -1)
        return cosine_similarity(input_vector, transformed_matrix)

    def calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix, batch_size=10000):
        song_row = songs_data.loc[
            (songs_data['name'].str.lower() == song_name) &
            (songs_data['artist'].str.lower() == artist_name)
        ]
        if song_row.empty:
            raise ValueError("Song not found for collaborative filtering")
        input_track_id = song_row['track_id'].values.item()
        try:
            ind = np.where(track_ids == input_track_id)[0].item()
        except IndexError:
            raise ValueError("Track ID not found")
        input_array = interaction_matrix[ind]
        if not isinstance(input_array, csr_matrix):
            input_array = csr_matrix(input_array)
        if not isinstance(interaction_matrix, csr_matrix):
            interaction_matrix = csr_matrix(interaction_matrix)
        n = interaction_matrix.shape[0]
        similarity_chunks = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = interaction_matrix[start:end]
            sim_batch = cosine_similarity(input_array, batch, dense_output=False)
            similarity_chunks.append(sim_batch.toarray())
        similarity_scores = np.hstack(similarity_chunks)
        return similarity_scores

    def normalize_similarities(self, similarity_scores):
        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        if maximum - minimum == 0:
            return similarity_scores
        return (similarity_scores - minimum) / (maximum - minimum)

    def weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        return (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)

    def give_recommendations(self):
        content_sim = self.calculate_content_based_similarities(self.song_name, self.artist_name, self.songs_data, self.transformed_matrix)
        collab_sim = self.calculate_collaborative_filtering_similarities(self.song_name, self.artist_name, self.track_ids, self.songs_data, self.interaction_matrix)
        content_track_ids = self.songs_data['track_id'].values
        collab_track_id_to_index = {track_id: idx for idx, track_id in enumerate(self.track_ids)}
        collab_indices_aligned = [collab_track_id_to_index.get(tid, None) for tid in content_track_ids]
        aligned_collab_scores = [collab_sim[0, idx] if idx is not None else 0 for idx in collab_indices_aligned]
        aligned_collab_scores = np.array(aligned_collab_scores).reshape(1, -1)
        content_sim = self.normalize_similarities(content_sim)
        aligned_collab_scores = self.normalize_similarities(aligned_collab_scores)
        weighted_scores = self.weighted_combination(content_sim, aligned_collab_scores)
        indices = np.argsort(weighted_scores.ravel())[::-1]
        input_index = self.songs_data[
            (self.songs_data['name'].str.lower() == self.song_name) &
            (self.songs_data['artist'].str.lower() == self.artist_name)
        ].index.values[0]
        indices = [i for i in indices if i != input_index][:self.number_of_recommendations]
        recommendation_track_ids = self.songs_data.iloc[indices]['track_id'].values
        score_values = weighted_scores.ravel()[indices]
        scores_df = pd.DataFrame({
            'track_id': recommendation_track_ids,
            'score': score_values
        })
        recommendations = (
            self.songs_data
            .loc[self.songs_data['track_id'].isin(recommendation_track_ids)]
            .merge(scores_df, on='track_id')
            .sort_values(by='score', ascending=False)
            .reset_index(drop=True)
        )
        return recommendations[['track_id', 'name', 'artist', 'spotify_preview_url']]

if __name__ == '__main__':
    transformed_data = load_npz('data/transformed_hybrid_data.npz')
    interaction_matrix = load_npz('data/interaction_matrix.npz')
    track_ids = np.load('data/track_ids.npy', allow_pickle=True)
    songs_data = pd.read_csv('data/collab_filtered_data.csv')
    recommender = HybridRecommendatonSystem(
        song_name="Love Story",
        artist_name="Taylor Swift",
        number_of_recommendations=10,
        weight_content_based=0.3,
        weight_collaborative=0.7,
        songs_data=songs_data,
        transformed_matrix=transformed_data,
        interaction_matrix=interaction_matrix,
        track_ids=track_ids
    )
    results = recommender.give_recommendations()
    print(results)
