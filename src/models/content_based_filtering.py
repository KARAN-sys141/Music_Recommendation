# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# from category_encoders.count import CountEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics.pairwise import cosine_similarity
# from src.data.data_cleaning import data_for_content_filtering
# from scipy.sparse import save_npz

# CLEANED_DATA_PATH = 'data/cleaned_data.csv'

# freq_cols = ['year']
# onh_cols = ['artist', 'time_signature', 'key']
# tfidf_col = 'tags'
# std_cols = ['duration_ms', 'loudness', 'tempo']
# min_max_scale_cols = [
#     'danceability', 'energy', 'speechiness', 'acousticness',
#     'instrumentalness', 'liveness', 'valence'
# ]

# def train_transformer(data):
#     transformer = ColumnTransformer(transformers=[
#         ('frequency_encode', CountEncoder(normalize=True, return_df=True), freq_cols),
#         ('ohe', OneHotEncoder(handle_unknown='ignore'), onh_cols),
#         ('tfidf', Pipeline([
#             ('tfidf_vec', TfidfVectorizer(max_features=85))
#         ]), 'tags'),
#         ('standard_scale', StandardScaler(), std_cols),
#         ('min_max_scale', MinMaxScaler(), min_max_scale_cols),
#     ],
#     remainder='passthrough',
#     n_jobs=1)

#     transformer.fit(data)
#     joblib.dump(transformer, 'models/transformer.joblib')

# def transform_data(data):
#     transformer = joblib.load('models/transformer.joblib')
#     return transformer.transform(data)

# def save_transformed_data(transformed_data, save_path):
#     save_npz(save_path, transformed_data)

# def calculate_similarity_scores(input_vector, data):
#     return cosine_similarity(input_vector, data)

# def recommend(song_name, songs_data, transformed_data, k=10):
#     song_name = song_name.lower()
#     song_row = songs_data.loc[songs_data['name'] == song_name]

#     if song_row.empty:
#         return None

#     song_index = song_row.index[0]
#     input_vector = transformed_data[song_index].reshape(1, -1)
#     similarity_scores = calculate_similarity_scores(input_vector, transformed_data)

#     top_k_songs_indexes = np.argsort(similarity_scores.ravel())[::-1]
#     top_k_songs_indexes = [i for i in top_k_songs_indexes if i != song_index][:k]

#     recommended_songs = songs_data.iloc[top_k_songs_indexes]
#     final_df = pd.concat([song_row, recommended_songs], ignore_index=True)
#     return final_df[['track_id', 'name', 'artist', 'spotify_preview_url']]

# def test_recommendation(data_path, song_name, k=10):
#     song_name = song_name.lower()
#     data = pd.read_csv(data_path)
#     data_content_filtering = data_for_content_filtering(data)

#     train_transformer(data_content_filtering)
#     transformed_data = transform_data(data_content_filtering)
#     save_transformed_data(transformed_data, 'data/transformed_data.npz')

#     song_row = data.loc[data['name'] == song_name]
#     if song_row.empty:
#         print(f"Song '{song_name}' not found.")
#         return

#     print("Matched Song:")
#     print(song_row)

#     recommendations = recommend(song_name, data, transformed_data, k)
#     print("\nTop Recommendations:")
#     print(recommendations)

# if __name__ == '__main__':
#     test_recommendation(CLEANED_DATA_PATH, "Hips Don't Lie")




import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from src.data.data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz

CLEANED_DATA_PATH = 'data/cleaned_data.csv'

freq_cols = ['year']
onh_cols = ['artist', 'time_signature', 'key']
tfidf_col = 'tags'
std_cols = ['duration_ms', 'loudness', 'tempo']
min_max_scale_cols = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence'
]

def train_transformer(data):
    transformer = ColumnTransformer(transformers=[
        ('frequency_encode', CountEncoder(normalize=True, return_df=True), freq_cols),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), onh_cols),
        ('tfidf', Pipeline([
            ('tfidf_vec', TfidfVectorizer(max_features=85))
        ]), 'tags'),
        ('standard_scale', StandardScaler(), std_cols),
        ('min_max_scale', MinMaxScaler(), min_max_scale_cols),
    ],
    remainder='passthrough',
    n_jobs=1)

    transformer.fit(data)
    joblib.dump(transformer, 'models/transformer.joblib')
    return transformer   # ✅ IMPORTANT ADD

def transform_data(data):
    transformer = train_transformer(data)   # ✅ LOAD हटाया, TRAIN use किया
    return transformer.transform(data)

def save_transformed_data(transformed_data, save_path):
    save_npz(save_path, transformed_data)

def calculate_similarity_scores(input_vector, data):
    return cosine_similarity(input_vector, data)

def recommend(song_name, songs_data, transformed_data, k=10):
    song_name = song_name.lower()
    song_row = songs_data.loc[songs_data['name'] == song_name]

    if song_row.empty:
        return None

    song_index = song_row.index[0]
    input_vector = transformed_data[song_index].reshape(1, -1)
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)

    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[::-1]
    top_k_songs_indexes = [i for i in top_k_songs_indexes if i != song_index][:k]

    recommended_songs = songs_data.iloc[top_k_songs_indexes]
    final_df = pd.concat([song_row, recommended_songs], ignore_index=True)
    return final_df[['track_id', 'name', 'artist', 'spotify_preview_url']]

def test_recommendation(data_path, song_name, k=10):
    song_name = song_name.lower()
    data = pd.read_csv(data_path)
    data_content_filtering = data_for_content_filtering(data)

    train_transformer(data_content_filtering)
    transformed_data = transform_data(data_content_filtering)
    save_transformed_data(transformed_data, 'data/transformed_data.npz')

    song_row = data.loc[data['name'] == song_name]
    if song_row.empty:
        print(f"Song '{song_name}' not found.")
        return

    print("Matched Song:")
    print(song_row)

    recommendations = recommend(song_name, data, transformed_data, k)
    print("\nTop Recommendations:")
    print(recommendations)

if __name__ == '__main__':
    test_recommendation(CLEANED_DATA_PATH, "Hips Don't Lie")