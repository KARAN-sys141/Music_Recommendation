import sqlite3
import logging
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRACK_IDS_PATH = 'data/track_ids.npy'
COLLAB_DATA_PATH = 'data/collab_filtered_data.csv'
INTERACTION_MATRIX_PATH = 'data/interaction_matrix.npz'
TRANSFORMED_HYBRID_PATH = 'data/transformed_hybrid_data.npz'
DB_PATH = 'data/predictions.db'

K = 10
WEIGHT_CONTENT = 0.75
WEIGHT_COLLAB = 0.25

def normalize_similarities_2d(sims):
    mins = np.min(sims, axis=1, keepdims=True)
    maxs = np.max(sims, axis=1, keepdims=True)
    diffs = maxs - mins
    diffs[diffs == 0] = 1.0
    return (sims - mins) / diffs

def main():
    logging.info("Starting database prediction generation (Optimized)...")
    
    start_time = time.time()
    
    track_ids = np.load(TRACK_IDS_PATH, allow_pickle=True)
    songs_data = pd.read_csv(COLLAB_DATA_PATH)
    songs_data['track_id_str'] = songs_data['track_id'].astype(str)
    
    interaction_matrix = load_npz(INTERACTION_MATRIX_PATH)
    if not isinstance(interaction_matrix, csr_matrix):
        interaction_matrix = csr_matrix(interaction_matrix)
        
    transformed_hybrid_data = load_npz(TRANSFORMED_HYBRID_PATH)

    total_songs = len(songs_data)
    logging.info(f"Loaded {total_songs} songs.")

    track_id_to_matrix_idx = {tid: idx for idx, tid in enumerate(track_ids)}
    songs_m_indices = np.array([track_id_to_matrix_idx.get(tid, -1) for tid in songs_data['track_id']])
    
    # Initialize SQLite DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            track_id TEXT PRIMARY KEY,
            collab_ids TEXT,
            hybrid_ids TEXT
        )
    ''')
    cursor.execute('DELETE FROM predictions')
    
    BATCH_SIZE = 1000
    songs_dicts = songs_data.fillna('').to_dict(orient='records')
    
    for start_idx in range(0, total_songs, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_songs)
        
        content_sim_batch = cosine_similarity(transformed_hybrid_data[start_idx:end_idx], transformed_hybrid_data, dense_output=True)
        content_sim_norm = normalize_similarities_2d(content_sim_batch)
        
        batch_m_indices = songs_m_indices[start_idx:end_idx]
        valid_mask = batch_m_indices != -1
        collab_sim_norm = np.zeros((end_idx - start_idx, total_songs))
        
        if np.any(valid_mask):
            valid_batch_m_indices = batch_m_indices[valid_mask]
            valid_collab_sim = cosine_similarity(interaction_matrix[valid_batch_m_indices], interaction_matrix, dense_output=True)
            aligned_valid_collab = np.zeros((len(valid_batch_m_indices), total_songs))
            
            global_valid_df_mask = songs_m_indices != -1
            global_valid_m_indices = songs_m_indices[global_valid_df_mask]
            
            aligned_valid_collab[:, global_valid_df_mask] = valid_collab_sim[:, global_valid_m_indices]
            aligned_valid_collab_norm = normalize_similarities_2d(aligned_valid_collab)
            collab_sim_norm[valid_mask] = aligned_valid_collab_norm
            
        weighted_scores = (WEIGHT_CONTENT * content_sim_norm) + (WEIGHT_COLLAB * collab_sim_norm)
        
        insert_records = []
        for i, df_idx in enumerate(range(start_idx, end_idx)):
            input_track_id_str = songs_dicts[df_idx]['track_id_str']
            
            # Collab predictions
            collab_ids = []
            if batch_m_indices[i] != -1:
                c_scores = collab_sim_norm[i]
                c_indices = np.argsort(c_scores)[-(K+1):][::-1]
                c_indices = [idx for idx in c_indices if idx != df_idx][:K]
                collab_ids = [str(songs_dicts[idx]['track_id']) for idx in c_indices]
                    
            # Hybrid predictions
            h_scores = weighted_scores[i]
            h_indices = np.argsort(h_scores)[-(K+1):][::-1]
            h_indices = [idx for idx in h_indices if idx != df_idx][:K]
            hybrid_ids = [str(songs_dicts[idx]['track_id']) for idx in h_indices]

            insert_records.append((
                input_track_id_str,
                ",".join(collab_ids),
                ",".join(hybrid_ids)
            ))
            
        cursor.executemany('''
            INSERT INTO predictions (track_id, collab_ids, hybrid_ids)
            VALUES (?, ?, ?)
        ''', insert_records)
        
        logging.info(f"Processed and inserted batch {start_idx}-{end_idx}/{total_songs}")

    conn.commit()
    conn.close()
    
    logging.info(f"Total processing time: {time.time() - start_time:.2f}s")
    logging.info("Database generation successfully completed! (No JSON file used)")

if __name__ == '__main__':
    main()
