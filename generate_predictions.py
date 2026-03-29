import json
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
OUTPUT_JSON_PATH = 'offline_predictions.json'

K = 10
WEIGHT_CONTENT = 0.3
WEIGHT_COLLAB = 0.7

def normalize_similarities_2d(sims):
    mins = np.min(sims, axis=1, keepdims=True)
    maxs = np.max(sims, axis=1, keepdims=True)
    diffs = maxs - mins
    diffs[diffs == 0] = 1.0
    return (sims - mins) / diffs

def main():
    logging.info("Starting offline prediction generation (Optimized)...")
    
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
    
    # Pre-align matrix indices to songs_data rows
    songs_m_indices = np.array([track_id_to_matrix_idx.get(tid, -1) for tid in songs_data['track_id']])
    
    offline_predictions = {}
    
    # We will compute in batches
    BATCH_SIZE = 1000
    
    # To quickly access song fields
    songs_dicts = songs_data.fillna('').to_dict(orient='records')
    
    for start_idx in range(0, total_songs, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_songs)
        
        # 1. Content similarities for batch
        # shape: (batch_size, total_songs)
        content_sim_batch = cosine_similarity(transformed_hybrid_data[start_idx:end_idx], transformed_hybrid_data, dense_output=True)
        content_sim_norm = normalize_similarities_2d(content_sim_batch)
        
        # 2. Collab similarities for batch
        batch_m_indices = songs_m_indices[start_idx:end_idx]
        
        # we need to compute cosine similarity of interaction_matrix[batch_m_indices] with interaction_matrix
        # But some batch_m_indices might be -1
        valid_mask = batch_m_indices != -1
        
        # initialize empty collab sims
        collab_sim_norm = np.zeros((end_idx - start_idx, total_songs))
        
        if np.any(valid_mask):
            valid_batch_m_indices = batch_m_indices[valid_mask]
            # shape: (num_valid, total_matrix_rows)
            valid_collab_sim = cosine_similarity(interaction_matrix[valid_batch_m_indices], interaction_matrix, dense_output=True)
            
            # aligned to total_songs (df indices)
            # wait, valid_collab_sim has columns=total_matrix_rows. 
            # We need to pick only the columns that correspond to songs_data (total_songs)
            # that is: songs_m_indices (ignoring -1)
            # Pre-filter columns:
            aligned_valid_collab = np.zeros((len(valid_batch_m_indices), total_songs))
            
            # Map valid items
            global_valid_df_mask = songs_m_indices != -1
            global_valid_m_indices = songs_m_indices[global_valid_df_mask]
            
            aligned_valid_collab[:, global_valid_df_mask] = valid_collab_sim[:, global_valid_m_indices]
            
            # Normalize
            aligned_valid_collab_norm = normalize_similarities_2d(aligned_valid_collab)
            
            collab_sim_norm[valid_mask] = aligned_valid_collab_norm
            
            # We also need pure collab tops mapping for the valid elements
            # collab_sim_norm is shape (num_valid, total_songs).
            # Wait, the user's pure collab logic expects predictions from songs_data! Yes.
            
        weighted_scores = (WEIGHT_CONTENT * content_sim_norm) + (WEIGHT_COLLAB * collab_sim_norm)
        
        # Process outcomes for each item in batch
        for i, df_idx in enumerate(range(start_idx, end_idx)):
            input_track_id_str = songs_dicts[df_idx]['track_id_str']
            
            # Collab predictions (only if valid)
            collab_recs = []
            if batch_m_indices[i] != -1:
                # Top K collab
                c_scores = collab_sim_norm[i]
                c_indices = np.argsort(c_scores)[-(K+1):][::-1]
                c_indices = [idx for idx in c_indices if idx != df_idx][:K]
                
                for idx in c_indices:
                    rec_data = songs_dicts[idx]
                    collab_recs.append({
                        'track_id': str(rec_data['track_id']),
                        'name': str(rec_data['name']),
                        'artist': str(rec_data['artist']),
                        'spotify_preview_url': str(rec_data['spotify_preview_url'])
                    })
                    
            # Hybrid predictions
            h_scores = weighted_scores[i]
            h_indices = np.argsort(h_scores)[-(K+1):][::-1]
            h_indices = [idx for idx in h_indices if idx != df_idx][:K]
            
            hybrid_recs = []
            for idx in h_indices:
                rec_data = songs_dicts[idx]
                hybrid_recs.append({
                    'track_id': str(rec_data['track_id']),
                    'name': str(rec_data['name']),
                    'artist': str(rec_data['artist']),
                    'spotify_preview_url': str(rec_data['spotify_preview_url'])
                })
                
            offline_predictions[input_track_id_str] = {
                'collab': collab_recs,
                'hybrid': hybrid_recs
            }
            
        logging.info(f"Processed batch {start_idx}-{end_idx}/{total_songs}")

    logging.info(f"Total processing time: {time.time() - start_time:.2f}s")
    
    logging.info(f"Saving to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(offline_predictions, f)
    
    logging.info("Offline predictions generated successfully!")

if __name__ == '__main__':
    main()
