
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time

COLLAB_DATA_PATH          = 'data/collab_filtered_data.csv'
INTERACTION_MATRIX_PATH   = 'data/interaction_matrix.npz'
TRANSFORMED_HYBRID_PATH   = 'data/transformed_hybrid_data.npz'
TRACK_IDS_PATH            = 'data/track_ids.npy'

OUTPUT_COLLAB_PATH        = 'data/precomputed_collab.csv'
OUTPUT_HYBRID_PATH        = 'data/precomputed_hybrid.csv'

TOP_K = 20
WEIGHT_CONTENT   = 0.3
WEIGHT_COLLAB    = 0.7

print("Loading data...")
songs_data        = pd.read_csv(COLLAB_DATA_PATH)
interaction_matrix = load_npz(INTERACTION_MATRIX_PATH)
transformed_hybrid = load_npz(TRANSFORMED_HYBRID_PATH)
track_ids         = np.load(TRACK_IDS_PATH, allow_pickle=True)

songs_data['name_lower']   = songs_data['name'].str.lower().str.strip()
songs_data['artist_lower'] = songs_data['artist'].str.lower().str.strip()

track_id_to_matrix_idx = {tid: idx for idx, tid in enumerate(track_ids)}

print(f"Songs loaded: {len(songs_data)}")
print(f"Interaction matrix shape: {interaction_matrix.shape}")
print(f"Hybrid matrix shape: {transformed_hybrid.shape}")
print()

def get_collab_sim(song_matrix_idx):
    """Returns 1D similarity array over all tracks in interaction_matrix."""
    input_vec = csr_matrix(interaction_matrix[song_matrix_idx])
    sim = cosine_similarity(input_vec, interaction_matrix, dense_output=True)
    return sim.ravel()

def get_content_sim(song_df_idx):
    """Returns 1D similarity array over all rows in transformed_hybrid."""
    input_vec = transformed_hybrid[song_df_idx]
    sim = cosine_similarity(input_vec, transformed_hybrid, dense_output=True)
    return sim.ravel()

cols_to_store = ['track_id', 'name', 'artist', 'spotify_preview_url']
for col in cols_to_store:
    if col not in songs_data.columns:
        songs_data[col] = ''

print("=" * 60)
print("Precomputing COLLABORATIVE recommendations...")
print("=" * 60)

collab_rows = []
total = len(songs_data)
start = time.time()

for df_idx, row in songs_data.iterrows():
    track_id = row['track_id']
    matrix_idx = track_id_to_matrix_idx.get(track_id, None)

    if matrix_idx is None:
        continue

    sim = get_collab_sim(matrix_idx)

    top_indices = np.argsort(sim)[::-1]

    count = 0
    rank  = 1
    for candidate_matrix_idx in top_indices:
        if count >= TOP_K:
            break
        if candidate_matrix_idx == matrix_idx:
            continue

        candidate_track_id = track_ids[candidate_matrix_idx]
        candidate_rows = songs_data[songs_data['track_id'] == candidate_track_id]
        if candidate_rows.empty:
            continue

        c = candidate_rows.iloc[0]
        collab_rows.append({
            'input_track_id':        track_id,
            'input_name_lower':      row['name_lower'],
            'input_artist_lower':    row['artist_lower'],
            'rank':                  rank,
            'track_id':              c['track_id'],
            'name':                  c['name'],
            'artist':                c['artist'],
            'spotify_preview_url':   c.get('spotify_preview_url', ''),
        })
        count += 1
        rank  += 1

    if (df_idx + 1) % 500 == 0:
        elapsed = time.time() - start
        pct = (df_idx + 1) / total * 100
        print(f"  [{df_idx+1}/{total}] {pct:.1f}% — {elapsed:.0f}s elapsed")

collab_df = pd.DataFrame(collab_rows)
collab_df.to_csv(OUTPUT_COLLAB_PATH, index=False)
print(f"\n✅ Saved {len(collab_df)} collab rows → {OUTPUT_COLLAB_PATH}")
print(f"   {len(collab_df['input_track_id'].unique())} unique songs precomputed")
print()

print("=" * 60)
print("Precomputing HYBRID recommendations...")
print("=" * 60)

hybrid_rows = []
start = time.time()

content_track_ids = songs_data['track_id'].values

for df_idx, row in songs_data.iterrows():
    track_id = row['track_id']
    matrix_idx = track_id_to_matrix_idx.get(track_id, None)

    if matrix_idx is None:
        continue

    content_sim = get_content_sim(df_idx)

    full_collab_sim = get_collab_sim(matrix_idx)

    aligned_collab = np.array([
        full_collab_sim[track_id_to_matrix_idx[tid]]
        if tid in track_id_to_matrix_idx else 0.0
        for tid in content_track_ids
    ])

    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else arr

    content_sim_norm = normalize(content_sim)
    collab_sim_norm  = normalize(aligned_collab)

    weighted = WEIGHT_CONTENT * content_sim_norm + WEIGHT_COLLAB * collab_sim_norm

    top_indices = np.argsort(weighted)[::-1]

    count = 0
    rank  = 1
    for cand_df_idx in top_indices:
        if count >= TOP_K:
            break
        if cand_df_idx == df_idx:
            continue

        c = songs_data.iloc[cand_df_idx]
        hybrid_rows.append({
            'input_track_id':        track_id,
            'input_name_lower':      row['name_lower'],
            'input_artist_lower':    row['artist_lower'],
            'rank':                  rank,
            'track_id':              c['track_id'],
            'name':                  c['name'],
            'artist':                c['artist'],
            'spotify_preview_url':   c.get('spotify_preview_url', ''),
        })
        count += 1
        rank  += 1

    if (df_idx + 1) % 500 == 0:
        elapsed = time.time() - start
        pct = (df_idx + 1) / total * 100
        print(f"  [{df_idx+1}/{total}] {pct:.1f}% — {elapsed:.0f}s elapsed")

hybrid_df = pd.DataFrame(hybrid_rows)
hybrid_df.to_csv(OUTPUT_HYBRID_PATH, index=False)
print(f"\n✅ Saved {len(hybrid_df)} hybrid rows → {OUTPUT_HYBRID_PATH}")
print(f"   {len(hybrid_df['input_track_id'].unique())} unique songs precomputed")
print()
print("🎉 Done! Now commit data/precomputed_collab.csv and data/precomputed_hybrid.csv to Git and redeploy on Render.")
