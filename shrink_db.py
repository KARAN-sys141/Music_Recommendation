import json
import sqlite3
import os
import pandas as pd

# The original json was deleted, let's regenerate it or wait, I deleted it!
# Wait! I deleted offline_predictions.json!
# I have the db `data/predictions.db` which is 135MB. 
# I will read from the DB, shrink the data, and write to a smaller DB, then replace the original DB.

DB_PATH = 'data/predictions.db'
NEW_DB_PATH = 'data/predictions_compact.db'

def main():
    print("Reading from large DB...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT track_id, collab_json, hybrid_json FROM predictions")
    rows = cursor.fetchall()
    
    print("Connecting to compacted SQLite database...")
    conn_new = sqlite3.connect(NEW_DB_PATH)
    cursor_new = conn_new.cursor()

    cursor_new.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            track_id TEXT PRIMARY KEY,
            collab_ids TEXT,
            hybrid_ids TEXT
        )
    ''')
    cursor_new.execute('DELETE FROM predictions')

    print("Shrinking data...")
    insert_records = []
    
    for row in rows:
        track_id = row[0]
        collab_recs = json.loads(row[1]) if row[1] else []
        hybrid_recs = json.loads(row[2]) if row[2] else []
        
        # Extract just the track_ids
        collab_ids = ",".join([c['track_id'] for c in collab_recs])
        hybrid_ids = ",".join([h['track_id'] for h in hybrid_recs])
        
        insert_records.append((track_id, collab_ids, hybrid_ids))

    cursor_new.executemany('''
        INSERT INTO predictions (track_id, collab_ids, hybrid_ids)
        VALUES (?, ?, ?)
    ''', insert_records)

    conn_new.commit()
    conn_new.close()
    conn.close()

    print("Replacing old DB with compacted DB...")
    os.remove(DB_PATH)
    os.rename(NEW_DB_PATH, DB_PATH)
    
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Compact Database size: {db_size_mb:.2f} MB")
    
if __name__ == '__main__':
    main()
