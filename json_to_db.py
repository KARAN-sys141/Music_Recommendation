import json
import sqlite3
import os

JSON_PATH = 'offline_predictions.json'
DB_PATH = 'data/predictions.db'

def main():
    print(f"Loading {JSON_PATH} into memory (this will take a few seconds)...")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {JSON_PATH} not found!")
        return

    print("Connecting to SQLite database...")
    # Connect to (or create) the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            track_id TEXT PRIMARY KEY,
            collab_json TEXT,
            hybrid_json TEXT
        )
    ''')

    # Clear existing data just in case
    cursor.execute('DELETE FROM predictions')

    print("Inserting data into database...")
    
    # Prepare data for bulk insert
    # data is expected to be dict: { "track_id_string": { "collab": [...], "hybrid": [...] } }
    insert_records = []
    for track_id, preds in data.items():
        collab_str = json.dumps(preds.get('collab', []))
        hybrid_str = json.dumps(preds.get('hybrid', []))
        insert_records.append((track_id, collab_str, hybrid_str))

    cursor.executemany('''
        INSERT INTO predictions (track_id, collab_json, hybrid_json)
        VALUES (?, ?, ?)
    ''', insert_records)

    conn.commit()
    conn.close()

    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Successfully converted! Database created at {DB_PATH}")
    print(f"Database size: {db_size_mb:.2f} MB")
    
    # Delete the large JSON file to save space and prevent accidental commits
    print(f"Removing {JSON_PATH} to save space...")
    try:
        os.remove(JSON_PATH)
        print("Done!")
    except Exception as e:
        print(f"Could not remove {JSON_PATH}: {e}")

if __name__ == '__main__':
    main()
