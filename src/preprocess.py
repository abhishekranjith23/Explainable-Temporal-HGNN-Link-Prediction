import pandas as pd
import numpy as np
import json
import os


def preprocess_data(file_path=r'data\soc-redditHyperlinks-title.tsv', sample_size=10000):
    print(f"Loading data from: {file_path}")
    
    # Load data (supporting both CSV and Reddit TSV)
    try:
        if file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
            # Map Reddit columns to our internal names
            df = df.rename(columns={
                'SOURCE_SUBREDDIT': 'user1',
                'TARGET_SUBREDDIT': 'user2',
                'POST_ID': 'post_id',
                'TIMESTAMP': 'timestamp',
                'LINK_SENTIMENT': 'interaction'
            })
        else:
            df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None

    # Sample for speed if dataset is huge
    if sample_size and len(df) > sample_size:
        print(f"Reducing dataset to {sample_size} rows for stability...")
        df = df.sample(n=sample_size, random_state=42)

    print("Original Data Preview:")
    print(df.head())

    # Remove missing values and duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Normalize timestamp (convert to numeric)
    df['time_numeric'] = df['timestamp'].astype(np.int64) // 10**9

    # Encode users (Subreddits)
    all_users = pd.concat([df['user1'], df['user2']]).unique()
    user_map = {user: idx for idx, user in enumerate(all_users)}

    df['user1_id'] = df['user1'].map(user_map)
    df['user2_id'] = df['user2'].map(user_map)

    # Encode interaction types (Sentiment)
    interaction_types = df['interaction'].unique()
    interaction_map = {rel: idx for idx, rel in enumerate(interaction_types)}
    df['interaction_id'] = df['interaction'].map(interaction_map)

    # Encode post_id (for hyperedges)
    post_map = {p: idx for idx, p in enumerate(df['post_id'].unique())}
    df['post_id_encoded'] = df['post_id'].map(post_map)

    # Calculate weights based on timestamp
    df['weight'] = df['time_numeric'] / df['time_numeric'].max()

    print("\nProcessed Data Preview:")
    print(df.head())

    return df, user_map, interaction_map, post_map


if __name__ == "__main__":
    file_path = r"data\soc-redditHyperlinks-title.tsv"
    df, user_map, interaction_map, post_map = preprocess_data(file_path)

    if df is not None:
        # Save processed file
        df.to_csv(r"data\processed_data.csv", index=False)
        # Save mapping for server
        os.makedirs("models", exist_ok=True)
        with open("models/user_map.json", "w") as f:
            json.dump(user_map, f)
        print("\nDone: Preprocessing complete. Saved to data/processed_data.csv and models/user_map.json")