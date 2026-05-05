import pandas as pd


def temporal_split(file_path, num_splits=3):
    print("Loading processed data...")

    df = pd.read_csv(file_path)

    print("\nData Preview:")
    print(df.head())

    # Sort by time
    df = df.sort_values(by='time_numeric')

    # Add Time Decay Weight
    df['weight'] = df['time_numeric'] / df['time_numeric'].max()

    print("\nData with Time Weights:")
    print(df[['user1_id', 'user2_id', 'time_numeric', 'weight']].head())

    # Split into time windows
    splits = []
    chunk_size = len(df) // num_splits

    print(f"\nTotal records: {len(df)}")
    print(f"Chunk size: {chunk_size}")

    for i in range(num_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_splits - 1 else len(df)

        split_df = df.iloc[start:end]
        splits.append(split_df)

        print(f"\nTime Snapshot {i+1}:")
        print(split_df[['user1_id', 'user2_id', 'time_numeric', 'weight']])

    return splits


if __name__ == "__main__":
    print("Temporal Modeling Started\n")

    file_path = "data/processed_data.csv"

    splits = temporal_split(file_path)

    print("\nDone: Temporal modeling with time decay complete!")