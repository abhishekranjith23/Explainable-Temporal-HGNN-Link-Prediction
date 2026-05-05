import pandas as pd
import numpy as np


def build_hypergraph(file_path, use_weights=True):
    print(f"Building Hypergraph from: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None

    # Get unique nodes and edges
    nodes = sorted(set(df['user1_id']).union(set(df['user2_id'])))
    num_nodes = len(nodes)
    hyperedges = sorted(df['post_id_encoded'].unique())
    num_edges = len(hyperedges)

    # 🔥 For Sparse implementation, we just need the indices and values
    indices_u = []
    indices_e = []
    values = []

    for _, row in df.iterrows():
        u1 = int(row['user1_id'])
        u2 = int(row['user2_id'])
        e = int(row['post_id_encoded'])
        
        weight = row['weight'] if (use_weights and 'weight' in df.columns) else 1.0

        # Node 1 in Edge e
        indices_u.append(u1)
        indices_e.append(e)
        values.append(weight)

        # Node 2 in Edge e
        indices_u.append(u2)
        indices_e.append(e)
        values.append(weight)

    # Combine into a format PyTorch Sparse can use
    indices = np.array([indices_u, indices_e])
    values = np.array(values, dtype=np.float32)

    print(f"Build Completed: {num_nodes} nodes, {num_edges} edges, {len(values)} connections")
    return (indices, values, (num_nodes, num_edges)), nodes, hyperedges


if __name__ == "__main__":
    file_path = "data/processed_data.csv"
    result = build_hypergraph(file_path)
    if result is not None:
        print("Done: Sparse Hypergraph data prepared.")