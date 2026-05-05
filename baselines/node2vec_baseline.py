from node2vec import Node2Vec
import networkx as nx
import torch
import numpy as np

def train_node2vec(df, dimensions=32):
    print("Training Node2Vec Baseline (Optimized)...")
    G = nx.Graph()
    
    # Build Graph
    for _, row in df.iterrows():
        G.add_edge(int(row['user1_id']), int(row['user2_id']))

    # 🔥 Optimized parameters for speed (from tip)
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=5, num_walks=20, workers=1)
    model = node2vec.fit(window=3, min_count=1, batch_words=4)

    embeddings = {}
    for node in G.nodes():
        embeddings[node] = torch.tensor(model.wv[str(node)])
        
    return embeddings


def compute_node2vec_scores(edge_index, embeddings, dimensions=32):
    scores = []
    src, dst = edge_index[0], edge_index[1]
    
    # Default embedding for missing nodes
    default_emb = torch.zeros(dimensions)
    
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        emb_u = embeddings.get(u, default_emb)
        emb_v = embeddings.get(v, default_emb)
        
        # Dot product score
        score = torch.dot(emb_u, emb_v)
        scores.append(score.item())
        
    return torch.tensor(scores, dtype=torch.float32)
