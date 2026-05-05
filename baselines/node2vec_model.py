from node2vec import Node2Vec
import networkx as nx
import torch

def train_node2vec(df, dimensions=64):
    print("Training Node2Vec Baseline...")
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(int(row['user1_id']), int(row['user2_id']))

    # Optimized for speed while maintaining representative walks
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=5, num_walks=20, workers=2)
    model = node2vec.fit(window=5, min_count=1)

    embeddings = {
        int(node): torch.tensor(model.wv[str(node)])
        for node in G.nodes()
    }
    return embeddings


def node2vec_scores(edge_pairs, embeddings, dimensions=64):
    scores = []
    src, dst = edge_pairs[0], edge_pairs[1]
    default_emb = torch.zeros(dimensions)
    
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        emb_u = embeddings.get(u, default_emb)
        emb_v = embeddings.get(v, default_emb)
        scores.append(torch.dot(emb_u, emb_v).item())
        
    return torch.tensor(scores, dtype=torch.float32)
