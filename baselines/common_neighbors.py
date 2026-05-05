import torch
from collections import defaultdict

def build_adj_list(df):
    adj = defaultdict(set)
    # Use the processed dataframe to build the graph
    for _, row in df.iterrows():
        u, v = int(row['user1_id']), int(row['user2_id'])
        adj[u].add(v)
        adj[v].add(u)
    return adj


def compute_cn_scores(edge_index, adj):
    scores = []
    src, dst = edge_index[0], edge_index[1]
    
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        # Common Neighbors Score = |N(u) ∩ N(v)|
        score = len(adj[u].intersection(adj[v]))
        scores.append(float(score))
        
    return torch.tensor(scores, dtype=torch.float32)
