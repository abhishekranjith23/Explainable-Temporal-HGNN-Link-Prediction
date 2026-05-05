import torch
from collections import defaultdict
import math

def build_adj_list(df):
    adj = defaultdict(set)
    for _, row in df.iterrows():
        u, v = int(row['user1_id']), int(row['user2_id'])
        adj[u].add(v)
        adj[v].add(u)
    return adj


def common_neighbors(u, v, adj):
    return len(adj[u] & adj[v])


def adamic_adar(u, v, adj):
    common = adj[u] & adj[v]
    score = 0.0
    for z in common:
        deg = len(adj[z])
        if deg > 1:
            score += 1.0 / math.log(deg)
    return score


def compute_heuristic_scores(edge_pairs, adj, method="cn"):
    scores = []
    src, dst = edge_pairs[0], edge_pairs[1]
    
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        if method == "cn":
            scores.append(float(common_neighbors(u, v, adj)))
        elif method == "aa":
            scores.append(float(adamic_adar(u, v, adj)))
            
    return torch.tensor(scores, dtype=torch.float32)
