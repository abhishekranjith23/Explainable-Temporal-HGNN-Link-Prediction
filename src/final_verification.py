import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score

# Baselines
from baselines.heuristics import build_adj_list, compute_heuristic_scores, adamic_adar, common_neighbors
from baselines.node2vec_model import train_node2vec, node2vec_scores
from baselines.gcn_model import GCN, gcn_score
from baselines.gcn_baseline import train_gcn
from model import HGNN_Optimized, get_sparse_laplacian
from hypergraph import build_hypergraph
from train import train_model, LinkPredictor

def generate_samples_internal(df):
    pos_edges = list(set(zip(df['user1_id'], df['user2_id'])))
    nodes = list(set(df['user1_id']).union(set(df['user2_id'])))
    neg_edges = set()
    while len(neg_edges) < len(pos_edges):
        u, v = random.choice(nodes), random.choice(nodes)
        if u != v and (u, v) not in pos_edges: neg_edges.add((u, v))
    return pos_edges, list(neg_edges)

@torch.no_grad()
def get_all_metrics(predictor_func, embeddings, pos_edge_index, eval_edges, labels, num_nodes, device="cpu"):
    scores = predictor_func(None, eval_edges, embeddings)
    auc = roc_auc_score(labels, scores.numpy())
    ks = [1, 5, 10]
    reciprocal_ranks = []
    hits = {k: 0 for k in ks}
    total = pos_edge_index.shape[1]
    src, dst = pos_edge_index[0], pos_edge_index[1]
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        neg_vs = torch.randint(0, num_nodes, (100,), device=device)
        candidates = torch.cat([torch.tensor([v], device=device), neg_vs])
        c_scores = predictor_func(u, candidates, embeddings)
        rank = (c_scores > c_scores[0]).sum().item() + 1
        reciprocal_ranks.append(1.0 / rank)
        for k in ks:
            if rank <= k: hits[k] += 1
    mrr = sum(reciprocal_ranks) / total
    hits_res = {f"Hits@{k}": hits[k] / total for k in ks}
    return {"AUC": auc, "MRR": mrr, **hits_res}

if __name__ == "__main__":
    print("CRITICAL VERIFICATION RUN STARTED...")
    os.makedirs("results", exist_ok=True)
    device = "cpu"
    df = pd.read_csv("data/processed_data.csv")
    H_result, nodes, _ = build_hypergraph("data/processed_data.csv", use_weights=True)
    num_nodes = len(nodes)
    pos_edges_raw, neg_edges_raw = generate_samples_internal(df)
    eval_edges = torch.tensor([[e[0] for e in pos_edges_raw] + [e[0] for e in neg_edges_raw],
                               [e[1] for e in pos_edges_raw] + [e[1] for e in neg_edges_raw]], dtype=torch.long)
    labels = np.concatenate([np.ones(len(pos_edges_raw)), np.zeros(len(neg_edges_raw))])
    pos_edge_index = torch.tensor([[e[0] for e in pos_edges_raw], [e[1] for e in pos_edges_raw]], dtype=torch.long)
    final_results = {}

    adj = build_adj_list(df)
    def aa_p(u, cand, dummy):
        if u is None: return compute_heuristic_scores(cand, adj, "aa")
        return torch.tensor([adamic_adar(u, int(c), adj) for c in cand])
    final_results["Adamic Adar"] = get_all_metrics(aa_p, None, pos_edge_index, eval_edges, labels, num_nodes)

    n2v_emb = train_node2vec(df)
    def n2v_p(u, cand, emb):
        if u is None: return node2vec_scores(cand, emb)
        u_emb = emb.get(u, torch.zeros(64)).unsqueeze(0)
        c_embs = torch.stack([emb.get(int(c), torch.zeros(64)) for c in cand])
        return torch.sum(u_emb * c_embs, dim=-1)
    final_results["Node2Vec"] = get_all_metrics(n2v_p, n2v_emb, pos_edge_index, eval_edges, labels, num_nodes)

    gcn_m, gcn_p, A_norm = train_gcn(df, num_nodes, epochs=40)
    h_gcn = gcn_m(A_norm).detach()
    def gcn_p_func(u, cand, h):
        if u is None: return gcn_p(h[cand[0]], h[cand[1]]).cpu()
        return torch.sum(h[u].unsqueeze(0) * h[cand], dim=-1).cpu()
    final_results["GCN"] = get_all_metrics(gcn_p_func, h_gcn, pos_edge_index, eval_edges, labels, num_nodes)

    hgnn_m, hgnn_p = train_model(H_result, df, num_nodes, epochs=50)
    indices, values, shape = H_result
    norm_c = get_sparse_laplacian(torch.tensor(indices, dtype=torch.long), torch.tensor(values, dtype=torch.float32), shape)
    h_hgnn = hgnn_m(norm_c).detach()
    def hgnn_p_func(u, cand, h):
        if u is None: return hgnn_p(h[cand[0]], h[cand[1]]).cpu()
        u_emb = h[u].unsqueeze(0).repeat(len(cand), 1)
        return hgnn_p(u_emb, h[cand]).cpu()
    final_results["Proposed HGNN"] = get_all_metrics(hgnn_p_func, h_hgnn, pos_edge_index, eval_edges, labels, num_nodes)

    report_df = pd.DataFrame(final_results).T
    report_df.to_csv("results/CONSOLIDATED_METRICS.csv")
    print("\n--- CONSOLIDATED RESEARCH METRICS ---")
    print(report_df.to_string())
