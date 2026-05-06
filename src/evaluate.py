import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

from model import HGNN_Optimized, get_sparse_laplacian
from hypergraph import build_hypergraph
from train import LinkPredictor, generate_samples, train_model

# Import Baselines
from baselines.heuristics import build_adj_list, compute_heuristic_scores
from baselines.node2vec_model import train_node2vec, node2vec_scores
from baselines.gcn_model import GCN, gcn_score


# --- 📊 VISUALIZATION HELPERS ---

def plot_roc(labels, scores, model_name):
    os.makedirs("plots", exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_score(labels, scores):.2f})")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"plots/roc_{model_name.lower().replace(' ', '_')}.png")
    plt.close()


def plot_model_comparison(results):
    os.makedirs("plots", exist_ok=True)
    models = list(results.keys())
    aucs = [results[m]["AUC"] for m in models]
    
    plt.figure(figsize=(10, 6))
    colors = ['gray', 'orange', 'blue', 'green', 'red'][:len(models)]
    plt.bar(models, aucs, color=colors, alpha=0.7)
    plt.title("Model Comparison (AUC Score)")
    plt.ylabel("AUC")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=30)
    for i, v in enumerate(aucs):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig("plots/model_comparison_auc.png")
    plt.close()


# --- ⚙️ EVALUATION LOGIC ---

@torch.no_grad()
def compute_ranking_metrics(predictor_func, embeddings, edge_index, num_nodes, ks=[1, 5, 10], num_neg=100, device="cpu"):
    src, dst = edge_index[0], edge_index[1]
    reciprocal_ranks = []
    hits = {k: 0 for k in ks}
    total = len(src)
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        neg_vs = torch.randint(0, num_nodes, (num_neg,), device=device)
        candidates = torch.cat([torch.tensor([v], device=device), neg_vs])
        scores = predictor_func(u, candidates, embeddings)
        rank = (scores > scores[0]).sum().item() + 1
        reciprocal_ranks.append(1.0 / rank)
        for k in ks:
            if rank <= k:
                hits[k] += 1
    return sum(reciprocal_ranks) / total, {k: hits[k] / total for k in ks}


if __name__ == "__main__":
    print("Full Production Evaluation Tournament (Updated Baselines)\n")
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    df = pd.read_csv("data/processed_data.csv")
    H_result, nodes, _ = build_hypergraph("data/processed_data.csv", use_weights=True)
    num_nodes = len(nodes)
    pos_edges_raw, neg_edges_raw = generate_samples(df)
    
    eval_edges = torch.tensor([[e[0] for e in pos_edges_raw] + [e[0] for e in neg_edges_raw],
                               [e[1] for e in pos_edges_raw] + [e[1] for e in neg_edges_raw]], dtype=torch.long)
    labels = np.concatenate([np.ones(len(pos_edges_raw)), np.zeros(len(neg_edges_raw))])
    pos_edge_index = torch.tensor([[e[0] for e in pos_edges_raw], [e[1] for e in pos_edges_raw]], dtype=torch.long)

    comparison_results = {}

    # --- Baseline 1: Adamic-Adar ---
    print("\n[1/5] Running Adamic-Adar (Heuristic)...")
    adj = build_adj_list(df)
    aa_scores = compute_heuristic_scores(eval_edges, adj, "aa")
    def aa_pred(u, cand, dummy):
        from baselines.heuristics import adamic_adar
        return torch.tensor([adamic_adar(u, int(c), adj) for c in cand])
    mrr_aa, hits_aa = compute_ranking_metrics(aa_pred, None, pos_edge_index, num_nodes)
    comparison_results["Adamic Adar"] = {"AUC": roc_auc_score(labels, aa_scores.numpy()), "MRR": mrr_aa, "Hits@10": hits_aa[10]}
    print(f"Adamic Adar AUC: {comparison_results['Adamic Adar']['AUC']:.4f}")

    # --- Baseline 2: Common Neighbors ---
    print("\n[2/5] Running Common Neighbors (Heuristic)...")
    cn_scores = compute_heuristic_scores(eval_edges, adj, "cn")
    def cn_pred(u, cand, dummy):
        from baselines.heuristics import common_neighbors
        return torch.tensor([common_neighbors(u, int(c), adj) for c in cand])
    mrr_cn, hits_cn = compute_ranking_metrics(cn_pred, None, pos_edge_index, num_nodes)
    comparison_results["Common Neighbors"] = {"AUC": roc_auc_score(labels, cn_scores.numpy()), "MRR": mrr_cn, "Hits@10": hits_cn[10]}
    print(f"Common Neighbors AUC: {comparison_results['Common Neighbors']['AUC']:.4f}")

    # --- Baseline 3: Node2Vec ---
    print("\n[3/5] Running Node2Vec...")
    n2v_emb = train_node2vec(df)
    n2v_s = node2vec_scores(eval_edges, n2v_emb)
    def n2v_pred(u, cand, emb):
        u_emb = emb.get(u, torch.zeros(64)).unsqueeze(0)
        c_embs = torch.stack([emb.get(int(c), torch.zeros(64)) for c in cand])
        return torch.sum(u_emb * c_embs, dim=-1)
    mrr_n2v, hits_n2v = compute_ranking_metrics(n2v_pred, n2v_emb, pos_edge_index, num_nodes)
    comparison_results["Node2Vec"] = {"AUC": roc_auc_score(labels, n2v_s.numpy()), "MRR": mrr_n2v, "Hits@10": hits_n2v[10]}
    print(f"Node2Vec AUC: {comparison_results['Node2Vec']['AUC']:.4f}")

    # --- Baseline 4: GCN ---
    print("\n[4/5] Running GCN...")
    u_list, v_list = df['user1_id'].tolist(), df['user2_id'].tolist()
    edge_idx_gcn = torch.tensor([u_list + v_list + list(range(num_nodes)), v_list + u_list + list(range(num_nodes))], dtype=torch.long)
    A = torch.sparse_coo_tensor(edge_idx_gcn, torch.ones(edge_idx_gcn.shape[1]), (num_nodes, num_nodes)).to(torch.float32)
    D = torch.sparse.sum(A, dim=1).to_dense()
    A_norm = torch.sparse.mm(torch.diag(1.0 / (D + 1e-5)).to_sparse(), A)
    
    gcn_m = GCN(num_nodes)
    opt_gcn = torch.optim.Adam(gcn_m.parameters(), lr=0.01)
    for _ in range(20):
        gcn_m.train(); opt_gcn.zero_grad()
        h = gcn_m(A_norm)
        scores = gcn_score(h, eval_edges)
        loss = ((scores - torch.tensor(labels, dtype=torch.float32))**2).mean()
        loss.backward(); opt_gcn.step()
    
    gcn_m.eval()
    with torch.no_grad():
        h_gcn = gcn_m(A_norm)
        gcn_s = gcn_score(h_gcn, eval_edges)
        def gcn_p(u, cand, h):
            return torch.sum(h[u].unsqueeze(0) * h[cand], dim=-1)
        mrr_gcn, hits_gcn = compute_ranking_metrics(gcn_p, h_gcn, pos_edge_index, num_nodes)
    comparison_results["GCN"] = {"AUC": roc_auc_score(labels, gcn_s.numpy()), "MRR": mrr_gcn, "Hits@10": hits_gcn[10]}
    print(f"GCN AUC: {comparison_results['GCN']['AUC']:.4f}")

    # --- Baseline 5: HGNN (Our Model) ---
    print("\n[5/5] Running Proposed HGNN Model...")
    hgnn_m, hgnn_p = train_model(H_result, df, num_nodes, epochs=40)
    indices, values, shape = H_result
    norm_c = get_sparse_laplacian(torch.tensor(indices, dtype=torch.long).to(device), 
                                 torch.tensor(values, dtype=torch.float32).to(device), shape)
    hgnn_m.eval(); hgnn_p.eval()
    with torch.no_grad():
        h_hgnn = hgnn_m(norm_c)
        hgnn_s = hgnn_p(h_hgnn[eval_edges[0]], h_hgnn[eval_edges[1]]).cpu()
        def hgnn_p_func(u, cand, h):
            u_emb = h[u].unsqueeze(0).repeat(len(cand), 1)
            return hgnn_p(u_emb, h[cand]).cpu()
        mrr_hgnn, hits_hgnn = compute_ranking_metrics(hgnn_p_func, h_hgnn, pos_edge_index, num_nodes)
    comparison_results["Proposed HGNN"] = {"AUC": roc_auc_score(labels, hgnn_s.numpy()), "MRR": mrr_hgnn, "Hits@10": hits_hgnn[10]}
    print(f"HGNN AUC: {comparison_results['Proposed HGNN']['AUC']:.4f}")

    # --- 📊 FINAL VISUALIZATION & TABLE ---
    print("\n--- Final Performance Summary ---")
    plot_model_comparison(comparison_results)
    
    res_df = pd.DataFrame(comparison_results).T.reset_index().rename(columns={'index': 'Model'})
    res_df.to_csv("results/comparison_table.csv", index=False)
    
    # Export for Frontend
    with open("results/data.json", "w") as f:
        json.dump(comparison_results, f, indent=4)
    
    print("\n" + res_df.to_string(index=False))
    print("\nVisualizations saved to 'plots/' directory and data.json updated.")
