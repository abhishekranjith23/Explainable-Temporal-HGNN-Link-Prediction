import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Use non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')

from model import HGNN_Optimized, get_sparse_laplacian
from hypergraph import build_hypergraph


def generate_samples(df):
    pos_edges = list(set(zip(df['user1_id'], df['user2_id'])))
    nodes = list(set(df['user1_id']).union(set(df['user2_id'])))
    neg_edges = set()
    while len(neg_edges) < len(pos_edges):
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and (u, v) not in pos_edges:
            neg_edges.add((u, v))
    return pos_edges, list(neg_edges)


class LinkPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
    def forward(self, h_u, h_v):
        x = torch.cat([h_u, h_v], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze()


def plot_loss(loss_history):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='royalblue', linewidth=2, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HGNN Training Convergence (Reddit Dataset)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("plots/loss_curve.png")
    print("Graph Saved: plots/loss_curve.png")
    plt.close()


def train_model(H_data, df, num_nodes, epochs=60):
    print("Starting Optimized Sparse Training...\n")
    indices, values, shape = H_data
    indices_torch = torch.tensor(indices, dtype=torch.long)
    values_torch = torch.tensor(values, dtype=torch.float32)

    L_components = get_sparse_laplacian(indices_torch, values_torch, shape)

    hidden_dim = 32
    model = HGNN_Optimized(num_nodes, hidden_dim=hidden_dim)
    predictor = LinkPredictor(hidden_dim)
    
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.005)
    loss_fn = nn.BCELoss()

    pos_edges, neg_edges = generate_samples(df)
    pos_u = torch.tensor([e[0] for e in pos_edges], dtype=torch.long)
    pos_v = torch.tensor([e[1] for e in pos_edges], dtype=torch.long)
    neg_u = torch.tensor([e[0] for e in neg_edges], dtype=torch.long)
    neg_v = torch.tensor([e[1] for e in neg_edges], dtype=torch.long)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        predictor.train()
        
        embeddings = model(L_components)
        pos_scores = predictor(embeddings[pos_u], embeddings[pos_v])
        neg_scores = predictor(embeddings[neg_u], embeddings[neg_v])
        
        all_scores = torch.cat([pos_scores, neg_scores])
        all_targets = torch.cat([torch.ones(len(pos_u)), torch.zeros(len(neg_u))])

        loss = loss_fn(all_scores, all_targets)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Plot final loss curve
    plot_loss(loss_history)

    return model, predictor


if __name__ == "__main__":
    print("Training Started with Visualization\n")
    df = pd.read_csv("data/processed_data.csv")
    H_result, nodes, _ = build_hypergraph("data/processed_data.csv", use_weights=True)
    num_nodes = len(nodes)
    model, predictor = train_model(H_result, df, num_nodes)
    print("\nDone: Training and Plotting complete!")