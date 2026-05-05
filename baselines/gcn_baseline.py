import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_sparse):
        # x: [N, in_dim], adj_sparse: [N, N] sparse tensor
        # Standard GCN: A_hat @ X @ W
        x = torch.sparse.mm(adj_sparse, x)
        return self.linear(x)

class SimpleGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super(SimpleGCN, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.layer1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.layer2 = SimpleGCNLayer(hidden_dim, hidden_dim)

    def forward(self, adj_sparse):
        x = self.embeddings.weight
        x = self.layer1(x, adj_sparse)
        x = F.relu(x)
        x = self.layer2(x, adj_sparse)
        return x

class GCNPredictor(nn.Module):
    def forward(self, h_u, h_v):
        # Dot product for link scoring
        return torch.sigmoid(torch.sum(h_u * h_v, dim=-1))

def train_gcn(df, num_nodes, epochs=40):
    print("Training Simple GCN Baseline (Pure PyTorch)...")
    hidden_dim = 32
    
    # 1. Build Symmetric Adjacency Matrix (A_hat)
    u_list = df['user1_id'].tolist()
    v_list = df['user2_id'].tolist()
    
    # Simple edge list for adjacency
    edge_idx = torch.tensor([u_list + v_list + list(range(num_nodes)), 
                             v_list + u_list + list(range(num_nodes))], dtype=torch.long)
    vals = torch.ones(edge_idx.shape[1])
    
    # Sparse A
    A = torch.sparse_coo_tensor(edge_idx, vals, (num_nodes, num_nodes)).to(torch.float32)
    
    # Normalize A (D^-1 @ A) - Simplified for baseline
    D = torch.sparse.sum(A, dim=1).to_dense()
    D_inv = torch.diag(1.0 / (D + 1e-5)).to_sparse()
    A_norm = torch.sparse.mm(D_inv, A)

    model = SimpleGCN(num_nodes, hidden_dim)
    predictor = GCNPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        h = model(A_norm)
        
        # Sample subset for training speed
        pos_u, pos_v = torch.tensor(u_list[:1000]), torch.tensor(v_list[:1000])
        neg_u = torch.randint(0, num_nodes, (1000,))
        neg_v = torch.randint(0, num_nodes, (1000,))
        
        pos_scores = predictor(h[pos_u], h[pos_v])
        neg_scores = predictor(h[neg_u], h[neg_v])
        
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + \
               F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
               
        loss.backward()
        optimizer.step()
        
    return model, predictor, A_norm
