import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_sparse):
        # A_hat @ X @ W
        x = torch.sparse.mm(adj_sparse, x)
        return self.linear(x)

class GCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = SimpleGCNLayer(embedding_dim, embedding_dim)
        self.conv2 = SimpleGCNLayer(embedding_dim, embedding_dim)

    def forward(self, adj_sparse):
        x = self.embedding.weight
        x = F.relu(self.conv1(x, adj_sparse))
        x = self.conv2(x, adj_sparse)
        return x

def gcn_score(embeddings, edge_pairs):
    src, dst = edge_pairs[0], edge_pairs[1]
    h_u = embeddings[src]
    h_v = embeddings[dst]
    return torch.sigmoid(torch.sum(h_u * h_v, dim=1))
