import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(HGNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, X, L):
        # L is the pre-calculated sparse Laplacian
        X_new = torch.sparse.mm(L, X)
        X_new = self.linear(X_new)
        X_new = self.norm(X_new)
        X_new = F.elu(X_new)
        return self.dropout(X_new)


class HGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, dropout=0.1):
        super(HGNN, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.layer1 = HGNNLayer(hidden_dim, hidden_dim, dropout)
        self.layer2 = HGNNLayer(hidden_dim, hidden_dim, dropout)

    def forward(self, L):
        X = self.node_embeddings.weight
        X1 = self.layer1(X, L)
        X2 = self.layer2(X1, L)
        return X1 + X2


def get_sparse_laplacian(indices, values, shape):
    """
    🔥 PURE SPARSE implementation of L = Dv^-1/2 * H * De^-1 * HT * Dv^-1/2
    No dense intermediates.
    """
    print("Pre-calculating Sparse Laplacian (Pure Sparse)...")
    num_nodes, num_edges = shape
    
    # Create H as sparse
    H = torch.sparse_coo_tensor(indices, values, shape).to(torch.float32)
    
    # 1. Node Degrees
    DV = torch.sparse.sum(H, dim=1).to_dense()
    DV_inv_sqrt_val = 1.0 / (torch.sqrt(DV) + 1e-5)
    
    # 🔥 Create Sparse Diagonal DV^-1/2 directly
    node_idx = torch.arange(num_nodes)
    DV_inv_sqrt = torch.sparse_coo_tensor(
        torch.stack([node_idx, node_idx]), 
        DV_inv_sqrt_val, 
        (num_nodes, num_nodes)
    )
    
    # 2. Edge Degrees
    DE = torch.sparse.sum(H, dim=0).to_dense()
    DE_inv_val = 1.0 / (DE + 1e-5)
    
    # 🔥 Create Sparse Diagonal DE^-1 directly
    edge_idx = torch.arange(num_edges)
    DE_inv = torch.sparse_coo_tensor(
        torch.stack([edge_idx, edge_idx]), 
        DE_inv_val, 
        (num_edges, num_edges)
    )
    
    # 3. L = Dv^-1/2 @ H @ De^-1 @ HT @ Dv^-1/2
    HT = H.t()
    
    # Use torch.sparse.mm for sparse @ sparse
    # Note: torch.sparse.mm usually expects (sparse, dense). 
    # For (sparse, sparse), we use torch.sparse.mm but result is sparse? 
    # Actually, PyTorch sparse.mm result is dense if the second arg is dense.
    # To keep it sparse, we should use torch.sparse.mm(sparse, sparse.to_dense())? No.
    # We'll use the sparse @ dense trick for the intermediate if needed, 
    # but let's try to keep the propagation as L @ X where L is sparse.
    
    # Actually, L = (DV_inv_sqrt @ H @ DE_inv @ HT @ DV_inv_sqrt)
    # This matrix is the node-node adjacency, it might be dense-ish.
    # But we can represent L as a sequence of operations in the layer!
    
    # Wait, the best way is to NOT pre-calculate L if it's too dense.
    # But for Reddit 10k, it should be fine.
    
    # Let's perform the multiplications carefully.
    # PyTorch 2.1 supports sparse @ sparse -> sparse in some cases? 
    # Actually, it's better to just store the components and do 5 multiplications in forward.
    
    # I'll return the components as a list to keep it truly sparse.
    return (DV_inv_sqrt, H, DE_inv, HT)


# Updated HGNNLayer for component-wise sparse multiplication
class HGNNLayerSparse(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(HGNNLayerSparse, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, X, components):
        DV_inv_sqrt, H, DE_inv, HT = components
        
        # X_new = Dv^-1/2 @ H @ De^-1 @ HT @ Dv^-1/2 @ X
        # All mm are (sparse @ dense) -> dense
        X_new = torch.sparse.mm(DV_inv_sqrt, X)
        X_new = torch.sparse.mm(HT, X_new)
        X_new = torch.sparse.mm(DE_inv, X_new)
        X_new = torch.sparse.mm(H, X_new)
        X_new = torch.sparse.mm(DV_inv_sqrt, X_new)
        
        X_new = self.linear(X_new)
        X_new = self.norm(X_new)
        X_new = F.elu(X_new)
        return self.dropout(X_new)

class HGNN_Optimized(nn.Module):
    def __init__(self, num_nodes, hidden_dim, dropout=0.1):
        super(HGNN_Optimized, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.layer1 = HGNNLayerSparse(hidden_dim, hidden_dim, dropout)
        self.layer2 = HGNNLayerSparse(hidden_dim, hidden_dim, dropout)

    def forward(self, components):
        X = self.node_embeddings.weight
        X1 = self.layer1(X, components)
        X2 = self.layer2(X1, components)
        return X1 + X2


if __name__ == "__main__":
    print("Testing Truly Sparse HGNN\n")
    num_nodes, num_edges, hidden_dim = 5, 3, 32
    indices = torch.tensor([[0, 1, 2, 0, 2, 3, 1, 2, 4], 
                            [0, 0, 0, 1, 1, 1, 2, 2, 2]])
    values = torch.ones(9)
    shape = (num_nodes, num_edges)
    
    components = get_sparse_laplacian(indices, values, shape)
    model = HGNN_Optimized(num_nodes, hidden_dim)
    output = model(components)
    print("Done: Node Embeddings Shape:", output.shape)