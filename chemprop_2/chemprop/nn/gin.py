# trong file chemprop/nn/gin.py
import torch
from torch import nn

class GINConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(GINConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x, edge_index):
        # edge_index là một tensor [2, num_edges] chứa các cặp nút có liên kết
        # Bước tổng hợp các nút lân cận
        row, col = edge_index
        agg_neighbors = torch.zeros_like(x)
        agg_neighbors.index_add_(0, row, x[col]) # Tổng hợp đặc trưng của các nút hàng xóm

        # Cập nhật đặc trưng nút: h_v = MLP(h_v + sum(h_u))
        x_updated = self.mlp(x + agg_neighbors)
        return x_updated

class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = in_dim
        for _ in range(n_layers):
            self.layers.append(GINConv(current_dim, hidden_dim))
            current_dim = hidden_dim

    def forward(self, graph):
        # graph là đối tượng MolGraph của Chemprop
        x, edge_index = graph.x, graph.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
        return x