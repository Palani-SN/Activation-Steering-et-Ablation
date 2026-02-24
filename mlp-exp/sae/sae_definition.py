import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=256, dict_size=2048, k=128):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.k = k  # Number of features to keep active

    def forward(self, x):
        # 1. Linear Projection
        pre_act = self.encoder(x)

        # 2. Top-K Sparsification (Modern Researcher Method)
        # We only keep the top K activations, setting others to 0
        topk_values, topk_indices = torch.topk(pre_act, self.k, dim=-1)
        hidden_features = torch.zeros_like(pre_act)
        hidden_features.scatter_(-1, topk_indices, topk_values)
        hidden_features = torch.relu(hidden_features)

        # 3. Reconstruction
        reconstructed = self.decoder(hidden_features)
        return reconstructed, hidden_features
