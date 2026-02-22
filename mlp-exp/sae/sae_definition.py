import torch.nn as nn

# --- 1. The Sparse Autoencoder Architecture ---
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=512, dict_size=2048):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Hidden features are the "Monosemantic" representations
        hidden_features = self.relu(self.encoder(x))
        reconstructed = self.decoder(hidden_features)
        return reconstructed, hidden_features