import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, act_dim=256, dict_size=1024):
        super().__init__()
        self.act_dim = act_dim
        self.dict_size = dict_size

        # Encoder: Linear + ReLU
        self.encoder = nn.Linear(act_dim, dict_size)
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        
        # Decoder: Linear (Weight shape: [act_dim, dict_size])
        self.decoder = nn.Linear(dict_size, act_dim, bias=True)
        self.make_decoder_weights_unit_norm()

    def forward(self, x, l1_coeff=1e-4):
        # Subtract decoder bias (standard SAE practice)
        x_centered = x - self.decoder.bias
        
        # Encode
        features = F.relu(self.encoder(x_centered))
        
        # Decode
        reconstruction = self.decoder(features)
        
        # Loss components
        mse_loss = F.mse_loss(reconstruction, x)
        l1_loss = l1_coeff * features.abs().sum(dim=-1).mean()
        
        return reconstruction, features, mse_loss, l1_loss

    @torch.no_grad()
    def make_decoder_weights_unit_norm(self):
        W = self.decoder.weight.data # [act_dim, dict_size]
        norms = torch.norm(W, dim=0, keepdim=True)
        W.div_(norms + 1e-8)