
import torch.nn as nn

# --- 1. MODEL with One-Hot Input ---
class FinalSpatialMLP(nn.Module):
    def __init__(self, grid_size=5, hidden_dim=256):
        super().__init__()
        # 4 coordinates * 5 possible values = 20 inputs for coords
        self.coord_dim = grid_size * 4 
        
        self.grid_net = nn.Sequential(
            nn.Linear(grid_size * grid_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.coord_net = nn.Sequential(
            nn.Linear(self.coord_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.activations = {}

    def forward(self, grid, coords_one_hot, return_activations=False):

        g = self.grid_net(grid)
        c = self.coord_net(coords_one_hot)
        
        # This is the "interconnected" representation you want to probe
        acts = g * c 
        self.activations['layer2'] = acts  # Store for later harvesting
        out = self.output_head(acts)
        
        if return_activations:
            return out, acts
        return out