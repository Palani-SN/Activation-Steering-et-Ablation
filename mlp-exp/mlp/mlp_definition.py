import torch.nn as nn

# --- 2. The MLP Model ---
class InterpretabilityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Using a wider architecture to allow "lookup table" behavior
        self.layers = nn.ModuleDict({
            'input': nn.Linear(10, 256),
            'bn1': nn.BatchNorm1d(256),
            'hidden1': nn.Linear(256, 512), # Wider layer for SAE injection
            'bn2': nn.BatchNorm1d(512),
            'hidden2': nn.Linear(512, 256),
            'output': nn.Linear(256, 1)
        })
        self.relu = nn.ReLU()
        self.activations = {}

    def forward(self, x):
        # Layer 1
        x = self.relu(self.layers['bn1'](self.layers['input'](x)))
        
        # Layer 2: THIS is where we will inject the SAE later
        x = self.relu(self.layers['bn2'](self.layers['hidden1'](x)))
        self.activations['layer2'] = x 
        
        # Layer 3
        x = self.relu(self.layers['hidden2'](x))
        
        # Output
        x = self.layers['output'](x)
        return x