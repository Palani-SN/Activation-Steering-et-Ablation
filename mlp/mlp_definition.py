import torch.nn as nn

# --- 2. The MLP Model ---


class InterpretabilityMLP(nn.Module):

    def __init__(self):

        super().__init__()

        # Using a wider architecture to allow "lookup table" behavior
        self.layers = nn.ModuleDict({
            'input': nn.Linear(10, 256),
            'bn1': nn.BatchNorm1d(256),
            'hidden1': nn.Linear(256, 512),  # Wider layer for SAE injection
            'bn2': nn.BatchNorm1d(512),
            'hidden2': nn.Linear(512, 256),
            'output': nn.Linear(256, 1)
        })
        
        self.relu = nn.ReLU()
        self.activations = {}

    def forward(self, x):

        # Layer 1
        x = self.relu(self.layers['bn1'](self.layers['input'](x)))

        # Layer 2: Save the RAW linear output for the SAE
        x = self.layers['hidden1'](x)

        # Continue the MLP pass
        x = self.relu(self.layers['bn2'](x))
        x = self.layers['hidden2'](x)

        self.activations['hidden2'] = x  # Use 'hidden2' to avoid confusion
        x = self.relu(x)

        x = self.layers['output'](x)
        return x
