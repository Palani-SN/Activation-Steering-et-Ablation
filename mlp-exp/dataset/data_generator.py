from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch

# --- 2. DATASET with One-Hot Conversion ---
class OneHotSpatialDataset(Dataset):
    def __init__(self, num_samples, grid_size=5):
        self.grid_size = grid_size
        self.data = []
        for _ in range(num_samples):
            grid = np.random.randint(0, 10, size=(grid_size, grid_size))
            coords = [random.randint(0, grid_size - 1) for _ in range(4)] # y1, x1, y2, x2
            target = float(grid[coords[0], coords[1]] - grid[coords[2], coords[3]])
            
            # One-hot encode the 4 coordinates and concatenate them
            oh_list = []
            for c in coords:
                oh = np.zeros(grid_size)
                oh[c] = 1
                oh_list.extend(oh)
            
            self.data.append({
                'grid': grid.flatten().astype(np.float32),
                'coords_oh': np.array(oh_list, dtype=np.float32),
                'target': [target]
            })

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return torch.tensor(d['grid']), torch.tensor(d['coords_oh']), torch.tensor(d['target'])
    

if __name__ == "__main__":

    import json
    dataset = OneHotSpatialDataset(10, 5)
    _len = len(dataset)
    for i in range(_len):
        print(dataset[i])