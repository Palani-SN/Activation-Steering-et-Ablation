import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import os

class OneHotSpatialDataset(Dataset):
    def __init__(self, num_samples=None, grid_size=5, pt_path=None, excel_path=None):
        self.grid_size = grid_size

        if pt_path and os.path.exists(pt_path):
            print(f"Loading binary data from {pt_path}...")
            # Load the pre-processed list of dictionaries
            self.data = torch.load(pt_path, weights_only=False)
        elif num_samples is not None:
            print(f"Generating {num_samples} samples...")
            self._generate_and_export(num_samples, excel_path, pt_path)
        else:
            raise ValueError("Provide num_samples or a valid pt_path.")

    def _generate_and_export(self, num_samples, excel_path, pt_path):
        rows_for_excel = []
        self.data = []
        
        for _ in range(num_samples):
            grid = np.random.randint(0, 10, size=(self.grid_size, self.grid_size))
            coords = [random.randint(0, self.grid_size - 1) for _ in range(4)] # y1, x1, y2, x2
            
            val1 = grid[coords[0], coords[1]]
            val2 = grid[coords[2], coords[3]]
            target = float(val1 - val2)
            
            # One-hot encode the 4 coordinates and concatenate
            oh_list = []
            for c in coords:
                oh = np.zeros(self.grid_size, dtype=np.float32)
                oh[c] = 1.0
                oh_list.extend(oh)
            
            # Store as dict for the list (torch.save handles this well)
            sample = {
                'grid': grid.flatten().astype(np.float32),
                'coords_oh': np.array(oh_list, dtype=np.float32),
                'target': np.array([target], dtype=np.float32)
            }
            self.data.append(sample)

            if excel_path:
                rows_for_excel.append({
                    **sample,
                    "y1": coords[0], "x1": coords[1], "y2": coords[2], "x2": coords[3],
                    "val1": val1, "val2": val2,
                    "target": target
                })

        # Save Binary
        if pt_path:
            torch.save(self.data, pt_path)
            print(f"Binary data saved to {pt_path}")
        
        # Save Excel (Note: .xlsx is slow for 60k rows. Consider .csv if it hangs)
        if excel_path and rows_for_excel:
            df = pd.DataFrame(rows_for_excel)
            df.to_excel(excel_path, index=False)
            print(f"Excel file saved to {excel_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert to tensor here (Non-blocking if they are already numpy arrays)
        return (
            torch.from_numpy(sample['grid']), 
            torch.from_numpy(sample['coords_oh']), 
            torch.from_numpy(sample['target'])
        )

if __name__ == "__main__":
    # Training Data
    train_ds = OneHotSpatialDataset(
        num_samples=60000, 
        grid_size=5, 
        pt_path="train_data.pt",
        excel_path="train_data.xlsx"
    )
    
    # Test Data
    test_ds = OneHotSpatialDataset(
        num_samples=5000, 
        grid_size=5, 
        pt_path="test_data.pt",
        excel_path="test_data.xlsx"
    )

    # Example DataLoader usage:
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    grids, coords, targets = next(iter(loader))
    print(f"Batch Shapes: Grid {grids.shape}, Coords {coords.shape}, Target {targets.shape}")