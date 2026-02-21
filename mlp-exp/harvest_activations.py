import torch
import pandas as pd
from torch.utils.data import DataLoader

def harvest_and_export(model, dataset, device="cpu", prefix="harvested_data"):
    """
    Harvests activations and saves:
    1. A .pt file containing the activation tensors.
    2. An .xlsx file containing the corresponding inputs/targets.
    """
    model.to(device)
    model.eval()
    
    # We use batch_size=1 to make mapping back to dataset metadata foolproof
    # If speed is an issue, you can increase this and flatten lists
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    all_acts = []
    
    print(f"Harvesting activations...")
    
    with torch.no_grad():
        for grids, coords_oh, targets in loader:
            # --- THE FIX: Move inputs to the same device as the model ---
            grids = grids.to(device)
            coords_oh = coords_oh.to(device)

            # Forward pass triggers the internal dictionary storage
            _ = model(grids, coords_oh)
            
            # Extract and move to CPU immediately for storage (to save GPU memory)
            acts = model.activations['layer2'].cpu()
            all_acts.append(acts)
                
    # 1. Save Tensors
    final_acts = torch.cat(all_acts, dim=0)
    pt_path = f"{prefix}.pt"
    torch.save(final_acts, pt_path)
    
    print(f"Successfully harvested {final_acts.shape[0]} samples.")
    print(f"Tensor file: {pt_path}")
    
    return pt_path

# --- Integration with your existing workflow ---
if __name__ == "__main__":

    from mlp.mlp_definition import FinalSpatialMLP
    from dataset.data_generator import OneHotSpatialDataset

    model = FinalSpatialMLP(grid_size=5, hidden_dim=256)
    model.load_state_dict(torch.load("mlp/final_spatial_model.pth"))

    train_ds = OneHotSpatialDataset(pt_path="dataset/train_data.pt")

    # Assuming model and dataset are already initialized
    pt_file = harvest_and_export(model, train_ds, device="cuda")