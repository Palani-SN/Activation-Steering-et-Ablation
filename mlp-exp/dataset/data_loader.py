import pandas as pd
import torch
import ast
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Map Concepts to Integers for PyTorch ---
CONCEPT_MAP = {
    "pos_odd": 0,
    "pos_even": 1,
    "neg_odd": 2,
    "neg_even": 3
}

def load_excel_to_dataloader(filename, batch_size=32, include_concepts=True):
    df = pd.read_excel(filename)
    
    # Standard input/output conversion
    X = torch.tensor([ast.literal_eval(x) for x in df['input_list']], dtype=torch.float32)
    y = torch.tensor([ast.literal_eval(y) for y in df['output_list']], dtype=torch.float32)
    
    if include_concepts and 'concept' in df.columns:
        # Convert string labels ('pos_odd') to integers for the DataLoader
        groups = torch.tensor([CONCEPT_MAP[g] for g in df['concept']], dtype=torch.long)
        dataset = TensorDataset(X, y, groups)
    else:
        dataset = TensorDataset(X, y)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. Researcher Utility: The Activation Buffer ---
def get_grouped_activations(model, sae, dataloader, device="cpu"):
    """
    Passes data through the model and SAE, then groups the 
    resulting activations by their concept.
    """
    model.eval()
    sae.eval()
    
    # Store latents for each group
    storage = {k: [] for k in CONCEPT_MAP.keys()}
    inv_map = {v: k for k, v in CONCEPT_MAP.items()}
    
    with torch.no_grad():
        for batch_x, _, batch_groups in dataloader:
            batch_x = batch_x.to(device)
            
            # 1. Get MLP hidden activations
            hidden = model.get_activations(batch_x) 
            
            # 2. Get SAE Latent activations (features)
            # Use .encode() or whatever your SAE latent method is
            latents = sae.encode(hidden) 
            
            # 3. Sort into buckets
            for i, group_idx in enumerate(batch_groups):
                group_name = inv_map[group_idx.item()]
                storage[group_name].append(latents[i])
                
    # Stack lists into tensors
    return {k: torch.stack(v) for k, v in storage.items()}