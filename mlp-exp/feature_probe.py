import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from sae.sae_definition import SparseAutoencoder

def set_seed(seed=42):
    """Locks random seeds to ensure consistent results across every run."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def array_to_key(arr):
    """Normalized key generator for robust Excel lookup."""
    if torch.is_tensor(arr):
        arr = arr.cpu().numpy()
    arr = arr.flatten()
    return ",".join([format(float(x), ".3f") for x in arr])

def clean_excel_cell(cell_val):
    """Parses messy stringified arrays from Excel into a clean key."""
    nums = str(cell_val).replace('[', '').replace(']', '').replace('\n', '').split()
    if not nums: return ""
    return ",".join([format(float(x), ".3f") for x in nums])

class TotalSuppressionSteering(nn.Module):
    def __init__(self, original_mlp, sae_path, dataset, ablate_ids=None, steer_ids=None, steer_multiplier=5.0, device="cuda"):
        super().__init__()
        self.mlp = original_mlp.to(device).eval()
        self.device = device
        self.ablate_ids = ablate_ids if ablate_ids else []
        self.steer_ids = steer_ids if steer_ids else [] # These are your positive drivers
        self.steer_multiplier = steer_multiplier
        
        # Load SAE and Normalization Stats
        ckpt = torch.load(sae_path, weights_only=False)
        self.encoder_w = ckpt['model_state_dict']['encoder.weight'].to(device)
        self.encoder_b = ckpt['model_state_dict']['encoder.bias'].to(device)
        self.decoder_w = ckpt['model_state_dict']['decoder.weight'].to(device)
        self.decoder_b = ckpt['model_state_dict']['decoder.bias'].to(device)
        self.mean_act = ckpt['acts_mean'].to(device)
        self.std_act = ckpt['acts_std'].to(device)

        # Pre-calculate Feature Means for stable ablation
        self.feature_means = self._compute_feature_means(dataset)

    def _compute_feature_means(self, dataset):
        print("Calculating stable feature means...")
        loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        all_f_acts = []
        with torch.no_grad():
            for grid, coords, _ in loader:
                grid, coords = grid.to(self.device), coords.to(self.device)
                acts = self.mlp.grid_net(grid) * self.mlp.coord_net(coords)
                norm_acts = (acts - self.mean_act) / (self.std_act + 1e-6)
                f_acts = torch.relu(torch.nn.functional.linear(norm_acts - self.decoder_b, self.encoder_w, self.encoder_b))
                all_f_acts.append(f_acts.mean(dim=0))
        return torch.stack(all_f_acts).mean(dim=0)

    def forward(self, grid, coords_one_hot):
        with torch.no_grad():
            # 1. Bottleneck Pass
            acts = self.mlp.grid_net(grid) * self.mlp.coord_net(coords_one_hot)
            
            # 2. SAE Projection
            norm_acts = (acts - self.mean_act) / (self.std_act + 1e-6)
            f_acts = torch.relu(torch.nn.functional.linear(norm_acts - self.decoder_b, self.encoder_w, self.encoder_b))
            
            # 3. INTERVENTION A: Mean Ablation (Neutralize Causal Negatives)
            for idx in self.ablate_ids:
                f_acts[:, idx] = self.feature_means[idx]
            
            # 4. INTERVENTION B: Additive Steering (Boost Causal Positives)
            # We add the multiplier to the positive features to shift the output bias
            for idx in self.steer_ids:
                f_acts[:, idx] += self.steer_multiplier 
            
            # 5. Reconstruction
            recon = torch.nn.functional.linear(f_acts, self.decoder_w, self.decoder_b)
            final_acts = (recon * self.std_act) + self.mean_act
            return self.mlp.output_head(final_acts)

def get_causal_drivers(mlp, sae_path, k=10, find_positive=True, device="cuda"):
    """Finds features mathematically driving output in a specific direction."""
    ckpt = torch.load(sae_path, weights_only=False)
    decoder_w = ckpt['model_state_dict']['decoder.weight'].to(device)
    
    target_layer = None
    for layer in mlp.output_head:
        if isinstance(layer, torch.nn.Linear):
            target_layer = layer
            break
    
    if target_layer is None: raise ValueError("No Linear layer found in output_head.")
    output_w = target_layer.weight.to(device)
    
    with torch.no_grad():
        # Causal Attribution: Decoder weights projected onto Output weights
        net_contributions = torch.matmul(decoder_w.t(), output_w.t()).sum(dim=1)
    
    vals, indices = torch.topk(net_contributions, k=k, largest=find_positive)
    return indices.cpu().tolist()

def deep_probe_neurons_and_features(mlp, sae_path, dataset, source_excel_path, k=5, device="cuda"):
    """Mechanistic mapping of samples to top neurons and SAE features."""
    mlp.to(device).eval()
    
    print("Indexing source Excel metadata...")
    src_df = pd.read_excel(source_excel_path)
    src_df['key_grid'] = src_df['grid'].apply(clean_excel_cell)
    src_df['key_coords'] = src_df['coords_oh'].apply(clean_excel_cell)
    src_df['key_target'] = src_df['target'].apply(lambda x: format(float(x), ".3f"))
    lookup_table = src_df.set_index(['key_grid', 'key_coords', 'key_target'])

    ckpt = torch.load(sae_path, weights_only=False)
    sae = SparseAutoencoder(ckpt['act_dim'], ckpt['dict_size']).to(device)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean, std = ckpt['acts_mean'].to(device), ckpt['acts_std'].to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []

    print("Running mechanistic mapping...")
    with torch.no_grad():
        for i, (grid, coords_oh, target) in enumerate(loader):
            grid_dev, coords_dev = grid.to(device), coords_oh.to(device)
            
            # --- 1. Capture Raw MLP Neurons ---
            g = mlp.grid_net(grid_dev)
            c = mlp.coord_net(coords_dev)
            neuron_acts = g * c
            
            # --- 2. Capture SAE Features ---
            norm_acts = (neuron_acts - mean) / (std + 1e-6)
            x_centered = norm_acts - sae.decoder.bias
            feature_acts = torch.relu(sae.encoder(x_centered))
            
            # --- 3. Extract Top-K ---
            top_n_val, top_n_idx = torch.topk(neuron_acts.abs(), k=k)
            top_f_val, top_f_idx = torch.topk(feature_acts, k=k)

            # --- 4. Metadata Lookup ---
            k_grid = array_to_key(grid)
            k_coords = array_to_key(coords_oh)
            k_target = format(float(target.item()), ".3f")

            try:
                row_meta = lookup_table.loc[(k_grid, k_coords, k_target)]
                if isinstance(row_meta, pd.DataFrame): row_meta = row_meta.iloc[0]
                geo_data = {
                    "y1": row_meta['y1'], "x1": row_meta['x1'],
                    "y2": row_meta['y2'], "x2": row_meta['x2'],
                    "val1": row_meta['val1'], "val2": row_meta['val2']
                }
            except KeyError:
                geo_data = {"y1": "NOT_FOUND"}

            results.append({
                **geo_data,
                "target": target.item(),
                "top_neurons": top_n_idx.cpu().numpy().flatten().tolist(),
                "neuron_strengths": top_n_val.cpu().numpy().flatten().tolist(),
                "top_features": top_f_idx.cpu().numpy().flatten().tolist(),
                "feature_strengths": top_f_val.cpu().numpy().flatten().tolist()
            })
            if i % 1000 == 0: print(f"Processed {i} samples...")

    final_df = pd.DataFrame(results)
    final_df.to_excel("mechanistic_map_full.xlsx", index=False)
    print("Mapping complete. File saved: mechanistic_map_full.xlsx")

def run_test(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    total, target_neg, pred_neg = 0, 0, 0
    
    with torch.no_grad():
        for grid, coords, target in loader:
            grid, coords = grid.to(model.device), coords.to(model.device)
            pred = model(grid, coords)
            target_neg += (target < 0).sum().item()
            pred_neg += (pred < 0).sum().item()
            total += target.size(0)
            
    reduction = (1 - (pred_neg / target_neg)) * 100 if target_neg > 0 else 0
    print(f"\nTarget Negatives: {target_neg}")
    print(f"Post-Mean-Ablation Negatives: {pred_neg}")
    print(f"Suppression Rate: {reduction:.2f}%")

if __name__ == "__main__":
    set_seed(42)
    from mlp.mlp_definition import FinalSpatialMLP
    from dataset.data_generator import OneHotSpatialDataset
    
    # 1. Initialize and Load Trained Weights
    mlp = FinalSpatialMLP()
    mlp.load_state_dict(torch.load("mlp/final_spatial_model.pth"))
    mlp.eval()
    
    sae_path = "sae/sae_weights.pth"
    dataset = OneHotSpatialDataset(pt_path="dataset/train_data.pt")
    source_excel_path = "dataset/train_data.xlsx"

    # 2. Run initial probe
    deep_probe_neurons_and_features(mlp, sae_path, dataset, source_excel_path)

    # 3. Identify and Print Causal Drivers
    neg_ids = get_causal_drivers(mlp, sae_path, k=10, find_positive=False)
    pos_ids = get_causal_drivers(mlp, sae_path, k=5, find_positive=True)
    print(f"\n--- CAUSAL DRIVERS FOUND ---\nNegative Drivers (for Ablation): {neg_ids}\nPositive Drivers (for Steering): {pos_ids}")

        # 3. Apply both
    surgical_model = TotalSuppressionSteering(mlp, "sae/sae_weights.pth", dataset, 
                                        ablate_ids=neg_ids, 
                                        steer_ids=pos_ids, 
                                        steer_multiplier=5.0)
    
    # Check the result
    run_test(surgical_model, dataset)