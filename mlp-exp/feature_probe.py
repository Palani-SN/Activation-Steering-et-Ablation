import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sae.sae_definition import SparseAutoencoder

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

def deep_probe_neurons_and_features(mlp, sae_path, dataset, source_excel_path, k=5, device="cuda"):
    mlp.to(device).eval()
    
    # 1. Load and Index Source Excel
    print("Indexing source Excel metadata...")
    src_df = pd.read_excel(source_excel_path)
    src_df['key_grid'] = src_df['grid'].apply(clean_excel_cell)
    src_df['key_coords'] = src_df['coords_oh'].apply(clean_excel_cell)
    src_df['key_target'] = src_df['target'].apply(lambda x: format(float(x), ".3f"))
    lookup_table = src_df.set_index(['key_grid', 'key_coords', 'key_target'])

    # 2. Load SAE & Normalization Stats
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
            neuron_acts = g * c  # The 256-dimensional "bottleneck"
            
            # --- 2. Capture SAE Features ---
            norm_acts = (neuron_acts - mean) / (std + 1e-6)
            x_centered = norm_acts - sae.decoder.bias
            feature_acts = torch.relu(sae.encoder(x_centered))
            
            # --- 3. Extract Top-K for both ---
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

            # --- 5. Compile Result ---
            results.append({
                **geo_data,
                "target": target.item(),
                # Neuron Data
                "top_neurons": top_n_idx.cpu().numpy().flatten().tolist(),
                "neuron_strengths": top_n_val.cpu().numpy().flatten().tolist(),
                # Feature Data
                "top_features": top_f_idx.cpu().numpy().flatten().tolist(),
                "feature_strengths": top_f_val.cpu().numpy().flatten().tolist()
            })

            if i % 1000 == 0: print(f"Processed {i} samples...")
            # if i >= 9999: break # Adjust based on how many samples you want to analyze

    # 3. Export to Excel
    final_df = pd.DataFrame(results)
    final_df.to_excel("mechanistic_map_full.xlsx", index=False)
    print("Mapping complete. File saved: mechanistic_map_full.xlsx")


def analyze_interventions(input_excel="mechanistic_map_full.xlsx", top_n_to_identify=10):
    """
    Expands the top-K lists into rows and identifies the most frequent 
    neurons/features associated with positive vs negative targets.
    """
    print(f"Loading {input_excel} for intervention analysis...")
    df = pd.read_excel(input_excel)

    # Helper to parse stringified lists from Excel back into Python lists
    import ast
    def parse_list(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    # 1. Explode the lists: Convert 1 row with 5 elements into 5 individual rows
    # We focus on neurons and features separately
    cols_to_parse = ['top_neurons', 'neuron_strengths', 'top_features', 'feature_strengths']
    for col in cols_to_parse:
        df[col] = df[col].apply(parse_list)

    # Create long-form data for neurons
    df_neurons = df.explode(['top_neurons', 'neuron_strengths'])
    # Create long-form data for SAE features
    df_features = df.explode(['top_features', 'feature_strengths'])

    # 2. Split by Target Sentiment (Positive vs Negative results)
    pos_neurons = df_neurons[df_neurons['target'] > 0]
    neg_neurons = df_neurons[df_neurons['target'] < 0]
    
    pos_features = df_features[df_features['target'] > 0]
    neg_features = df_features[df_features['target'] < 0]

    # 3. Aggregate Top 10 unique IDs per category based on frequency and avg strength
    def get_top_intervention_targets(subset, id_col, val_col):
        return (subset.groupby(id_col)[val_col]
                .agg(['count', 'mean'])
                .sort_values(by='count', ascending=False)
                .head(top_n_to_identify))

    report = {
        "Negative_Neurons_Ablation_Targets": get_top_intervention_targets(neg_neurons, 'top_neurons', 'neuron_strengths'),
        "Positive_Neurons_Steering_Targets": get_top_intervention_targets(pos_neurons, 'top_neurons', 'neuron_strengths'),
        "Negative_Features_Ablation_Targets": get_top_intervention_targets(neg_features, 'top_features', 'feature_strengths'),
        "Positive_Features_Steering_Targets": get_top_intervention_targets(pos_features, 'top_features', 'feature_strengths')
    }

    # 4. Save results to a summary sheet
    with pd.ExcelWriter("intervention_strategy.xlsx") as writer:
        for sheet_name, data in report.items():
            data.to_excel(writer, sheet_name=sheet_name)
    
    print("Intervention analysis complete. Summary saved to: intervention_strategy.xlsx")
    
    # Print the specific IDs for the user to use in ablation
    neg_feats = report["Negative_Features_Ablation_Targets"].index.tolist()
    print(f"\n>>> RECOMMENDED SAE FEATURES TO ABLATE (to block negative values): {neg_feats}")
    return report

if __name__ == "__main__":

    from mlp.mlp_definition import FinalSpatialMLP
    from dataset.data_generator import OneHotSpatialDataset
    
    mlp = FinalSpatialMLP()
    sae_path = "sae/sae_weights.pth"
    dataset = OneHotSpatialDataset(pt_path="dataset/train_data.pt")
    source_excel_path = "dataset/train_data.xlsx"

    # Step 1: Run the original probe
    deep_probe_neurons_and_features(mlp, sae_path, dataset, source_excel_path)

    # Step 2: Run the new intervention analysis
    analyze_interventions("mechanistic_map_full.xlsx")