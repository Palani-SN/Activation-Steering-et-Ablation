import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# --- 1. IMPORTS & INSTANTIATION ---
# These imports rely on your provided file structure
from mlp.mlp_definition import InterpretabilityMLP
from sae.sae_definition import SparseAutoencoder

def load_trained_models(mlp_path="mlp/perfect_mlp.pth", sae_path="sae/universal_sae.pth"):
    """Explicitly instantiates classes and loads weights."""
    # Instantiate MLP based on architecture
    mlp = InterpretabilityMLP()
    mlp_state = torch.load(mlp_path, map_location='cpu')
    # Standard load_state_dict handling
    mlp.load_state_dict(mlp_state if 'layers.output.weight' in mlp_state else mlp_state.get('model_state_dict', mlp_state))
    mlp.eval()

    # Instantiate SAE based on architecture
    # Defaulting to dict_size=2048 as per Phase I context
    sae = SparseAutoencoder(input_dim=512, dict_size=2048) 
    sae_state = torch.load(sae_path, map_location='cpu')
    sae.load_state_dict(sae_state if 'decoder.weight' in sae_state else sae_state.get('sae_state_dict', sae_state))
    sae.eval()
    
    return mlp, sae

# --- 2. THE CONCEPT COMPASS (Geometry) ---

def plot_steering_basis_compass(file_path="steering_basis.pt"):
    """Visualizes the geometric relationship (orthogonality) of logic vectors."""
    if not os.path.exists(file_path):
        print(f"Skipping Compass: {file_path} not found.")
        return

    data = torch.load(file_path, map_location='cpu')
    v_sign, v_parity = data["v_sign"], data["v_parity"]

    # Define the 4 cardinal logic directions
    vectors = torch.stack([v_sign, -v_sign, v_parity, -v_parity]).cpu().numpy()
    labels = ["Positive (+Sign)", "Negative (-Sign)", "Odd (+Parity)", "Even (-Parity)"]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]

    # Project to 2D for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 10), facecolor='white')
    for i in range(len(labels)):
        plt.quiver(0, 0, coords[i, 0], coords[i, 1], angles='xy', scale_units='xy', scale=1,
                   color=colors[i], label=labels[i], width=0.01)
        plt.text(coords[i, 0]*1.15, coords[i, 1]*1.15, labels[i], fontweight='bold', ha='center')

    plt.axhline(0, color='black', lw=1, alpha=0.2); plt.axvline(0, color='black', lw=1, alpha=0.2)
    plt.title("Phase III: The Logic Basis Compass\nGeometric Disentanglement", fontsize=14, pad=20)
    plt.savefig("concept_compass.png", dpi=300); plt.close()

# --- 3. LOGIC HEATMAP & PARETO (Performance) ---

def plot_steering_performance(pkl_path="alpha_sweep_results.pkl"):
    """Generates heatmaps and the Pareto Frontier from sweep data."""
    if not os.path.exists(pkl_path):
        print(f"Skipping Performance Plots: {pkl_path} not found.")
        return

    df = pd.read_pickle(pkl_path)

    # 3a. Heatmaps (Success Rate by Alpha)
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        plot_data = ds_df.melt(id_vars=['alpha'], value_vars=['sign_acc', 'parity_acc'],
                               var_name='Metric', value_name='Success_Rate')
        pivot_df = plot_data.pivot(index='Metric', columns='alpha', values='Success_Rate')

        plt.figure(figsize=(12, 4))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".1f", vmin=0, vmax=100)
        plt.title(f"Steering Success Rate: {dataset}")
        plt.savefig(f"heatmap_{dataset}.png", bbox_inches='tight'); plt.close()

    # 3b. Pareto Frontier (Intensity vs Accuracy)
    interp_df = df[df['dataset'] == 'Interpolation'].sort_values('alpha')
    plt.figure(figsize=(10, 6))
    plt.plot(interp_df['alpha'], interp_df['total_acc'], 'o--', color='purple', label='Compliance')
    plt.title("Steering Pareto: Success vs. Alpha Intensity")
    plt.xlabel("Alpha (Steering Strength)"); plt.ylabel("Total Accuracy %")
    plt.savefig("pareto_frontier.png"); plt.close()

# --- 4. LOGIT-LENS (Causality) ---

def plot_logit_lens_automated(mlp, sae, feature_log="feature_subsets.pt"):
    """Maps the causal impact of SAE features directly to the MLP scalar output."""
    if not os.path.exists(feature_log):
        print(f"Skipping Logit-Lens: {feature_log} not found.")
        return

    subsets = torch.load(feature_log)

    for label, ids in subsets.items():
        clean_ids = [int(i) for i in (ids if isinstance(ids, (list, np.ndarray)) else list(ids))]
        
        if "Distinct" in label and len(clean_ids) > 0:
            with torch.no_grad():
                # Logic: [Output Weight] @ [Hidden2 Weight] @ [SAE Decoder]
                # Hidden2 (512 -> 256), Output (256 -> 1)
                w_out = mlp.layers['output'].weight # (1, 256)
                w_hid2 = mlp.layers['hidden2'].weight # (256, 512)
                w_dec = sae.decoder.weight # (512, 2048)
                
                # Combined attribution: (1, 512) @ (512, 2048) -> (1, 2048)
                w_causal = (w_out @ w_hid2) @ w_dec
                impact = w_causal[0, clean_ids].cpu().numpy()

            plt.figure(figsize=(max(len(clean_ids)*0.8, 8), 3))
            sns.heatmap(impact.reshape(1, -1), annot=True, cmap="RdBu_r", center=0,
                        xticklabels=clean_ids, yticklabels=[label])
            plt.title(f"Logit-Lens: Causal Attribution ({label})")
            plt.savefig(f"logit_lens_{label.replace(' ', '_')}.png", bbox_inches='tight'); plt.close()

# --- EXECUTION ---
if __name__ == '__main__':
    print("[Phase III] Generating Visualization Suite...")
    
    # 1. Geometry
    plot_steering_basis_compass()
    
    # 2. Performance & Trade-offs
    plot_steering_performance()
    
    # 3. Causality & Attribution
    try:
        mlp_model, sae_model = load_trained_models()
        plot_logit_lens_automated(mlp_model, sae_model)
    except Exception as e:
        print(f"Logit-Lens failed during model load: {e}")

    print("[Done] All visualizations exported to current directory.")