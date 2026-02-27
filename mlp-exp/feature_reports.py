
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
    mlp.load_state_dict(mlp_state if 'layers.output.weight' in mlp_state else mlp_state.get(
        'model_state_dict', mlp_state))
    mlp.eval()

    # Instantiate SAE based on architecture
    # Defaulting to dict_size=2048 as per Phase I context
    sae = SparseAutoencoder(input_dim=256, dict_size=2048)
    sae_state = torch.load(sae_path, map_location='cpu')
    sae.load_state_dict(sae_state if 'decoder.weight' in sae_state else sae_state.get(
        'sae_state_dict', sae_state))
    sae.eval()

    return mlp, sae

# --- 2. THE CONCEPT COMPASS (Geometry) ---


def plot_elegant_dual_compass(file_path="steering_basis.pt"):
    """
    Generates a professional dual-view compass.
    Left: High-detail quadrant zoom (fixed scale).
    Right: Adaptive professional overview (window-fitted).
    """
    if not os.path.exists(file_path):
        print(f"Skipping Compass: {file_path} not found.")
        return

    data = torch.load(file_path, map_location='cpu')
    # v_parity now represents subset (0-5 vs 5-10)
    v_sign, v_subset = data["v_sign"], data["v_parity"]

    # 1. Coordinate Setup
    vectors = torch.stack([v_sign, -v_sign, v_subset, -v_subset]).cpu().numpy()
    labels = ["Positive (+Sign)", "Negative (-Sign)",
              "Subset 0-5 (+Subset)", "Subset 5-10 (-Subset)"]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]

    # PCA Projection
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    max_mag = np.max(np.linalg.norm(coords, axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

    # 2. Left Plot: Quadrant Detail (Strict Zoom)
    # Scaled to capture the very center of the basis interaction
    for i in range(len(labels)):
        ax1.quiver(0, 0, coords[i, 0], coords[i, 1],
                   angles='xy', scale_units='xy', scale=1,
                   color=colors[i], width=0.018, headwidth=4, headlength=5)

    # Recreating the quadrant-focused scale (approx 0.05 per side)
    detail_limit = max_mag * 0.05
    ax1.set_xlim(-detail_limit, detail_limit)
    ax1.set_ylim(-detail_limit, detail_limit)
    ax1.set_title("Zoomed In: Quadrant Detail",
                  fontsize=12, fontweight='bold', pad=15)

    # 3. Right Plot: Professional Overview (Adaptive Scaling)
    # Scaled just enough to show arrows and legend elegantly
    for i in range(len(labels)):
        ax2.quiver(0, 0, coords[i, 0], coords[i, 1],
                   angles='xy', scale_units='xy', scale=1,
                   color=colors[i], label=labels[i],
                   width=0.012, headwidth=5, headlength=7)

        # Professional label placement at vector tips
        ax2.text(coords[i, 0] * 1.05, coords[i, 1] * 1.05, labels[i],
                 fontsize=9, fontweight='bold', ha='center', va='center')

    overview_limit = max_mag * 1.35  # Provides breathing room for legend/labels
    ax2.set_xlim(-overview_limit, overview_limit)
    ax2.set_ylim(-overview_limit, overview_limit)
    ax2.set_title("Adaptive Overview: Basis Compass",
                  fontsize=12, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)

    # 4. Global Styling
    for ax in [ax1, ax2]:
        circle = plt.Circle((0, 0), max_mag, color='gray',
                            fill=False, linestyle='--', alpha=0.15)
        ax.add_artist(circle)
        ax.axhline(0, color='black', lw=0.8, alpha=0.3)
        ax.axvline(0, color='black', lw=0.8, alpha=0.3)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.2)

    plt.suptitle(" Logic Basis Geometric Disentanglement",
                 fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("images/concept_compass_elegant.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print("     Successfully generated concept compass with zoomed-in & zoomed out views.")

# --- 3. LOGIC HEATMAP & PARETO (Performance) ---


def plot_steering_performance_unified(pkl_path="alpha_sweep_results.pkl"):
    """
    Generates:
    1. Unified Nested Heatmap (Dataset > Metric)
    2. Unified Pareto Frontier (Compliance across all datasets)
    """
    if not os.path.exists(pkl_path):
        print(f"  [!] Skipping Performance Plots - {pkl_path} not found.")
        return

    df = pd.read_pickle(pkl_path)
    os.makedirs("images", exist_ok=True)

    # --- 1. Define Custom Ordering & Categories ---
    dataset_order = ["Interpolation", "Extrapolation", "Scaling", "Precision"]

    # Convert to categorical to force the sort order
    df['dataset'] = pd.Categorical(
        df['dataset'], categories=dataset_order, ordered=True)
    df = df.sort_values(['dataset', 'alpha'])

    # --- 2. Unified Nested Heatmap ---
    melted_df = df.melt(
        id_vars=['dataset', 'alpha'],
        value_vars=['sign_acc', 'subset_acc'],
        var_name='Metric',
        value_name='Success_Rate'
    )

    # Pivot with Dataset as the primary index and Metric as the sub-index
    unified_pivot = melted_df.pivot_table(
        index=['dataset', 'Metric'],
        columns='alpha',
        values='Success_Rate',
        sort=False  # Preserves our categorical sort
    )

    # Clean up Metric labels (sign_acc -> Sign, subset_acc -> Subset)
    new_labels = ["Sign" if label ==
                  "sign_acc" else "Subset" for label in unified_pivot.index.levels[1]]
    unified_pivot.index = unified_pivot.index.set_levels(new_labels, level=1)

    # --- 3. Visualization ---
    plot_height = len(df['dataset'].unique()) * 1.8 + 1
    plt.figure(figsize=(14, plot_height))

    sns.heatmap(
        unified_pivot,
        annot=True,
        cmap="YlGnBu",
        fmt=".1f",
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Success Rate %'},
        linewidths=.5
    )

    plt.title("Steering Performance: Hierarchical Dataset Breakdown",
              fontsize=14, pad=20)
    plt.xlabel("Alpha (Steering Strength)", fontsize=12)
    plt.ylabel("Dataset | Control Metric", fontsize=12)

    # Save the consolidated heatmap
    plt.savefig("images/unified_logic_heatmap.png",
                bbox_inches='tight', dpi=300)
    plt.close()

    # --- 3. UNIFIED PARETO FRONTIER ---
    plt.figure(figsize=(10, 6))

    # Professional color palette for lines
    colors = ["#8e44ad", "#e67e22", "#2980b9",
              "#27ae60"]  # Purple, Orange, Blue, Green

    for i, dataset in enumerate(dataset_order):
        ds_df = df[df['dataset'] == dataset].sort_values('alpha')

        if not ds_df.empty:
            plt.plot(
                ds_df['alpha'],
                ds_df['total_acc'],
                marker='o',
                linestyle='--',
                linewidth=2,
                color=colors[i],
                label=f'Compliance ({dataset})'
            )

    plt.title("Unified Steering Pareto: Compliance vs. Alpha Intensity",
              fontsize=14, pad=15)
    plt.xlabel("Alpha (Steering Strength)", fontsize=12)
    plt.ylabel("Total Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)  # Ensure visibility of 100% baseline
    plt.grid(True, linestyle=':', alpha=0.6)

    # Legend placement to avoid overlapping lines
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

    plt.tight_layout()
    plt.savefig("images/unified_pareto_frontier.png", dpi=300)
    plt.close()

    print("     Successfully generated unified heatmap and Pareto frontier in /images.")

# --- 4. LOGIT-LENS (Causality) ---


def plot_unified_logit_lens(mlp, sae, feature_log="feature_subsets.pt"):
    """
    Generates a single, sorted heatmap of causal attribution across all categories
    to visualize feature overlaps and functional correlations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    if not os.path.exists(feature_log):
        print(f"Skipping Unified Lens: {feature_log} not found.")
        return

    subsets = torch.load(feature_log)

    # 1. Collect and sort the union of all identified features
    all_ids = []
    for label, ids in subsets.items():
        all_ids.extend([int(i) for i in (ids if isinstance(
            ids, (list, np.ndarray)) else list(ids))])

    unique_ids = sorted(list(set(all_ids)))

    if not unique_ids:
        print("No features found in log.")
        return

    # 2. Compute Global Causal Attribution (Logit Lens)
    # Math: Since SAE is at output of hidden2, we only project through the output layer.
    # Dimensions: w_out (1, 256) @ w_dec (256, 2048) -> (1, 2048)
    with torch.no_grad():
        w_out = mlp.layers['output'].weight    # Shape: (1, 256)
        w_dec = sae.decoder.weight             # Shape: (256, 2048)

        # Simplified Projection for the new injection site
        w_causal = w_out @ w_dec

        # Extract weights only for our unique features of interest
        global_impacts = w_causal[0, unique_ids].cpu().numpy()

    # 3. Build the Matrix for Visualization
    # We want to see how these specific weights relate to our category labels
    matrix_data = []
    category_labels = []

    # Clean up category labels for new style
    for label, ids in subsets.items():
        clean_label = label.replace('Odd Parity', 'Subset 0-5').replace('Even Parity', 'Subset 5-10').replace(
            'Positive Sign', 'Positive').replace('Negative Sign', 'Negative')
        cat_ids = [int(i) for i in (ids if isinstance(
            ids, (list, np.ndarray)) else list(ids))]

        # Create a row where we only show the impact if the feature is in this category
        row = []
        for fid in unique_ids:
            if fid in cat_ids:
                idx = unique_ids.index(fid)
                row.append(global_impacts[idx])
            else:
                row.append(np.nan)

        matrix_data.append(row)
        category_labels.append(clean_label)

    # 4. Plotting
    plt.figure(figsize=(max(len(unique_ids) * 0.8, 12),
                        len(category_labels) * 1.5))

    # RdBu_r: Red = Positive Output Push, Blue = Negative Output Push
    sns.heatmap(matrix_data,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                xticklabels=unique_ids,
                yticklabels=category_labels,
                cbar_kws={'label': 'Causal Push to Logit'})

    plt.title("Unified Logit-Lens: Causal Attribution (Hidden2 Injection)",
              fontsize=15, pad=20)
    plt.xlabel("SAE Feature ID", fontsize=12)
    plt.ylabel("Logic Category", fontsize=12)

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/unified_logit_lens.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(
        f"     Success: Unified Logit-Lens generated for {len(unique_ids)} features.")


# --- EXECUTION ---
if __name__ == '__main__':
    print("\n" + "="*70)
    print(" GENERATING VISUALIZATION SUITE")
    print("="*70 + "\n")

    # Create images directory
    os.makedirs("images", exist_ok=True)

    # 1. Geometry
    print("  -> Generating Steering Basis Compass...")
    plot_elegant_dual_compass()

    # 2. Performance & Trade-offs
    print("  -> Generating Performance Heatmaps & Pareto Frontier...")
    plot_steering_performance_unified()

    # 3. Causality & Attribution
    print("  -> Generating Logit-Lens Visualizations...")
    try:
        mlp_model, sae_model = load_trained_models()
        plot_unified_logit_lens(mlp_model, sae_model)
    except Exception as e:
        print(f"  [FAIL] Logit-Lens failed during model load: {e}")

    print("\n" + "="*70)
    print("  [OK] VISUALIZATION SUITE COMPLETE")
    print("  All visualizations exported to images/ folder")
    print("="*70 + "\n")
