from sklearn.decomposition import PCA
import torch
from sae.sae_definition import SparseAutoencoder
from mlp.mlp_definition import InterpretabilityMLP
import pandas as pd
from collections import defaultdict


def get_universal_vectors(harvested_data, sae_model):

    # 1. Load Data
    payload = torch.load(harvested_data)
    activations = payload["activations"]
    tags = payload["labels"]

    # 2. Load SAE (Ensure parameters match your training config)
    sae = SparseAutoencoder(input_dim=256, dict_size=2048).cuda()
    sae.load_state_dict(torch.load(sae_model))

    sae.eval()
    with torch.no_grad():
        # Get the sparse features (hidden layer of SAE)
        _, latent_features = sae(activations.cuda())
        latent_features = latent_features.cpu()

    # New mapping: sign (+/-) and subset (0-5, 5-10)
    # Assume tags: 0 = +00 < pos <= +05, 1 = +05 < pos <= +10, 2 = -05 <= neg < +00, 3 = -10 <= neg < -05
    pos_mask = (tags == 0) | (tags == 1)
    neg_mask = (tags == 2) | (tags == 3)
    subset1_mask = (tags == 0) | (tags == 2)  # 0-5 (pos or neg)
    subset2_mask = (tags == 1) | (tags == 3)  # 5-10 (pos or neg)

    # Calculate Basis Vectors in Feature Space
    v_sign = latent_features[pos_mask].mean(
        0) - latent_features[neg_mask].mean(0)
    v_subset = latent_features[subset1_mask].mean(
        0) - latent_features[subset2_mask].mean(0)

    return v_sign, v_subset


def get_pca_vectors(harvested_data, sae_model, n_components=2):
    """
    Extracts the top PCA directions from SAE latent space for use as steering baselines.
    Returns the first n_components principal components as torch tensors.
    """
    payload = torch.load(harvested_data)
    activations = payload["activations"]
    # 2. Load SAE
    sae = SparseAutoencoder(input_dim=256, dict_size=2048).cuda()
    sae.load_state_dict(torch.load(sae_model))
    sae.eval()
    with torch.no_grad():
        _, latent_features = sae(activations.cuda())
        latent_features = latent_features.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca.fit(latent_features)
    pcs = [torch.tensor(pc, dtype=torch.float32) for pc in pca.components_]
    return pcs


def get_top_k_features_by_group(mlp_path, sae_path, excel_path, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    mlp = InterpretabilityMLP().to(device)
    mlp.load_state_dict(torch.load(mlp_path))

    # CRITICAL FIX: Set to evaluation mode to handle Batch Norm with batch size 1
    mlp.eval()

    sae = SparseAutoencoder(input_dim=256, dict_size=2048).to(device)
    sae.load_state_dict(torch.load(sae_path))
    sae.eval()  # SAE should also be in eval mode

    # 2. Load Data
    df = pd.read_excel(excel_path)

    # Storage for feature activations: {group: {feature_index: total_activation}}
    group_features = defaultdict(lambda: defaultdict(float))
    group_counts = defaultdict(int)

    print("\n" + "="*85)
    print("  ANALYZING FEATURE ACTIVATIONS ACROSS GROUPS")
    print("="*85)
    print("  -> Processing test samples and extracting SAE features...\n")
    with torch.no_grad():
        for _, row in df.iterrows():
            # New concept: sign and subset
            # Expecting row['concept'] to be one of: '+00 < pos <= +05', '+05 < pos <= +10', '-05 <= neg < +00', '-10 <= neg < -05'
            group = row['concept']
            input_data = torch.tensor(
                eval(row['input_list']), dtype=torch.float32).unsqueeze(0).to(device)

            # Get SAE latents
            _ = mlp(input_data)
            hidden_acts = mlp.activations['hidden2']
            _, latents = sae(hidden_acts)

            # Record non-zero activations (Top-K ensures most are 0)
            active_indices = torch.nonzero(latents[0]).squeeze()
            if active_indices.dim() == 0:
                active_indices = active_indices.unsqueeze(0)

            for idx in active_indices:
                val = latents[0][idx].item()
                group_features[group][idx.item()] += val

            group_counts[group] += 1

    # 3. Analyze Subsets (Intersection Logic)
    results = {}
    for group in group_features:
        # Sort features by total activation strength
        sorted_feats = sorted(
            group_features[group].items(), key=lambda x: x[1], reverse=True)
        results[group] = [f[0] for f in sorted_feats[:k]]

    return results, group_features


def get_distinct_features_by_group(k=128):

    feats_by_grp, raw_data = get_top_k_features_by_group(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", "dataset/mlp_test.xlsx", k=k
    )

    print("\n" + "="*85)
    print(f"  TOP-{k} FEATURES PER CONCEPT GROUP")
    print("="*85)
    for group, feats in feats_by_grp.items():
        print(f"  {group:28} : {feats}")

    # New group names
    group1 = "+00 < pos <= +05"
    group2 = "+05 < pos <= +10"
    group3 = "-05 <= neg < +00"
    group4 = "-10 <= neg < -05"

    all_feats = feats_by_grp[group1] + feats_by_grp[group2] + \
        feats_by_grp[group3] + feats_by_grp[group4]

    # Logic to find "Pure" features
    pos_feats = set(feats_by_grp[group1]) | set(feats_by_grp[group2])
    neg_feats = set(feats_by_grp[group3]) | set(feats_by_grp[group4])
    subset1_feats = set(feats_by_grp[group1]) | set(
        feats_by_grp[group3])  # 0-5
    subset2_feats = set(feats_by_grp[group2]) | set(
        feats_by_grp[group4])  # 5-10

    # Common Features
    common_feats = pos_feats & neg_feats & subset1_feats & subset2_feats
    print(f"\n  [OK] Universal Common Features: {sorted(list(common_feats))}")

    print("\n" + "="*85)
    print("  IDENTIFIED FEATURE SUBSETS (UNIONS)")
    print("="*85)
    print(f"  Positive Sign Features : {sorted(list(pos_feats))}")
    print(f"  Subset 0-5 Features    : {sorted(list(subset1_feats))}")
    print(f"  Negative Sign Features : {sorted(list(neg_feats))}")
    print(f"  Subset 5-10 Features   : {sorted(list(subset2_feats))}")

    pos_only_feats = pos_feats - common_feats
    neg_only_feats = neg_feats - common_feats
    subset1_only_feats = subset1_feats - common_feats
    subset2_only_feats = subset2_feats - common_feats

    print("\n  DISTINCT (Non-Common) Features:")
    print(f"    → Subset 5-10        : {sorted(list(subset2_only_feats))}")
    print(f"    → Positive Sign      : {sorted(list(pos_only_feats))}")
    print(f"    → Subset 0-5         : {sorted(list(subset1_only_feats))}")
    print(f"    → Negative Sign      : {sorted(list(neg_only_feats))}")

    # 1. Create the dictionary of discovered feature IDs
    feature_subsets = {
        # Mixed features
        f"Mixed {group1}": feats_by_grp[group1],
        f"Mixed {group2}": feats_by_grp[group2],
        f"Mixed {group3}": feats_by_grp[group3],
        f"Mixed {group4}": feats_by_grp[group4],

        # All top features across groups
        "All": all_feats,

        "Positive (Group1/Group2)": pos_feats,
        "Negative (Group3/Group4)": neg_feats,
        "Subset 0-5 (Group1/Group3)": subset1_feats,
        "Subset 5-10 (Group2/Group4)": subset2_feats,

        "Universal Common": common_feats,

        # Distinct features
        "Distinct Subset 5-10": subset2_only_feats,
        "Distinct Subset 0-5": subset1_only_feats,
        "Distinct Positive Sign": pos_only_feats,
        "Distinct Negative Sign": neg_only_feats,

    }

    # 2. Save the dictionary for Phase III
    torch.save(feature_subsets, "temp/feature_subsets.pt")
    print(f"\n  [OK] Successfully saved {len(feature_subsets)} feature groups")
    print("="*85 + "\n")

    return {
        "pos_sign": sorted(list(pos_only_feats)),
        "neg_sign": sorted(list(neg_only_feats)),
        "subset_0_5": sorted(list(subset1_only_feats)),
        "subset_5_10": sorted(list(subset2_only_feats)),
    }


class PCAUniversalSteeringController:
    def __init__(self, mlp_path, sae_path, pca_vectors, calibration_excel_path="dataset/interp_test.xlsx"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = InterpretabilityMLP().to(self.device)
        self.mlp.load_state_dict(torch.load(
            mlp_path, map_location=self.device))
        self.mlp.eval()
        self.sae = SparseAutoencoder(
            input_dim=256, dict_size=2048).to(self.device)
        self.sae.load_state_dict(torch.load(
            sae_path, map_location=self.device))
        self.sae.eval()
        self.pca_vectors = [v.to(self.device) for v in pca_vectors]
        # Empirical calibration for output effect (optional, can be added as above)

    def steer_input(self, input_tensor, pca_idx=0, alpha=2.0):
        with torch.no_grad():
            _ = self.mlp(input_tensor.to(self.device))
            raw_neurons = self.mlp.activations['hidden2']
            _, baseline_latents = self.sae(raw_neurons)
            steered_latents = baseline_latents + \
                (alpha * self.pca_vectors[pca_idx])
            steered_neurons = self.sae.decoder(steered_latents)
            output = self.mlp.layers['output'](self.mlp.relu(steered_neurons))
            return output.item()


class UniversalSteeringController:

    def __init__(self, mlp_path, sae_path, basis_path, latent_stats_path=None, calibration_excel_path="dataset/interp_test.xlsx"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load MLP & Correct Architecture Mapping
        self.mlp = InterpretabilityMLP().to(self.device)
        self.mlp.load_state_dict(torch.load(
            mlp_path, map_location=self.device))
        self.mlp.eval()

        # 2. Load SAE (Input 512 matches 'hidden1' output)
        self.sae = SparseAutoencoder(
            input_dim=256, dict_size=2048).to(self.device)
        self.sae.load_state_dict(torch.load(
            sae_path, map_location=self.device))
        self.sae.eval()

        # 3. Load Steering Basis
        basis = torch.load(basis_path, map_location=self.device)
        v_sign = basis['v_sign']
        v_parity = basis['v_parity']

        # 4. Feature Normalization Logic
        if latent_stats_path:
            stats = torch.load(latent_stats_path)
            std_sign = stats.get('sign_std', 1.0)
            std_parity = stats.get('parity_std', 1.0)
        else:
            std_sign = 1.0
            std_parity = 0.25

        # Initial normalization
        norm_v_sign = v_sign / (std_sign + 1e-8)
        norm_v_parity = v_parity / (std_parity + 1e-8)

        # --- Empirical calibration for output effect ---
        # Use a batch of calibration samples to match the output effect of sign and parity steering
        try:
            import pandas as pd
            df = pd.read_excel(calibration_excel_path).head(100)
            sign_effects = []
            parity_effects = []
            with torch.no_grad():
                for _, row in df.iterrows():
                    input_data = torch.tensor(
                        eval(row['input_list']), dtype=torch.float32).unsqueeze(0).to(self.device)
                    # Baseline
                    _ = self.mlp(input_data)
                    raw_neurons = self.mlp.activations['hidden2']
                    _, latents = self.sae(raw_neurons)
                    # Steer sign
                    steered_latents_sign = latents + norm_v_sign
                    steered_neurons_sign = self.sae.decoder(
                        steered_latents_sign)
                    out_sign = self.mlp.layers['output'](
                        self.mlp.relu(steered_neurons_sign)).item()
                    # Steer parity
                    steered_latents_parity = latents + norm_v_parity
                    steered_neurons_parity = self.sae.decoder(
                        steered_latents_parity)
                    out_parity = self.mlp.layers['output'](
                        self.mlp.relu(steered_neurons_parity)).item()
                    # Baseline output
                    out_base = self.mlp.layers['output'](
                        self.mlp.relu(raw_neurons)).item()
                    sign_effects.append(abs(out_sign - out_base))
                    parity_effects.append(abs(out_parity - out_base))
            mean_sign = sum(sign_effects) / (len(sign_effects) + 1e-8)
            mean_parity = sum(parity_effects) / (len(parity_effects) + 1e-8)
            # Compute scaling factor for parity
            parity_scale = mean_sign / \
                (mean_parity + 1e-8) if mean_parity > 0 else 1.0
            self.norm_v_sign = norm_v_sign
            self.norm_v_parity = norm_v_parity * parity_scale
            print(
                f"[Calibration] Parity steering scaled by {parity_scale:.3f} to match sign effect.")
        except Exception as e:
            # Fallback: no scaling
            self.norm_v_sign = norm_v_sign
            self.norm_v_parity = norm_v_parity
            print(f"[Calibration] Parity scaling skipped due to error: {e}")

    def steer_input(self, input_tensor, target_sign=0, target_parity=0, alpha=2.0):
        with torch.no_grad():
            # 1. Forward pass to injection site
            _ = self.mlp(input_tensor.to(self.device))
            raw_neurons = self.mlp.activations['hidden2']

            # 2. Get baseline latents
            _, baseline_latents = self.sae(raw_neurons)

            # 3. Apply Steering using the correctly named normalized vectors
            steered_latents = baseline_latents + \
                (target_sign * alpha * self.norm_v_sign) + \
                (target_parity * alpha * self.norm_v_parity)

            # 4. Decode and finish MLP pass
            steered_neurons = self.sae.decoder(steered_latents)
            output = self.mlp.layers['output'](self.mlp.relu(steered_neurons))

            return output.item()


def print_ablation_spatial(label, baseline, final):
    diff = final - baseline
    # Scale: 10 slots on each side. Each slot = 0.2 units of shift (adjust as needed)
    scale = 0.2
    num_chars = min(10, int(abs(diff) / scale))

    if diff >= 0:
        # Shift to the right: Space | Block + Arrow
        bar = " " * 10 + "|" + ("█" * num_chars + ">").ljust(10)
        print(
            f"{label:<22} : {baseline:7.3f} (baseline) [{bar}] {final:7.3f} (finalize) ({diff:+6.3f})")
    else:
        # Shift to the left: Arrow + Block | Space
        bar = ("<" + "█" * num_chars).rjust(10) + "|" + " " * 10
        print(
            f"{label:<22} : {final:7.3f} (finalize) [{bar}] {baseline:7.3f} (baseline) ({diff:+6.3f})")


def run_surgical_ablation(input_vals, target_features, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    mlp = InterpretabilityMLP().to(device)
    mlp.load_state_dict(torch.load("mlp/perfect_mlp.pth"))
    mlp.eval()

    sae = SparseAutoencoder(input_dim=256, dict_size=2048).to(device)
    sae.load_state_dict(torch.load("sae/universal_sae.pth"))
    sae.eval()

    # 2. Prepare Input
    # input_vals is a list of 256 floats representing your matrix
    input_tensor = torch.tensor(
        input_vals, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Baseline Run (No interference)
    with torch.no_grad():
        baseline_out = mlp(input_tensor).item()

        # Capture activations and pass through SAE
        hidden_acts = mlp.activations['hidden2']
        _, latents = sae(hidden_acts)

        # 4. ABLATION: Zero out the "Hero Features"
        ablated_latents = latents.clone()
        for idx in target_features:
            ablated_latents[0, idx] = 0.0

        # 5. RECONSTRUCTION: Using the Linear layer's own weights and bias
        # This is mathematically equivalent to sae.decoder(ablated_latents)
        reconstructed_acts = sae.decoder(ablated_latents)

        # Pass through output
        final_out = mlp.layers['output'](mlp.relu(reconstructed_acts)).item()

    print_ablation_spatial(label, baseline_out, final_out)


def print_steering_dashboard(expected, predicted, interventions):
    print(f"\n{'='*85}")
    print(f" TARGET: {expected:^10} | INPUT LOGIC: {'Positive' if expected > 0 else 'Negative'}, Subset {'0-5' if abs(expected) <= 5 else '5-10'}")
    print(f"{'='*85}")

    # helper for the visual line
    def get_line(val):
        # Scale: -20 to +20 mapped to 40 chars
        center = 20
        pos = int(val + center)
        pos = max(0, min(39, pos))
        line = [" "] * 40
        line[20] = "|"  # Center marker
        line[pos] = "●"  # Current value
        return "".join(line)

    print(f" Original Prediction : {predicted:7.3f}  [{get_line(predicted)}]")
    print(f"{'-'*85}")

    for label, steered_val in interventions.items():
        diff = steered_val - predicted
        arrow = "→" if diff > 0 else "←"
        # Color-coded feel (symbolic)
        print(
            f" {label:<20}: {steered_val:7.3f}  [{get_line(steered_val)}] (Shift: {diff:+6.2f} {arrow})")


if __name__ == "__main__":

    # 1. Extract Vectors
    v_sign, v_parity = get_universal_vectors(
        "temp/harvested_data.pt", "sae/universal_sae.pth")

    # 2. Metric: Orthogonality Check
    cosine_sim = torch.nn.functional.cosine_similarity(
        v_sign.unsqueeze(0), v_parity.unsqueeze(0)
    )

    print("\n" + "="*85)
    print("  STEERING BASIS VECTORS ANALYSIS")
    print("="*85)
    print(f"  Sign-Parity Cosine Similarity: {cosine_sim.item():.4f}")
    print("  Interpretation: Near 0.0 → concepts are perfectly disentangled ✓")
    print("="*85 + "\n")

    torch.save({"v_sign": v_sign, "v_parity": v_parity}, "temp/steering_basis.pt")

    # 3. PCA Baseline Extraction
    print("\n" + "="*85)
    print("  PCA BASELINE VECTOR EXTRACTION")
    print("="*85)
    pca_vectors = get_pca_vectors(
        "temp/harvested_data.pt", "sae/universal_sae.pth", n_components=2)
    print(f"  Extracted {len(pca_vectors)} PCA directions from SAE latents.")

    # 4. PCA Steering Controller
    pca_controller = PCAUniversalSteeringController(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", pca_vectors)

    # 5. Compliance Evaluation: SAE vs PCA
    print("\n" + "="*85)
    print("  COMPLIANCE EVALUATION: SAE vs PCA Steering")
    print("="*85)

    import pandas as pd
    df = pd.read_excel("dataset/interp_test.xlsx").head(100)

    sae_controller = UniversalSteeringController(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", "temp/steering_basis.pt")

    alpha_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    print("\nAlpha Sweep Compliance (SAE vs PCA):")
    for alpha in alpha_values:

        sae_sign_success = 0
        pca_sign_success = 0
        sae_subset_success = 0
        pca_subset_success = 0
        total = 0

        for _, row in df.iterrows():

            input_data = torch.tensor(
                eval(row['input_list']), dtype=torch.float32).unsqueeze(0)
            concept = row['concept']

            # Determine sign from concept string
            orig_is_pos = ('+' in concept)
            target_s = -1 if orig_is_pos else 1
            # Determine subset from concept string
            in_0_5 = '0 <' in concept or '0< ' in concept or '0<' in concept or '0-5' in concept or '0 <=' in concept
            target_subset = -1 if in_0_5 else 1

            # SAE steering (sign)
            sae_out = sae_controller.steer_input(
                input_data, target_sign=target_s, target_parity=0, alpha=alpha)
            sae_sign_success += int((target_s == 1 and sae_out > 0)
                                    or (target_s == -1 and sae_out < 0))
            # PCA steering (sign)
            pca_out = pca_controller.steer_input(
                input_data, pca_idx=0, alpha=alpha)
            pca_sign_success += int((target_s == 1 and pca_out > 0)
                                    or (target_s == -1 and pca_out < 0))

            # SAE steering (subset)
            sae_out_subset = sae_controller.steer_input(
                input_data, target_sign=0, target_parity=target_subset, alpha=alpha)
            sae_subset_success += int((target_subset == 1 and abs(sae_out_subset) > 5)
                                      or (target_subset == -1 and 0 < abs(sae_out_subset) <= 5))
            # PCA steering (subset)
            pca_out_subset = pca_controller.steer_input(
                input_data, pca_idx=0, alpha=alpha)
            pca_subset_success += int((target_subset == 1 and abs(pca_out_subset) > 5)
                                      or (target_subset == -1 and 0 < abs(pca_out_subset) <= 5))

            total += 1
        print(f"  Alpha={alpha:5.1f} | SAE Sign: {sae_sign_success}/{total} ({100*sae_sign_success/total:.1f}%) | PCA Sign: {pca_sign_success}/{total} ({100*pca_sign_success/total:.1f}%) | SAE Subset: {sae_subset_success}/{total} ({100*sae_subset_success/total:.1f}%) | PCA Subset: {pca_subset_success}/{total} ({100*pca_subset_success/total:.1f}%)")

    # 6. Surgical Ablation
    dist_feat = get_distinct_features_by_group(k=128)

    test_inputs = [
        # Edge Cases near category boundaries
        ([0, 7, 10, 6, 0, 1, 2, 10, 9, 2], -10),
        ([1, 7, 9, 10, 3, 1, 10, 5, 0, 3], 10),

        # Incremental Category Shifts
        ([9, 10, 8, 1, 3, 10, 1, 0, 9, 3], -8),
        ([9, 0, 1, 8, 2, 3, 9, 7, 5, 2], -6),
        ([7, 3, 6, 5, 3, 3, 2, 8, 8, 2], -3),
        ([1, 6, 4, 6, 1, 3, 9, 3, 7, 3], -1),
        ([5, 1, 4, 4, 3, 5, 9, 3, 3, 2], 1),
        ([1, 6, 10, 5, 2, 5, 5, 3, 7, 3], 3),
        ([5, 4, 8, 8, 2, 5, 3, 1, 2, 3], 6),
        ([10, 9, 7, 7, 1, 6, 7, 10, 1, 3], 8)

    ]

    for inp in test_inputs:
        # inp[0] is the tensor, inp[-1] is the scalar target/ground truth
        val = inp[-1]

        if val < 0:

            # 1. Test Sign: Kill Negative Features
            run_surgical_ablation(
                inp[0], dist_feat["neg_sign"], "Kill Neg Sign")

            # 2. Test Subset: Check if it's in the 'Small' or 'Large' negative range
            if -5 <= val < 0:

                run_surgical_ablation(
                    inp[0], dist_feat["subset_0_5"], "Kill (-5, 0) Subset")

            else:  # val is between -10 and -5

                run_surgical_ablation(
                    inp[0], dist_feat["subset_5_10"], "Kill (-10, -5) Subset")

        else:  # val >= 0

            # 1. Test Sign: Kill Positive Features
            run_surgical_ablation(
                inp[0], dist_feat["pos_sign"], "Kill Pos Sign")

            # 2. Test Subset: Check if it's in the 'Small' or 'Large' positive range
            if 0 <= val <= 5:

                run_surgical_ablation(
                    inp[0], dist_feat["subset_0_5"], "Kill (0, 5) Subset")

            else:  # val is between 5 and 10

                run_surgical_ablation(
                    inp[0], dist_feat["subset_5_10"], "Kill (5, 10) Subset")

    # 7. Latent Steering
    controller = UniversalSteeringController(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", "temp/steering_basis.pt")

    for inp in test_inputs:
        print(f"Actual Input: {inp[0]}, Expected Output: {inp[1]}")
        # Determine sign and subset for display
        sign_str = 'Negative' if inp[1] < 0 else 'Positive'
        subset_str = '0-5' if 0 < abs(inp[1]) <= 5 else '5-10'
        print(f"({sign_str}, Subset {subset_str})")

        input_tensor = torch.tensor([inp[0]], dtype=torch.float32)

        interventions = {

            "Flipped: POS + SML": controller.steer_input(input_tensor, target_sign=1, target_parity=1),
            "Steer to Positive": controller.steer_input(input_tensor, target_sign=1, target_parity=0),
            "Flipped: POS + LRG": controller.steer_input(input_tensor, target_sign=1, target_parity=-1),

            "Steer to Subset 5-10": controller.steer_input(input_tensor, target_sign=0, target_parity=-1),

            "Flipped: NEG + LRG": controller.steer_input(input_tensor, target_sign=-1, target_parity=-1),
            "Steer to Negative": controller.steer_input(input_tensor, target_sign=-1, target_parity=0),
            "Flipped: NEG + SML": controller.steer_input(input_tensor, target_sign=-1, target_parity=1),

            "Steer to Subset 0-5": controller.steer_input(input_tensor, target_sign=0, target_parity=1),

        }

        print_steering_dashboard(
            inp[1],
            controller.steer_input(
                input_tensor, 0, 0
            ),
            interventions
        )

        print("-" * 85)
