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

    # Mapping based on CONCEPT_MAP: pos_odd=0, pos_even=1, neg_odd=2, neg_even=3
    pos_mask = (tags == 0) | (tags == 1)
    neg_mask = (tags == 2) | (tags == 3)
    odd_mask = (tags == 0) | (tags == 2)
    even_mask = (tags == 1) | (tags == 3)

    # Calculate Basis Vectors in Feature Space
    v_sign = latent_features[pos_mask].mean(
        0) - latent_features[neg_mask].mean(0)
    v_parity = latent_features[odd_mask].mean(
        0) - latent_features[even_mask].mean(0)

    return v_sign, v_parity


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

    print("\n" + "="*70)
    print("  ANALYZING FEATURE ACTIVATIONS ACROSS GROUPS")
    print("="*70)
    print("  -> Processing test samples and extracting SAE features...\n")
    with torch.no_grad():
        for _, row in df.iterrows():
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

    print("\n" + "="*70)
    print(f"  TOP-{k} FEATURES PER CONCEPT GROUP")
    print("="*70)
    for group, feats in feats_by_grp.items():
        print(f"  {group:12} : {feats}")

    all_feats = feats_by_grp["pos_odd"] + feats_by_grp["pos_even"] + \
        feats_by_grp["neg_odd"] + feats_by_grp["neg_even"]

    # Logic to find "Pure" features
    pos_feats = set(feats_by_grp['pos_odd']) & set(
        feats_by_grp['pos_even'])
    neg_feats = set(feats_by_grp['neg_odd']) & set(
        feats_by_grp['neg_even'])
    odd_feats = set(feats_by_grp['pos_odd']) & set(
        feats_by_grp['neg_odd'])
    eve_feats = set(feats_by_grp['pos_even']) & set(
        feats_by_grp['neg_even'])

    # Common Features
    common_feats = pos_feats & odd_feats & neg_feats & eve_feats
    print(f"\n  [OK] Universal Common Features: {sorted(list(common_feats))}")

    print("\n" + "="*70)
    print("  IDENTIFIED FEATURE SUBSETS (INTERSECTIONS)")
    print("="*70)
    print(f"  Positive Sign Features : {sorted(list(pos_feats))}")
    print(f"  Odd Parity Features    : {sorted(list(odd_feats))}")
    print(f"  Negative Sign Features : {sorted(list(neg_feats))}")
    print(f"  Even Parity Features   : {sorted(list(eve_feats))}")

    pos_only_feats = pos_feats - common_feats
    neg_only_feats = neg_feats - common_feats
    odd_only_feats = odd_feats - common_feats
    eve_only_feats = eve_feats - common_feats

    print("\n  DISTINCT (Non-Common) Features:")
    print(f"    → Even Parity        : {sorted(list(eve_only_feats))}")
    print(f"    → Positive Sign      : {sorted(list(pos_only_feats))}")
    print(f"    → Odd Parity        : {sorted(list(odd_only_feats))}")
    print(f"    → Negative Sign      : {sorted(list(neg_only_feats))}")

    # 1. Create the dictionary of discovered feature IDs
    feature_subsets = {
        # Mixed features
        "Mixed Positive Odd": feats_by_grp["pos_odd"],
        "Mixed Positive Even": feats_by_grp["pos_even"],
        "Mixed Negative Odd": feats_by_grp["neg_odd"],
        "Mixed Negative Even": feats_by_grp["neg_even"],

        # All top features across groups
        "All": all_feats,

        "Positive (Odd/Even)": pos_feats,
        "Negative (Odd/Even)": neg_feats,
        "Odd Parity (Pos/Neg)": odd_feats,
        "Even Parity (Pos/Neg)": eve_feats,

        "Universal Common": common_feats,

        # Distinct features
        "Distinct Even Parity": eve_only_feats,
        "Distinct Odd Parity": odd_only_feats,
        "Distinct Positive Sign": pos_only_feats,
        "Distinct Negative Sign": neg_only_feats,

    }

    # 2. Save the dictionary for Phase III
    torch.save(feature_subsets, "feature_subsets.pt")
    print(f"\n  [OK] Successfully saved {len(feature_subsets)} feature groups")
    print("="*70 + "\n")

    return {
        "pos_sign": sorted(list(pos_only_feats)),
        "neg_sign": sorted(list(neg_only_feats)),
        "odd_parity": sorted(list(odd_only_feats)),
        "even_parity": sorted(list(eve_only_feats)),
    }


class UniversalSteeringController:
    def __init__(self, mlp_path, sae_path, basis_path, latent_stats_path=None):
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
        # If you haven't saved stats, we use a small epsilon.
        # Ideally, load the 'std' of these features calculated during Phase II.
        if latent_stats_path:
            stats = torch.load(latent_stats_path)
            std_sign = stats.get('sign_std', 1.0)
            std_parity = stats.get('parity_std', 1.0)
        else:
            std_sign = 1.0
            std_parity = 0.25

        # CHANGE THESE NAMES:
        self.norm_v_sign = v_sign / (std_sign + 1e-8)
        self.norm_v_parity = v_parity / (std_parity + 1e-8)

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

    print(f"\n{'-'*70}")
    print(f"  [*] ABLATION TEST: {label}")
    print(f"{'-'*70}")
    print(f"  Original Output : {baseline_out:8.4f}")
    print(f"  Ablated Output  : {final_out:8.4f}")

    shift = final_out - baseline_out
    print(f"  Causal Shift    : {shift:+8.4f}")


if __name__ == "__main__":

    # 3. Extract Vectors
    v_sign, v_parity = get_universal_vectors(
        "harvested_data.pt", "sae/universal_sae.pth")

    # 4. Metric: Orthogonality Check
    cosine_sim = torch.nn.functional.cosine_similarity(
        v_sign.unsqueeze(0), v_parity.unsqueeze(0))

    print("\n" + "="*70)
    print("  STEERING BASIS VECTORS ANALYSIS")
    print("="*70)
    print(f"  Sign-Parity Cosine Similarity: {cosine_sim.item():.4f}")
    print("  Interpretation: Near 0.0 → concepts are perfectly disentangled ✓")
    print("="*70 + "\n")

    torch.save({"v_sign": v_sign, "v_parity": v_parity}, "steering_basis.pt")

    dist_feat = get_distinct_features_by_group(k=64)

    test_inputs = [
        ([6, 3, 0, 6, 2, 2, 8, 5, 3, 3], -3),  # Original: -3 (Negative, Odd)
        ([6, 3, 2, 7, 1, 7, 5, 7, 9, 1], -2),  # Original: -2 (Negative, Even)
        ([0, 8, 4, 6, 2, 5, 3, 8, 4, 0], -1),  # Original: -1 (Negative, Odd)
        ([8, 3, 9, 0, 0, 7, 2, 2, 7, 0], 1),  # Original: 1 (Positive, Odd)
        ([1, 3, 4, 6, 3, 4, 9, 0, 6, 0], 2),  # Original: 2 (Positive, Even)
        ([6, 5, 2, 2, 1, 7, 2, 5, 4, 1], 3)   # Original: 3 (Positive, Odd)
    ]

    for inp in test_inputs:

        if inp[-1] < 0:
            # Ablation Test: Kill Negative Sign (Expect output to rise or become positive)
            run_surgical_ablation(
                inp[0], dist_feat["neg_sign"], "Killing Negative Sign")
        else:
            # Ablation Test: Kill Positive Sign (Expect output to rise or become positive)
            run_surgical_ablation(
                inp[0], dist_feat["pos_sign"], "Killing Positive Sign")

        if inp[-1] % 2 != 0:
            # Ablation Test: Kill Odd Parity (Expect output to rise or become positive)
            run_surgical_ablation(
                inp[0], dist_feat["odd_parity"], "Killing Odd Parity")
        else:
            # Ablation Test: Kill Even Parity (Expect output to rise or become positive)
            run_surgical_ablation(
                inp[0], dist_feat["even_parity"], "Killing Even Parity")

    controller = UniversalSteeringController(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", "steering_basis.pt")

    for inp in test_inputs:

        print(
            f"Actual Input: {inp[0]}, Expected Output: {inp[1]}"
        )
        print(
            f"({'Negative' if inp[1] < 0 else 'Positive'}, {'Odd' if inp[1] % 2 != 0 else 'Even'})"
        )

        input_tensor = torch.tensor([inp[0]], dtype=torch.float32)
        print(
            f"Predicted Output: {controller.steer_input(input_tensor, 0, 0)}")
        print(
            f"    Steer to Positive: {controller.steer_input(input_tensor, target_sign=1, target_parity=0)}")
        print(
            f"    Steer to Negative: {controller.steer_input(input_tensor, target_sign=-1, target_parity=0)}")

        print(
            f"    Steer to Odd: {controller.steer_input(input_tensor, target_sign=0, target_parity=1)}")
        print(
            f"    Steer to Even: {controller.steer_input(input_tensor, target_sign=0, target_parity=-1)}")

        print(
            f"        Steer to Positive-Odd: {controller.steer_input(input_tensor, target_sign=1, target_parity=1)}")
        print(
            f"        Steer to Positive-Even: {controller.steer_input(input_tensor, target_sign=1, target_parity=-1)}")

        print(
            f"        Steer to Negative-Odd: {controller.steer_input(input_tensor, target_sign=-1, target_parity=1)}")
        print(
            f"        Steer to Negative-Even: {controller.steer_input(input_tensor, target_sign=-1, target_parity=-1)}")
        print("-" * 50)
