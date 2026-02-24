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
    sae = SparseAutoencoder(input_dim=512, dict_size=2048, k=20).cuda()
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

    sae = SparseAutoencoder(input_dim=512, dict_size=2048).to(device)
    sae.load_state_dict(torch.load(sae_path))
    sae.eval()  # SAE should also be in eval mode

    # 2. Load Data
    df = pd.read_excel(excel_path)

    # Storage for feature activations: {group: {feature_index: total_activation}}
    group_features = defaultdict(lambda: defaultdict(float))
    group_counts = defaultdict(int)

    print("Analyzing feature activations across groups...")
    with torch.no_grad():
        for _, row in df.iterrows():
            group = row['concept']
            input_data = torch.tensor(
                eval(row['input_list']), dtype=torch.float32).unsqueeze(0).to(device)

            # Get SAE latents
            _ = mlp(input_data)
            hidden_acts = mlp.activations['layer2']
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


def get_distinct_features_by_group():

    k = 50
    features_by_group, raw_data = get_top_k_features_by_group(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", "dataset/mlp_test.xlsx", k=k
    )

    print(f"\n--- Top-{k} Features per Group ---")
    for group, feats in features_by_group.items():
        print(f"{group}: {feats}")

    # Logic to find "Pure" features
    pos_feats = set(features_by_group['pos_odd']) & set(
        features_by_group['pos_even'])
    neg_feats = set(features_by_group['neg_odd']) & set(
        features_by_group['neg_even'])
    odd_feats = set(features_by_group['pos_odd']) & set(
        features_by_group['neg_odd'])
    eve_feats = set(features_by_group['pos_even']) & set(
        features_by_group['neg_even'])

    # Common Features
    common_feats = pos_feats & odd_feats & neg_feats & eve_feats
    print(f"Universal Common Features: {sorted(list(common_feats))}")

    print("\n--- Identified Subsets (Intersections) ---")
    print(f"Universal Positive Sign Features: {sorted(list(pos_feats))}")
    print(f"Universal Odd Parity Features:    {sorted(list(odd_feats))}")
    print(f"Universal Negative Sign Features: {sorted(list(neg_feats))}")
    print(f"Universal Even Parity Features:   {sorted(list(eve_feats))}")

    pos_only_feats = pos_feats - common_feats
    neg_only_feats = neg_feats - common_feats
    odd_only_feats = odd_feats - common_feats
    eve_only_feats = eve_feats - common_feats

    print(
        f"  Distinct Even Parity Features: {sorted(list(eve_only_feats))}")
    print(
        f"  Distinct Positive Sign Features: {sorted(list(pos_only_feats))}")
    print(
        f"  Distinct Odd Parity Features: {sorted(list(odd_only_feats))}")
    print(
        f"  Distinct Negative Sign Features: {sorted(list(neg_only_feats))}")

    # 1. Create the dictionary of discovered feature IDs
    feature_subsets = {
        "Distinct Even Parity": eve_only_feats,  # Replace with your actual variable names
        "Distinct Odd Parity": odd_only_feats,
        "Distinct Positive Sign": pos_only_feats,
        "Distinct Negative Sign": neg_only_feats,
        "Universal Common": common_feats
    }

    # 2. Save the dictionary for Phase III
    torch.save(feature_subsets, "feature_subsets.pt")
    print(f"Successfully saved {len(feature_subsets)} feature groups to feature_subsets.pt")

    return {
        "pos_sign": sorted(list(pos_only_feats)),
        "neg_sign": sorted(list(neg_only_feats)),
        "odd_parity": sorted(list(odd_only_feats)),
        "even_parity": sorted(list(eve_only_feats)),
    }


class UniversalSteeringController:
    def __init__(self, mlp_path, sae_path, basis_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load MLP
        self.mlp = InterpretabilityMLP().to(self.device)
        self.mlp.load_state_dict(torch.load(mlp_path))
        self.mlp.eval()

        # Load SAE
        self.sae = SparseAutoencoder(
            input_dim=512, dict_size=2048, k=20).to(self.device)
        self.sae.load_state_dict(torch.load(sae_path))
        self.sae.eval()

        # Load Steering Basis (v_sign, v_parity)
        basis = torch.load(basis_path)
        self.v_sign = basis['v_sign'].to(self.device)
        self.v_parity = basis['v_parity'].to(self.device)

    def steer_input(self, input_tensor, target_sign=0, target_parity=0, alpha=2.0):
        with torch.no_grad():
            # 1. Get Baseline from MLP
            _ = self.mlp(input_tensor.to(self.device))
            raw_neurons = self.mlp.activations['layer2']

            # 2. Map Neurons to SAE Latents
            # If using your provided SAE definition, use self.sae.encoder or forward
            _, baseline_latents = self.sae(raw_neurons)

            # 3. Apply Meta-Steering logic
            steered_latents = baseline_latents + \
                (target_sign * alpha * self.v_sign) + \
                (target_parity * alpha * self.v_parity)

            # 4. Reconstruct to Neuron Space (the steered 'layer2')
            steered_neurons = self.sae.decoder(steered_latents)

            # 5. Manually finish the MLP forward pass using self.layers dictionary
            # We start from hidden2 because steered_neurons replaces the old layer2
            x = self.mlp.relu(self.mlp.layers['hidden2'](steered_neurons))
            output = self.mlp.layers['output'](x)

            return output.item()


def run_surgical_ablation(input_vals, target_features, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    mlp = InterpretabilityMLP().to(device)
    mlp.load_state_dict(torch.load("mlp/perfect_mlp.pth"))
    mlp.eval()

    sae = SparseAutoencoder(input_dim=512, dict_size=2048).to(device)
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
        hidden_acts = mlp.activations['layer2']
        _, latents = sae(hidden_acts)

        # 4. ABLATION: Zero out the "Hero Features"
        ablated_latents = latents.clone()
        for idx in target_features:
            ablated_latents[0, idx] = 0.0

        # 5. RECONSTRUCTION: Using the Linear layer's own weights and bias
        # This is mathematically equivalent to sae.decoder(ablated_latents)
        reconstructed_acts = sae.decoder(ablated_latents)

        # 6. FINAL PASS: Manually compute the rest of your MLP's forward pass
        # We pick up right AFTER where 'layer2' was saved

        # Pass through hidden2
        x = mlp.relu(mlp.layers['hidden2'](reconstructed_acts))

        # Pass through output
        final_out = mlp.layers['output'](x).item()

    print(f"\n--- Ablation Test: {label} ---")
    print(f"Original Output: {baseline_out:.4f}")
    print(f"Ablated Output:  {final_out:.4f}")

    shift = final_out - baseline_out
    print(f"Causal Shift:    {shift:.4f}")


if __name__ == "__main__":

    # 3. Extract Vectors
    v_sign, v_parity = get_universal_vectors(
        "harvested_data.pt", "sae/universal_sae.pth")

    # 4. Metric: Orthogonality Check
    cosine_sim = torch.nn.functional.cosine_similarity(
        v_sign.unsqueeze(0), v_parity.unsqueeze(0))
    print(f"Sign-Parity Cosine Similarity: {cosine_sim.item():.4f}")
    print("Interpretation: Near 0.0 means the concepts are perfectly disentangled.")

    torch.save({"v_sign": v_sign, "v_parity": v_parity}, "steering_basis.pt")

    dist_feat = get_distinct_features_by_group()

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