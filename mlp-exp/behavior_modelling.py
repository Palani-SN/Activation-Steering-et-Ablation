import torch
from mlp.mlp_definition import InterpretabilityMLP
from sae.sae_definition import SparseAutoencoder
from dataset.data_loader import load_excel_to_dataloader
import pandas as pd
import ast
from torch.utils.data import DataLoader, TensorDataset


def inspect_features(mlp_path, sae_path, test_input, exp_output):
    # 1. Setup Models
    mlp = InterpretabilityMLP()
    mlp.load_state_dict(torch.load(mlp_path))
    mlp.eval()

    # Enable activations capture
    mlp.activations = {}

    sae = SparseAutoencoder(input_dim=512, dict_size=2048)
    sae.load_state_dict(torch.load(sae_path))
    sae.eval()

    # 2. Process Input through MLP
    # Convert list to tensor and add batch dimension
    input_tensor = torch.tensor([test_input], dtype=torch.float32)
    with torch.no_grad():
        output = mlp(input_tensor)
        mlp_acts = mlp.activations['layer2']

    # 3. Process MLP Activations through SAE
    with torch.no_grad():
        _, sae_hidden = sae(mlp_acts)

    # 4. Find Active Features
    # Get indices of features that are non-zero
    active_indices = torch.where(sae_hidden[0] > 0.01)[0].tolist()
    values = sae_hidden[0][active_indices].tolist()

    print("\n--- Interpretability Report ---")
    print(
        f"Sample Input: {test_input}     |     Expected Output: {exp_output}")
    print(f"MLP Output: {output.item():.4f}")
    print(f"Number of active SAE features: {len(active_indices)}")

    # Sort by activation strength
    sorted_features = sorted(zip(active_indices, values),
                             key=lambda x: x[1], reverse=True)

    print("\nTop Active Features (Monosemantic Candidates):")
    for idx, val in sorted_features[:5]:
        print(f"Feature #{idx:4} | Activation: {val:.4f}")


def generate_feature_map(mlp_path, sae_path, data_path, k):
    # 1. Setup Models
    mlp = InterpretabilityMLP()
    mlp.load_state_dict(torch.load(mlp_path))
    mlp.eval()

    sae = SparseAutoencoder(input_dim=512, dict_size=2048)
    sae.load_state_dict(torch.load(sae_path))
    sae.eval()

    # 2. Load Data
    df = pd.read_excel(data_path)
    inputs = torch.tensor([ast.literal_eval(x)
                          for x in df['input_list']], dtype=torch.float32)
    # We use the actual targets to group the features by behavior
    targets = torch.tensor([ast.literal_eval(x)
                           for x in df['output_list']], dtype=torch.float32)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=64)

    # 3. Aggregate Activations [0-9]
    # Storage for sum of activations and frequency counts per output value
    feature_sums = torch.zeros((10, 2048))
    counts = torch.zeros(10)

    print("Profiling SAE features across output groups...")
    with torch.no_grad():
        for x_batch, y_batch in loader:
            # Get MLP internal activations
            _ = mlp(x_batch)
            acts = mlp.activations['layer2']

            # Get SAE hidden features
            _, hidden_features = sae(acts)

            for i in range(len(y_batch)):
                # Round output to find the behavioral group (0-9)
                out_val = int(torch.clamp(
                    torch.round(y_batch[i]), 0, 9).item())
                feature_sums[out_val] += hidden_features[i]
                counts[out_val] += 1

    # 4. Calculate Mean Activations
    avg_activations = feature_sums / counts.unsqueeze(1).clamp(min=1)

    # 5. Extract Top Features per Group
    group_report = {}
    print("\n--- Top Features per Output Group ---")
    for group in range(10):
        # Get top k features for this numerical group
        top_vals, top_indices = torch.topk(avg_activations[group], k=k)

        group_report[group] = top_indices.tolist()

        print(f"Group {group} (n={int(counts[group].item())}):")
        feature_list = [f"#{idx} ({val:.4f})" for idx,
                        val in zip(top_indices, top_vals)]
        print(f"  Top Features: {', '.join(feature_list)}")

    return group_report


def get_pure_sets(bad_indices, good_indices):

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 5, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 5. Index is 3.
        2, 4, 4, 7, 1   # Col 2: value at 1 is 4. Index is 1.
    ]
    # Expected math: abs(5 - 4) = 1.0
    inspect_features("mlp/perfect_mlp.pth", "sae/sae_model.pth",
                     sample_input, exp_output=1.0)

    # Ensure these paths match your saved model weights
    target_map = generate_feature_map(
        mlp_path="mlp/perfect_mlp.pth",
        sae_path="sae/sae_model.pth",
        data_path="dataset/mlp_train.xlsx",
        k=50
    )

    pos_set = set()
    neg_set = set()

    for k, v in target_map.items():
        if k in good_indices:  # These are the "good" output groups
            pos_set.update(v)
        else:                  # These are the "bad" output groups
            neg_set.update(v)

    # 1. Intersection (Common elements)
    intersection_set = pos_set & neg_set  # Or set1.intersection(set2)
    print("Intersection (Common):", intersection_set)  # {4, 5}

    # 2. Purely Set 1 (Elements in 1 but not 2)
    pure_pos_set = pos_set - neg_set  # Or set1.difference(set2)
    print("Purely Positive Set 1:", pure_pos_set)  # {1, 2, 3}

    # 3. Purely Set 2 (Elements in 2 but not 1)
    pure_neg_set = neg_set - pos_set  # Or set2.difference(set1)
    print("Purely Negative Set 2:", pure_neg_set)  # {6, 7, 8}

    ablate_ids = list(pure_neg_set)
    active_ids = list(pure_pos_set)
    return ablate_ids, active_ids


def perform_steering_intervention(ablate_ids, active_ids, mlp, sae, data_loader, target_range=(3, 6)):
    mlp.eval()
    sae.eval()

    total_samples = 0
    correct_post_intervention = 0
    correct_baseline = 0
    t_min, t_max = target_range

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Baseline Accuracy
            baseline_output = mlp(inputs)
            correct_baseline += ((torch.round(baseline_output) >= t_min)
                                 & (torch.round(baseline_output) <= t_max)).sum().item()

            # Extraction
            x1 = mlp.relu(mlp.layers['bn1'](mlp.layers['input'](inputs)))
            l2_acts = mlp.relu(mlp.layers['bn2'](mlp.layers['hidden1'](x1)))
            _, latent_features = sae(l2_acts)

            # ABLATION (Static)
            # We don't ablate here to allow the "Bad" features to be used as "Down-Steer" rudders

            # ITERATIVE DUAL-STEERING
            step_size = 0.4
            for _ in range(50):
                steered_acts = sae.decoder(latent_features)
                current_output = mlp.layers['output'](
                    mlp.relu(mlp.layers['hidden2'](steered_acts)))

                # Squeeze the masks to be 1D vectors [batch_size]
                mask_low = (current_output < t_min).float().view(-1)
                mask_high = (current_output > t_max).float().view(-1)

                if (mask_low.sum() + mask_high.sum()) == 0:
                    break

                # UP-STEER: If too low, boost the "Good" features
                for idx in active_ids:
                    # latent_features[:, idx] is [32], mask_low is [32] -> Perfect match
                    latent_features[:, idx] = torch.clamp(
                        latent_features[:, idx] + (mask_low * step_size),
                        min=0.0,
                        max=15.0
                    )

                # DOWN-STEER: If too high, boost the "Bad" features
                for idx in ablate_ids:
                    latent_features[:, idx] = torch.clamp(
                        latent_features[:, idx] + (mask_high * step_size),
                        min=0.0,
                        max=15.0
                    )

            # Final Eval
            final_output = mlp.layers['output'](
                mlp.relu(mlp.layers['hidden2'](sae.decoder(latent_features))))
            correct_post_intervention += ((torch.round(final_output) >= t_min) & (
                torch.round(final_output) <= t_max)).sum().item()
            total_samples += targets.size(0)

    print(f"\nTarget Range: [{t_min}, {t_max}]")
    print(f"Baseline Acc: {(correct_baseline/total_samples)*100:.2f}%")
    print(
        f"Steered Acc:  {(correct_post_intervention/total_samples)*100:.2f}%")

# Note: This script assumes you have already trained 'mlp' and 'sae'
# and have them loaded as objects.


def run_visual_demo(mlp, sae, data_loader, ablate_ids, active_ids, target_range=(3, 6), num_samples=30):
    mlp.eval()
    sae.eval()
    t_min, t_max = target_range
    print(f"\n{'#'*80}")
    print(f"{'IDX':<4} | {'ORIGINAL':<10} | {'STEERED':<10} | {'TARGET':<8} | {'STATUS'}")
    print(f"{'-'*80}")

    count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            # We process one batch but iterate through samples for display
            x1 = mlp.relu(mlp.layers['bn1'](mlp.layers['input'](inputs)))
            l2 = mlp.relu(mlp.layers['bn2'](mlp.layers['hidden1'](x1)))
            _, latents = sae(l2)

            orig_outputs = mlp(inputs)

            # Iterative Steering for the batch
            for _ in range(50):
                out = mlp.layers['output'](
                    mlp.relu(mlp.layers['hidden2'](sae.decoder(latents))))
                m_low = (out < t_min).float().view(-1)
                m_high = (out > t_max).float().view(-1)
                if (m_low.sum() + m_high.sum()) == 0:
                    break

                for idx in active_ids:
                    latents[:, idx] += (m_low * 0.4)
                for idx in ablate_ids:
                    latents[:, idx] += (m_high * 0.4)
                latents.clamp_(0, 15)

            steered_outputs = mlp.layers['output'](
                mlp.relu(mlp.layers['hidden2'](sae.decoder(latents))))

            for j in range(inputs.size(0)):
                if count >= num_samples:
                    return
                orig_v = orig_outputs[j].item()
                steer_v = steered_outputs[j].item()

                # Logic for status labels
                if t_min <= steer_v <= t_max:
                    status = "FIXED ✅" if not (
                        t_min <= orig_v <= t_max) else "STABLE 🆗"
                else:
                    status = "FAIL ❌"

                print(
                    f"{count:<4} | {orig_v:<10.4f} | {steer_v:<10.4f} | {str(target_range):<8} | {status}")
                count += 1


if __name__ == "__main__":

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 1, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 1. Index is 3.
        2, 9, 4, 7, 1   # Col 2: value at 1 is 9. Index is 1.
    ]
    # Expected math: abs(9 - 1) = 8.0
    inspect_features("mlp/perfect_mlp.pth", "sae/sae_model.pth",
                     sample_input, exp_output=8.0)

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 2, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 2. Index is 3.
        2, 8, 4, 7, 1   # Col 2: value at 1 is 8. Index is 1.
    ]
    # Expected math: abs(2 - 8) = 6.0
    inspect_features("mlp/perfect_mlp.pth", "sae/sae_model.pth",
                     sample_input, exp_output=6.0)

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 3, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 3. Index is 3.
        2, 7, 4, 7, 1   # Col 2: value at 1 is 7. Index is 1.
    ]
    # Expected math: abs(3 - 7) = 4.0
    inspect_features("mlp/perfect_mlp.pth", "sae/sae_model.pth",
                     sample_input, exp_output=4.0)

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 4, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 4. Index is 3.
        2, 5, 4, 7, 1   # Col 2: value at 1 is 5. Index is 1.
    ]
    # Expected math: abs(4 - 5) = 1.0
    inspect_features("mlp/perfect_mlp.pth", "sae/sae_model.pth",
                     sample_input, exp_output=1.0)

    # 1. Setup Models
    # Ensure these match the dimensions defined in your uploaded files
    mlp = InterpretabilityMLP()  # input: 10, hidden1: 512
    # matches mlp hidden1
    sae = SparseAutoencoder(input_dim=512, dict_size=2048)

    # 2. Load your trained weights (Replace with your actual paths)
    mlp.load_state_dict(torch.load('mlp/perfect_mlp.pth'))
    sae.load_state_dict(torch.load('sae/sae_model.pth'))

    # 3. Prepare the Data
    # Using the 'mlp_train.xlsx' file created by your MLPExcelGenerator
    train_loader = load_excel_to_dataloader("dataset/mlp_train.xlsx")

    ablate_ids, active_ids = get_pure_sets(
        bad_indices=[0, 1, 2, 7, 8, 9],
        good_indices=[3, 4, 5, 6]
    )
    print("Ablate IDs:", ablate_ids)
    print("Active IDs:", active_ids)

    # 4. Execute the steering and ablation function
    perform_steering_intervention(
        ablate_ids, active_ids, mlp, sae, train_loader, target_range=(3, 6)
    )

    run_visual_demo(mlp, sae, train_loader, ablate_ids, active_ids)
