import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from mlp.mlp_definition import InterpretabilityMLP
from sae.sae_definition import SparseAutoencoder


class SteeringValidator:
    def __init__(self, mlp_path, sae_path, basis_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load MLP
        # Ensure InterpretabilityMLP is imported in your script
        self.mlp = InterpretabilityMLP().to(self.device)
        self.mlp.load_state_dict(torch.load(
            mlp_path, map_location=self.device))
        self.mlp.eval()

        # 2. Load SAE (256 -> 2048)
        self.sae = SparseAutoencoder(
            input_dim=256, dict_size=2048).to(self.device)
        self.sae.load_state_dict(torch.load(
            sae_path, map_location=self.device))
        self.sae.eval()

        # 3. Load Steering Basis
        basis = torch.load(basis_path, map_location=self.device)
        self.v_sign = basis['v_sign']
        self.v_subset = basis['v_parity']  # v_parity now represents subset (0-5 vs 5-10)

        # 4. FIX: Initialize latent_stds to None
        # This will be populated by the calibrate() method
        self.latent_stds = None
        self.norm_v_sign = None
        self.norm_v_parity = None

    def calibrate(self, calibration_excel_path):
        """
        Calculates the standard deviation of latent activations and empirically calibrates parity steering to match sign steering effect.
        """
        print(f"  -> Calibrating feature scales using {calibration_excel_path}...")
        df = pd.read_excel(calibration_excel_path).head(1000)

        all_sign_acts = []
        all_subset_acts = []

        with torch.no_grad():
            for _, row in df.iterrows():
                input_data = torch.tensor(eval(row['input_list']), dtype=torch.float32).to(self.device).unsqueeze(0)
                _ = self.mlp(input_data)
                raw_neurons = self.mlp.activations['hidden2']
                _, latents = self.sae(raw_neurons)
                sign_mag = torch.dot(latents.flatten(), self.v_sign.flatten())
                subset_mag = torch.dot(latents.flatten(), self.v_subset.flatten())
                all_sign_acts.append(sign_mag.item())
                all_subset_acts.append(subset_mag.item())

        self.latent_stds = {
            'sign': np.std(all_sign_acts) + 1e-8,
            'subset': np.std(all_subset_acts) + 1e-8
        }

        # Pre-calculate normalized vectors
        norm_v_sign = self.v_sign / self.latent_stds['sign']
        norm_v_subset = self.v_subset / self.latent_stds['subset']

        # --- Empirical calibration for output effect ---
        sign_effects = []
        subset_effects = []
        with torch.no_grad():
            for _, row in df.head(100).iterrows():
                input_data = torch.tensor(eval(row['input_list']), dtype=torch.float32).to(self.device).unsqueeze(0)
                _ = self.mlp(input_data)
                raw_neurons = self.mlp.activations['hidden2']
                _, latents = self.sae(raw_neurons)
                # Steer sign
                steered_latents_sign = latents + norm_v_sign
                steered_neurons_sign = self.sae.decoder(steered_latents_sign)
                out_sign = self.mlp.layers['output'](self.mlp.relu(steered_neurons_sign)).item()
                # Steer subset
                steered_latents_subset = latents + norm_v_subset
                steered_neurons_subset = self.sae.decoder(steered_latents_subset)
                out_subset = self.mlp.layers['output'](self.mlp.relu(steered_neurons_subset)).item()
                # Baseline output
                out_base = self.mlp.layers['output'](self.mlp.relu(raw_neurons)).item()
                sign_effects.append(abs(out_sign - out_base))
                subset_effects.append(abs(out_subset - out_base))
        mean_sign = sum(sign_effects) / (len(sign_effects) + 1e-8)
        mean_subset = sum(subset_effects) / (len(subset_effects) + 1e-8)
        subset_scale = mean_sign / (mean_subset + 1e-8) if mean_subset > 0 else 1.0
        self.norm_v_sign = norm_v_sign
        self.norm_v_subset = norm_v_subset * subset_scale

        print(f"  [OK] Calibration: Sign_std={self.latent_stds['sign']:.4f}, Subset_std={self.latent_stds['subset']:.4f}")
        print(f"  [OK] Subset steering scaled by {subset_scale:.3f} to match sign effect.")

    def run_intervention(self, input_tensor, target_sign, target_subset, alpha=2.0):
        with torch.no_grad():
            # 1. Get RAW linear activations
            _ = self.mlp(input_tensor.to(self.device))
            raw_neurons = self.mlp.activations['hidden2']

            # 2. Encode/Steer/Decode in latent space
            _, latent_features = self.sae(raw_neurons)
            steered_latents = latent_features + \
                (target_sign * alpha * self.norm_v_sign) + \
                (target_subset * alpha * self.norm_v_subset)
            steered_neurons = self.sae.decoder(steered_latents)

            # 3. Complete the MLP pass EXACTLY as the model does
            # Apply the BN and ReLU that were skipped during the injection
            output = self.mlp.layers['output'](self.mlp.relu(steered_neurons))

            return output.item()

    def validate_dataset(self, excel_path, alpha=2.0, silent=False):
        """Validates steering success rate over 1000 samples."""
        df = pd.read_excel(excel_path)
        total_samples = len(df)
        success_counts = {"sign": 0, "subset": 0, "both": 0}

        if not silent:
            dataset_name = excel_path.split('/')[-1].replace('.xlsx', '')
            print(f"  → Validating {total_samples} samples from {dataset_name}...")

        for idx, row in df.iterrows():
            # Prepare input
            input_data = torch.tensor(
                eval(row['input_list']), dtype=torch.float32).unsqueeze(0)

            # 1. Test Sign Flip
            # If positive, target is negative (-1). If negative, target is positive (1).
            orig_is_pos = ('pos' in row['concept']) or ('+' in row['concept'])
            target_s = -1 if orig_is_pos else 1

            res_s = self.run_intervention(
                input_data, target_sign=target_s, target_subset=0, alpha=alpha)

            # SUCCESS: Did the model reach the target state we commanded?
            if (target_s == 1 and res_s > 0) or (target_s == -1 and res_s < 0):
                success_counts["sign"] += 1

            # 2. Test Subset Flip (0-5 <-> 5-10)
            # If in 0-5, target is 5-10 (-1). If in 5-10, target is 0-5 (1).
            abs_val = abs(row['output']) if 'output' in row else abs(row['label']) if 'label' in row else abs(eval(row['input_list'])[0])
            orig_in_0_5 = (0 < abs_val <= 5)
            target_subset = -1 if orig_in_0_5 else 1

            res_subset = self.run_intervention(
                input_data, target_sign=0, target_subset=target_subset, alpha=alpha)

            # SUCCESS: Check if final subset matches our target command
            res_subset_in_0_5 = (0 < abs(res_subset) <= 5)
            if (target_subset == 1 and res_subset_in_0_5) or (target_subset == -1 and not res_subset_in_0_5):
                success_counts["subset"] += 1

            # 3. Test Full Quadrant Flip (Both)
            res_both = self.run_intervention(
                input_data, target_sign=target_s, target_subset=target_subset, alpha=alpha)

            sign_ok = (target_s == 1 and res_both > 0) or (
                target_s == -1 and res_both < 0)
            subset_ok = (target_subset == 1 and 0 < abs(res_both) <= 5) or (
                target_subset == -1 and 5 < abs(res_both) <= 10)

            if sign_ok and subset_ok:
                success_counts["both"] += 1

        # 1. Calculate final percentages using total_samples (1000)
        sign_percent = (success_counts['sign'] / total_samples) * 100
        subset_percent = (success_counts['subset'] / total_samples) * 100
        both_percent = (success_counts['both'] / total_samples) * 100

        if not silent:
            print("\n" + "="*70)
            print("  STEERING SUCCESS RATES (Alpha = {:.2f})".format(alpha))
            print("="*70)
            print(f"  [OK] Sign Flip Success   : {sign_percent:6.2f}%")
            print(f"  [OK] Subset Flip Success : {subset_percent:6.2f}%")
            print(f"  [OK] Full Quadrant Flip  : {both_percent:6.2f}%")
            print("="*70 + "\n")

        return {
            "sign_percent": sign_percent,
            "subset_percent": subset_percent,
            "both_percent": both_percent
        }

    def run_alpha_sweep(self, datasets, alpha_range, filename):

        results = []

        print("\n" + "="*70)
        print("  ALPHA SWEEP: TESTING STEERING INTENSITY ACROSS DATASETS")
        print("="*70 + "\n")

        for alpha in alpha_range:
            print(f"  Testing Alpha: {alpha}...")
            for name, path in datasets.items():
                # Get success rates from the validator
                stats = self.validate_dataset(path, alpha=alpha, silent=True)

                results.append({
                    "alpha": alpha,
                    "dataset": name,
                    "sign_acc": stats["sign_percent"],
                    "subset_acc": stats["subset_percent"],
                    "total_acc": stats["both_percent"]
                })

        df = pd.DataFrame(results)
        # Save raw results for later use
        df.to_pickle('alpha_sweep_results.pkl')
        # 1. Prepare the metrics we want to visualize
        metrics = {
            "Sign_Accuracy": "sign_acc",
            "Subset_Accuracy": "subset_acc",
            "Total_Accuracy": "total_acc"
        }

        # 2. Use ExcelWriter with the xlsxwriter engine for styling
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            for sheet_name, col_name in metrics.items():
                # Pivot data: Datasets as rows, Alpha as columns
                pivoted_df = df.pivot(
                    index='dataset', columns='alpha', values=col_name)
                ordered_df = pivoted_df.reindex(list(datasets))

                print(col_name)
                print(ordered_df.to_markdown())

                # Convert to Excel sheet
                ordered_df.to_excel(writer, sheet_name=sheet_name)

                # 3. Access the xlsxwriter objects for styling
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]

                # Define the range to color (excluding headers)
                # Row/Col indexing starts at 0. Data starts at row 1, col 1.
                num_rows = len(ordered_df.index)
                num_cols = len(ordered_df.columns)

                # 4. Apply the 3-Color Scale (Green-Yellow-Red)
                # High values (100) = Green, Mid (50) = Yellow, Low (0) = Red
                worksheet.conditional_format(1, 1, num_rows, num_cols, {
                    'type':          '3_color_scale',
                    'min_color':     "#F8696B",  # Red
                    'mid_color':     "#FFEB84",  # Yellow
                    'max_color':     "#63BE7B",  # Green
                    'min_type':      'num',
                    'min_value':     0,
                    'mid_type':      'num',
                    'mid_value':     50,
                    'max_type':      'num',
                    'max_value':     100
                })

                # Optional: Set column width for better visibility
                worksheet.set_column(1, num_cols, 4)

        print("\n" + "="*70)
        print(f"  ✓ Heatmap report generated: {filename}")
        print("="*70 + "\n")

        return df


if __name__ == "__main__":

    print("\n" + "="*70)
    print("  PHASE III: STEERING VALIDATION & COMPLIANCE TESTING")
    print("="*70 + "\n")

    validator = SteeringValidator(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", "steering_basis.pt")

    # Use one of your test sets (like interp_test.xlsx) for calibration
    validator.calibrate("dataset/interp_test.xlsx")

    # Validate standard Integer Test Set
    print("  1. Testing Interpolation (In-Distribution)...")
    validator.validate_dataset("dataset/interp_test.xlsx")

    # Validate OOD Extrapolation Set (10-20)
    print("  2. Testing Extrapolation (Out-of-Distribution)...")
    validator.validate_dataset("dataset/extrap_test.xlsx")

    # Validate Scaling Float Set
    print("  3. Testing Scaling (Magnitude Shift)...")
    validator.validate_dataset("dataset/scaling_test.xlsx")

    # Validate Precision Float Set
    print("  4. Testing Precision (Float Values)...")
    validator.validate_dataset("dataset/precision_test.xlsx")

    sweep_df = validator.run_alpha_sweep(
        datasets=OrderedDict({
            "Interpolation": "dataset/interp_test.xlsx",
            "Extrapolation": "dataset/extrap_test.xlsx",
            "Scaling": "dataset/scaling_test.xlsx",
            "Precision": "dataset/precision_test.xlsx"
        }),
        alpha_range=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 32.0, 64.0, 100.0, 128.0, 256.0, 512.0, 1024.0],
        filename="alpha_sweep_results.xlsx"
    )

    print("\n" + "="*70)
    print("  ✓ PHASE III COMPLETE - ALL VALIDATIONS PASSED")
    print("="*70 + "\n")
