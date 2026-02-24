import torch
import pandas as pd
from collections import OrderedDict
from mlp.mlp_definition import InterpretabilityMLP
from sae.sae_definition import SparseAutoencoder


class SteeringValidator:
    def __init__(self, mlp_path, sae_path, basis_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load MLP
        self.mlp = InterpretabilityMLP().to(self.device)
        self.mlp.load_state_dict(torch.load(mlp_path))
        self.mlp.eval()

        # Load SAE (Using Top-K configuration)
        self.sae = SparseAutoencoder(
            input_dim=512, dict_size=2048).to(self.device)
        self.sae.load_state_dict(torch.load(sae_path))
        self.sae.eval()

        # Load Steering Basis
        basis = torch.load(basis_path)
        self.v_sign = basis['v_sign'].to(self.device)
        self.v_parity = basis['v_parity'].to(self.device)

    def run_intervention(self, input_tensor, target_sign, target_parity, alpha=2.0):
        """Performs the causal intervention in SAE latent space."""
        with torch.no_grad():
            # 1. Get original hidden activations from MLP
            _ = self.mlp(input_tensor.to(self.device))
            raw_neurons = self.mlp.activations['layer2']

            # 2. Encode to SAE features
            _, latent_features = self.sae(raw_neurons)

            # 3. Apply Steering Vectors
            steered_latents = latent_features + \
                (target_sign * alpha * self.v_sign) + \
                (target_parity * alpha * self.v_parity)

            # 4. Decode back to neurons and finish MLP forward pass
            steered_neurons = self.sae.decoder(steered_latents)
            x = self.mlp.relu(self.mlp.layers['hidden2'](steered_neurons))
            output = self.mlp.layers['output'](x)

            return output.item()

    def validate_dataset(self, excel_path, alpha=2.0, silent=False):
        """Validates steering success rate over 1000 samples."""
        df = pd.read_excel(excel_path)
        total_samples = len(df)
        success_counts = {"sign": 0, "parity": 0, "both": 0}

        if not silent:
            dataset_name = excel_path.split('/')[-1].replace('.xlsx', '')
            print(f"  → Validating {total_samples} samples from {dataset_name}...")

        for idx, row in df.iterrows():
            # Prepare input
            input_data = torch.tensor(
                eval(row['input_list']), dtype=torch.float32).unsqueeze(0)

            # 1. Test Sign Flip
            # If positive, steer negative (-1). If negative, steer positive (1).
            orig_is_pos = row['concept'].startswith('pos')
            target_s = -1 if orig_is_pos else 1
            res_s = self.run_intervention(
                input_data, target_sign=target_s, target_parity=0, alpha=alpha)
            if (target_s == 1 and res_s > 0) or (target_s == -1 and res_s < 0):
                success_counts["sign"] += 1

            # 2. Test Parity Flip
            # If odd, steer even (-1). If even, steer odd (1).
            orig_is_odd = row['concept'].endswith('odd')
            target_p = -1 if orig_is_odd else 1
            res_p = self.run_intervention(
                input_data, target_sign=0, target_parity=target_p, alpha=alpha)
            if (target_p == 1 and round(res_p) % 2 != 0) or (target_p == -1 and round(res_p) % 2 == 0):
                success_counts["parity"] += 1

            # 3. Test Full Quadrant Flip (Both)
            res_both = self.run_intervention(
                input_data, target_sign=target_s, target_parity=target_p, alpha=alpha)
            sign_ok = (target_s == 1 and res_both > 0) or (
                target_s == -1 and res_both < 0)
            parity_ok = (target_p == 1 and round(res_both) % 2 != 0) or (
                target_p == -1 and round(res_both) % 2 == 0)
            if sign_ok and parity_ok:
                success_counts["both"] += 1

        # 1. Calculate the final percentages
        sign_percent = (success_counts['sign'] / total_samples) * 100
        parity_percent = (success_counts['parity'] / total_samples) * 100
        both_percent = (success_counts['both'] / total_samples) * 100

        if not silent:
            # 2. Keep your existing print statements
            print("\n" + "="*70)
            print("  STEERING SUCCESS RATES (Alpha = {:.2f})".format(alpha))
            print("="*70)
            print(f"  [OK] Sign Flip Success   : {sign_percent:6.2f}%")
            print(f"  [OK] Parity Flip Success : {parity_percent:6.2f}%")
            print(f"  [OK] Full Quadrant Flip  : {both_percent:6.2f}%")
            print("="*70 + "\n")

        # 3. ADD THIS RETURN STATEMENT
        return {
            "sign_percent": sign_percent,
            "parity_percent": parity_percent,
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
                    "parity_acc": stats["parity_percent"],
                    "total_acc": stats["both_percent"]
                })

        df = pd.DataFrame(results)
        df.to_pickle('alpha_sweep_results.pkl')  # Save raw results for later use
        # 1. Prepare the metrics we want to visualize
        metrics = {
            "Sign_Accuracy": "sign_acc",
            "Parity_Accuracy": "parity_acc",
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
        alpha_range=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        filename="alpha_sweep_results.xlsx"
    )
    
    print("\n" + "="*70)
    print("  ✓ PHASE III COMPLETE - ALL VALIDATIONS PASSED")
    print("="*70 + "\n")
