import torch
from mlp.mlp_definition import InterpretabilityMLP
from sae.sae_definition import SparseAutoencoder


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


# --- RUNTIME DEMO ---
if __name__ == "__main__":
    controller = UniversalSteeringController(
        "mlp/perfect_mlp.pth", "sae/universal_sae.pth", "steering_basis.pt")

    test_inputs = [
        ([6, 3, 0, 6, 2, 2, 8, 5, 3, 3], -3),  # Original: -3 (Negative, Odd)
        ([6, 3, 2, 7, 1, 7, 5, 7, 9, 1], -2),  # Original: -2 (Negative, Even)
        ([0, 8, 4, 6, 2, 5, 3, 8, 4, 0], -1),  # Original: -1 (Negative, Odd)
        ([8, 3, 9, 0, 0, 7, 2, 2, 7, 0], 1),  # Original: 1 (Positive, Odd)
        ([1, 3, 4, 6, 3, 4, 9, 0, 6, 0], 2),  # Original: 2 (Positive, Even)
        ([6, 5, 2, 2, 1, 7, 2, 5, 4, 1], 3)   # Original: 3 (Positive, Odd)
    ]

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
